"""Existing GRU-D/TCN and verified official MTM, new matched daily inputs."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from . import sensor_models as m
from .representations import gaps, validate_schedule
from mindwatch.models import person_weights
from .neural import GRUDEncoder, MaskAwareTCN
from .mtm_adapter import load_official_mtm, verify_mtm_source


def _values(v, rows, channels=None):
    v = np.asarray(v, np.float32)
    if (v.ndim != 3 or len(v) != rows or min(v.shape[1:]) < 1 or
            np.isinf(v).any() or (channels is not None and v.shape[2] != channels)):
        raise ValueError('Aligned finite-or-missing sensor sequences required')
    return v


def _scores(values, shape, name):
    if values is None:
        raise ValueError(f'Explicit {name} required')
    values = np.asarray(values, float)
    if (values.shape != shape or np.isinf(values).any() or
            np.any(np.isfinite(values) & ((values < 0) | (values > 10)))):
        raise ValueError(f'{name} must be aligned finite-or-missing 0 to 10 scores')
    return values


def _loss(prediction, target, weight, task, normalizer):
    error = (nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none')
             if task == 'phq' else abs(prediction-target))
    return (error*weight).sum()/(len(prediction)*normalizer)


class Core(nn.Module):
    def __init__(self, arch, nf, ns, source_dir=None):
        super().__init__()
        self.arch = arch
        if arch == 'mtm':
            verify_mtm_source(source_dir)
            cls = load_official_mtm(source_dir)
            self.encoder = cls(num_chn=nf, d_static=ns, num_cls=1, ratios=[2, 2], d_model=64,
                               r_hid=2, drop=.2, norm_first=True, down_mode='concat')
        else:
            self.encoder = GRUDEncoder(nf, 64) if arch == 'grud' else MaskAwareTCN(nf, 64, .2)
            self.head = nn.Sequential(nn.Linear(64+ns, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 1))

    def forward(self, x, mask, delta, static):
        if self.arch == 'mtm':
            times = torch.arange(x.shape[1], device=x.device).expand(len(x), -1)
            return self.encoder(x, ~mask.bool(), times, static).reshape(-1)
        return self.head(torch.cat([self.encoder(x, mask, delta), static], dim=1)).reshape(-1)


@dataclass
class NeuralModel:
    task: str
    spec: dict
    base: m.Head
    mean: np.ndarray
    scale: np.ndarray
    seen: np.ndarray
    static_mean: np.ndarray
    static_scale: np.ndarray
    states: list
    trace: list
    sensor_cols: tuple = ()

    def arrays(self, x, v):
        v = _values(v, len(x), len(self.seen))
        if v.shape[1] != self.spec['sequence_length']:
            raise ValueError('Sequence length changed after fitting')
        mask = np.isfinite(v) & self.seen[None, None, :]
        z = np.where(mask, (v-self.mean)/self.scale, 0).astype(np.float32)
        delta = gaps(mask)
        delta[:, :, ~self.seen] = 0
        if self.task == 'phq':
            a = x[list(self.base.cols)].to_numpy(dtype=float, copy=True)
            if np.isinf(a).any():
                raise ValueError('Infinite static values are not missing observations')
            a[:, ~self.base.seen] = np.nan
            static = np.nan_to_num((a-self.static_mean)/self.static_scale).astype(np.float32)
        else:
            static = np.empty((len(x), 0), np.float32)
        return z, mask.astype(np.float32), delta, static

    def raw(self, x, v, p0=None):
        if any(not n.startswith(('h_', 'e_', 's_')) for n in x):
            raise ValueError('Current targets or metadata passed to neural model')
        z, mask, delta, static = self.arrays(x, v)
        base = (_scores(p0, (len(x),), 'strictly prior P0 forecasts') if self.task == 'ema'
                else self.base.predict(x[list(self.base.cols)]) if len(x) else np.empty(0))
        if not len(x):
            return base.copy()
        if not self.states:
            raise ValueError('No fitted neural states')
        pred = []
        for state in self.states:
            net = Core(self.spec['architecture'], len(self.seen), static.shape[-1], self.spec.get('mtm_source_dir'))
            net.load_state_dict(state)
            net.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(x), 256):
                    parts.append(net(*[torch.tensor(a[start:start+256]) for a in [z, mask, delta, static]]).numpy())
            q = np.concatenate(parts)
            pred.append(1/(1+np.exp(-np.clip(q, -30, 30))) if self.task == 'phq' else q)
        p = np.mean(pred, axis=0) if self.task == 'phq' else np.median(pred, axis=0)
        quality = np.minimum(1, mask.mean(axis=(1, 2))*4)
        available = mask.any(axis=(1, 2))
        q = base+np.where(available, quality*.5*(p-base if self.task == 'phq' else p), 0)
        clipped = np.clip(q, 1e-7, 1-1e-7) if self.task == 'phq' else np.clip(q, 0, 10)
        return np.where(available, clipped, base)


def fit(f, x, v, y, task, domain, arch, *, p0_values=None, base_columns=None,
        mtm_source_dir=None, epochs=40, seeds=(42, 2026, 2027)):
    if task not in ['phq', 'ema'] or arch not in ['grud', 'tcn', 'mtm']:
        raise ValueError('Unknown neural task/architecture')
    seeds = validate_schedule(epochs, seeds)
    if (not isinstance(f, pd.DataFrame) or not isinstance(x, pd.DataFrame) or
            len(f) != len(x) or not f.index.equals(x.index) or 'subject_id' not in f or
            x.columns.duplicated().any() or any(not n.startswith(('h_', 'e_', 's_')) for n in x)):
        raise ValueError('Aligned participant metadata and historical features required')
    v = _values(v, len(x))
    if arch == "mtm" and v.shape[1] > 638:
        raise ValueError("Official MTM requires at most 638 time bins")
    y = np.asarray(y, float)
    if y.shape != (len(x),) or np.isinf(y).any():
        raise ValueError('Aligned finite-or-missing labels required')
    if task == 'phq' and not np.isin(y[np.isfinite(y)], [0, 1]).all():
        raise ValueError('PHQ classification labels must be zero or one')
    good = np.isfinite(y)
    if task == 'ema':
        y = _scores(y, (len(x),), 'EMA labels')
        p0_values = _scores(p0_values, y.shape, 'strictly prior P0 forecasts')
        good &= np.isfinite(p0_values)
    if not good.any():
        raise ValueError('No usable training target')
    if base_columns is None:
        base_columns = tuple(n for n in x if n.startswith('e_'))
    if len(set(base_columns)) != len(base_columns) or not set(base_columns) <= set(x):
        raise ValueError('Invalid baseline feature schema')
    f = f.loc[good].reset_index(drop=True)
    x = x.loc[good].reset_index(drop=True)
    v, y = v[good], y[good]
    base = (m.fit_head(x[list(base_columns)], y, f.subject_id, 'phq') if task == 'phq'
            else m.Head(tuple(), np.array([], bool), None, 0, 'ema'))
    mask = np.isfinite(v)
    w = person_weights(f.subject_id)
    den = (mask*w[:, None, None]).sum((0, 1))
    mean = (np.nan_to_num(v)*w[:, None, None]).sum((0, 1))/np.maximum(den, 1e-12)
    scale = np.sqrt((np.where(mask, (v-mean)**2, 0)*w[:, None, None]).sum((0, 1))/np.maximum(den, 1e-12))
    scale[scale < 1e-6] = 1
    if task == 'phq':
        s = x[list(base.cols)].to_numpy(float)
        sm = np.array([np.nanmedian(s[:, j]) if base.seen[j] else 0 for j in range(s.shape[1])])
        ss = np.array([np.nanstd(s[:, j]) if base.seen[j] else 1 for j in range(s.shape[1])])
        ss[~np.isfinite(ss) | (ss < 1e-6)] = 1
    else:
        sm = ss = np.empty(0)
    result = NeuralModel(task, {'kind': 'neural', 'architecture': arch, 'view': 'daily', 'alpha': .5,
                               'mtm_source_dir': mtm_source_dir, 'sequence_length': v.shape[1]},
                         base, mean, scale, den > 0, sm, ss, [], [])
    arrays = result.arrays(x, v)
    target = y if task == 'phq' else y-p0_values[good]
    eligible = np.flatnonzero(mask.any((1, 2)))
    if not len(eligible):
        raise ValueError('No supported training sensor window')
    w = np.zeros(len(f), float)
    w[eligible] = person_weights(f.subject_id.iloc[eligible])
    normalizer = float(w[eligible].mean())
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Core(arch, v.shape[-1], len(sm), mtm_source_dir)
        if arch != 'mtm':
            net.encoder.feature_observed.fill_(True)
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
        for epoch in range(epochs):
            order = rng.permutation(eligible)
            net.train()
            total = 0.
            for start in range(0, len(order), 64):
                ix = order[start:start+64]
                p = net(*[torch.tensor(a[ix]) for a in arrays])
                yy, ww = torch.tensor(target[ix], dtype=torch.float32), torch.tensor(w[ix], dtype=torch.float32)
                loss = _loss(p, yy, ww, task, normalizer)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite neural loss')
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
                opt.step()
                total += float(loss.detach())*len(ix)
            if epoch in [0, epochs-1]:
                result.trace.append({'seed': seed, 'epoch': epoch+1, 'loss': total/len(order)})
        result.states.append({k: a.detach().cpu() for k, a in net.state_dict().items()})
    return result
