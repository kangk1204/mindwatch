"""Four-domain EMA sensor residual heads; no private cache or fitted states bundled.

Callers supply training-only targets and strictly prior P0 predictions explicitly.
An unsupervised domain uses its supplied P0 exactly; it never publishes a random head.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from mindwatch.models import person_weights
from .neural import GRUDEncoder, MaskAwareTCN
from .mtm_adapter import load_official_mtm, verify_mtm_source
from .representations import gaps, validate_schedule
from .neural_comparators import _values, _scores

DOMAINS = ('ema_depression', 'ema_anxiety', 'ema_sleep', 'ema_stress')


def _loss(prediction, target, weight, normalizers, supervised):
    per_domain = (abs(prediction-target)*weight).sum(0)/(len(prediction)*normalizers).clamp(min=1e-12)
    return per_domain[supervised].mean()


class JointCore(nn.Module):
    def __init__(self, arch, nf, source_dir=None):
        super().__init__()
        self.arch = arch
        if arch == 'mtm':
            verify_mtm_source(source_dir)
            cls = load_official_mtm(source_dir)
            self.encoder = cls(num_chn=nf, d_static=0, num_cls=4, ratios=[2, 2], d_model=64,
                               r_hid=2, drop=.2, norm_first=True, down_mode='concat')
        else:
            self.encoder = GRUDEncoder(nf, 64) if arch == 'grud' else MaskAwareTCN(nf, 64, .2)
            self.encoder.feature_observed.fill_(True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 4))

    def forward(self, x, mask, delta):
        if self.arch == 'mtm':
            with torch.sparse.check_sparse_tensor_invariants(enable=True):
                return self.encoder(x, ~mask.bool(), torch.arange(x.shape[1], device=x.device).expand(len(x), -1), x.new_empty(len(x), 0))
        return self.head(self.encoder(x, mask, delta))


@dataclass
class JointModel:
    spec: dict
    mean: np.ndarray
    scale: np.ndarray
    seen: np.ndarray
    states: list
    trace: list
    domain: str
    supervised: np.ndarray | None = None
    task: str = 'ema'
    sensor_cols: tuple = ()

    def arrays(self, v):
        v = np.asarray(v, np.float32)
        v = _values(v, len(v), len(self.seen))
        if v.shape[1] != self.spec['sequence_length']:
            raise ValueError('Sequence length changed after fitting')
        mask = np.isfinite(v) & self.seen[None, None, :]
        a = np.where(mask, (v-self.mean)/self.scale, 0).astype(np.float32)
        delta = gaps(mask)
        delta[:, :, ~self.seen] = 0
        return a, mask.astype(np.float32), delta

    def raw(self, x, v, p0=None):
        if any(not n.startswith(('h_', 'e_', 's_')) for n in x):
            raise ValueError('Current targets forbidden')
        _values(v, len(x), len(self.seen))
        arrays = self.arrays(v)
        base = _scores(p0, (len(x),), 'strictly prior P0 forecasts')
        if self.supervised is None or np.asarray(self.supervised).shape != (4,):
            raise ValueError('Fitted domain supervision metadata required')
        if not len(x) or not self.supervised[DOMAINS.index(self.domain)]:
            return base.copy()
        if not self.states:
            raise ValueError('No fitted joint neural states')
        pred = []
        for state in self.states:
            net = JointCore(self.spec['architecture'], len(self.seen), self.spec.get('mtm_source_dir'))
            net.load_state_dict(state)
            net.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(x), 256):
                    parts.append(net(*[torch.tensor(a[start:start+256]) for a in arrays]).numpy())
            pred.append(np.concatenate(parts)[:, DOMAINS.index(self.domain)])
        correction = np.median(pred, axis=0)
        mask = arrays[1]
        quality = np.minimum(1, mask.mean((1, 2))*4)
        available = mask.any((1, 2))
        return np.where(available, np.clip(base+np.where(available, .5*quality*correction, 0), 0, 10), base)


def fit(groups, values, targets, p0_values, domain, architecture='grud', *,
        epochs=40, seeds=(42, 2026, 2027), mtm_source_dir=None):
    if domain not in DOMAINS or architecture not in ('grud', 'tcn', 'mtm'):
        raise ValueError('Unknown EMA domain or architecture')
    seeds = validate_schedule(epochs, seeds)
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError('One participant key per training row required')
    values = _values(values, len(groups))
    if architecture == "mtm" and values.shape[1] > 638:
        raise ValueError("Official MTM requires at most 638 time bins")
    targets = _scores(targets, (len(values), 4), 'EMA labels')
    p0_values = _scores(p0_values, targets.shape, 'strictly prior P0 forecasts')
    y = targets-p0_values
    valid, mask = np.isfinite(y), np.isfinite(values)
    weights = person_weights(groups)[:, None, None]
    den = (mask*weights).sum((0, 1))
    mean = (np.nan_to_num(values)*weights).sum((0, 1))/np.maximum(den, 1e-12)
    scale = np.sqrt((np.where(mask, (values-mean)**2, 0)*weights).sum((0, 1))/np.maximum(den, 1e-12))
    scale[scale < 1e-6] = 1
    eligible = mask.any((1, 2))
    fitrows = np.flatnonzero(valid.any(1) & eligible)
    if not len(fitrows):
        raise ValueError('No observed training window with a target')
    weight = np.zeros(y.shape, np.float32)
    for j in range(4):
        selected = valid[:, j] & eligible
        if selected.any():
            # Equal total weight for every participant observed in this domain.
            _, inverse, counts = np.unique(groups[selected], return_inverse=True, return_counts=True)
            weight[selected, j] = 1./counts[inverse]
    supervised = weight.sum(0) > 0
    normalizers = torch.tensor(weight.sum(0)/len(fitrows))
    active = torch.tensor(supervised)
    result = JointModel({'architecture': architecture, 'mtm_source_dir': mtm_source_dir,
                         'sequence_length': values.shape[1]}, mean, scale, den > 0, [], [], domain, supervised)
    arrays = result.arrays(values)
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = JointCore(architecture, values.shape[-1], mtm_source_dir)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
        for epoch in range(epochs):
            net.train()
            order = rng.permutation(fitrows)
            total = 0.
            for start in range(0, len(order), 64):
                ix = order[start:start+64]
                prediction = net(*[torch.tensor(a[ix]) for a in arrays])
                target = torch.tensor(np.nan_to_num(y[ix]), dtype=torch.float32)
                w = torch.tensor(weight[ix])
                loss = _loss(prediction, target, w, normalizers, active)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite joint EMA loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ix)
            if epoch in (0, epochs-1):
                result.trace.append({'seed': seed, 'epoch': epoch+1, 'L1_loss': total/len(order),
                                     'supervised_domains': supervised.tolist(),
                                     'objective': 'equal-domain participant-total weighted L1'})
        result.states.append({key: value.detach().cpu() for key, value in net.state_dict().items()})
    return result
