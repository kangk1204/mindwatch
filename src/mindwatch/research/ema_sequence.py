"""Prespecified, CPU, four-output residual GRU comparator for observed EMA.

The caller supplies strictly prior, oldest-to-newest EMA events (8 x 4), prior
sensor days (14 x 7), released history, and the earlier same-domain EMA value.
This module cannot establish chronology from arrays and does not select rows,
read files, tune on validation data, or infer clinical meaning for 0--10 scores.

C = persistence (or weighted TRAIN target median) + H/E correction. D freezes C
and adds a sensor correction multiplied by its observed fraction. The latter
fits residuals from C's OWN TRAIN predictions, not out-of-fold predictions.
Both correction heads start at zero. L1 estimates conditional medians; the
three-seed pointwise median is a prespecified ensemble, not uncertainty bounds.
No clipping enters the training gradient; returned predictions are clipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


DEFAULT_HISTORY_NAMES = ("h_phq9", "h_gad7", "h_isi", "h_days",
                         "h_initial_phq9", "h_since_initial")
DEFAULT_SEEDS = (42, 2026, 2027)
_FORBIDDEN_NAMES = {"subject_id", "sample_id", "date", "index_date", "phq9",
                    "gad7", "isi", "ema_depression", "ema_anxiety", "ema_sleep",
                    "ema_stress"}


def _history_schema(names, allowed):
    names, allowed = tuple(names), tuple(allowed)
    if len(names) != len(set(names)) or len(allowed) != len(set(allowed)):
        raise ValueError("History schema must be unique")
    if any(not isinstance(n, str) or n in _FORBIDDEN_NAMES or
           n.lower().startswith(("target", "current", "y_")) for n in names):
        raise ValueError("Current targets and identifiers cannot be history inputs")
    if not set(names).issubset(allowed):
        raise ValueError("History names are outside the exact approved allowlist")
    return names


@dataclass
class SeqBatch:
    ema_values: np.ndarray
    ema_mask: np.ndarray
    ema_delta: np.ndarray
    history: np.ndarray
    history_names: tuple[str, ...]
    persistence: np.ndarray
    sensor_values: np.ndarray
    sensor_mask: np.ndarray
    sensor_delta: np.ndarray
    sensor_available: np.ndarray

    def __len__(self):
        return len(self.ema_values)

    def subset(self, indices):
        idx = np.asarray(indices)
        return SeqBatch(**{k: v if k == "history_names" else np.asarray(v)[idx]
                           for k, v in vars(self).items()})

    def validate(self, allowed_history_names):
        n = len(self)
        _history_schema(self.history_names, allowed_history_names)
        if np.asarray(self.history).shape != (n, len(self.history_names)):
            raise ValueError("History shape differs from its declared schema")
        if np.isinf(np.asarray(self.history)[_history_eligible(self)]).any():
            raise ValueError("History cannot contain infinity")
        if np.asarray(self.persistence).shape != (n, 4) or np.isinf(self.persistence).any():
            raise ValueError("Persistence must be [N,4] with finite values or NaNs")
        p = np.asarray(self.persistence)
        if np.any(np.isfinite(p) & ((p < 0) | (p > 10))):
            raise ValueError("Persistence is outside the recorded 0--10 scale")
        a = np.asarray(self.sensor_available)
        if a.shape != (n,) or not np.isin(a, [False, True]).all():
            raise ValueError("sensor_available must be binary [N]")
        for kind, shape in (("ema", (n, 8, 4)), ("sensor", (n, 14, 7))):
            x, m, d = [np.asarray(getattr(self, kind + "_" + s))
                       for s in ("values", "mask", "delta")]
            if x.shape != shape or m.shape != shape or d.shape != shape:
                raise ValueError(f"{kind} arrays have incorrect shapes")
            if not np.isin(m, [0, 1]).all():
                raise ValueError(f"{kind} mask must be binary")
            active = np.ones(n, bool) if kind == "ema" else a.astype(bool)
            effective = m.astype(bool) & active[:, None, None]
            if not np.isfinite(x[effective]).all():
                raise ValueError(f"Observed {kind} values must be finite")
            if not np.isfinite(d[active]).all() or (d[active] < 0).any():
                raise ValueError(f"{kind} gaps must be finite and nonnegative")


def _history_eligible(batch):
    """Required history gates, when present in the declared exact schema.

    A missing/invalid initial questionnaire or unreleased date cannot be rescued
    by learned imputation or the training-median persistence fallback.
    """
    history = np.asarray(batch.history, float)
    names = tuple(batch.history_names)
    if history.shape != (len(batch), len(names)):
        raise ValueError("History shape differs from its declared schema")
    eligible = np.ones(len(batch), bool)
    if "h_initial_phq9" in names:
        eligible &= np.isfinite(history[:, names.index("h_initial_phq9")])
    if "h_days" in names:
        days = history[:, names.index("h_days")]
        eligible &= np.isfinite(days) & (days >= 1)
    return eligible


def _person_weights(groups):
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) == 0 or any(v is None for v in g):
        raise ValueError("Nonempty participant groups are required")
    if np.issubdtype(g.dtype, np.number) and not np.isfinite(g).all():
        raise ValueError("Participant groups cannot be missing")
    _, inverse, counts = np.unique(g, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _label_weights(y, groups):
    """For every observed domain, each observed participant has total weight 1."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    valid = np.isfinite(y)
    w = np.zeros(y.shape, float)
    for j in range(4):
        counts = np.bincount(inverse, weights=valid[:, j], minlength=inverse.max() + 1)
        w[:, j] = np.divide(valid[:, j], counts[inverse],
                            out=np.zeros(len(y)), where=counts[inverse] > 0)
    return w


def _weighted_median(x, w):
    keep = np.isfinite(x) & (w > 0)
    if not keep.any():
        return float("nan")
    order = np.argsort(x[keep], kind="stable")
    values, weights = x[keep][order], w[keep][order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")])


@dataclass
class _SequenceScale:
    mean: np.ndarray
    scale: np.ndarray
    observed: np.ndarray

    @classmethod
    def fit(cls, x, m, weights):
        x, m = np.asarray(x, float), np.asarray(m, bool)
        w = m * np.asarray(weights)[:, None, None]
        count = w.sum((0, 1))
        clean = np.where(m, x, 0)
        mean = (clean * w).sum((0, 1)) / np.maximum(count, 1e-12)
        variance = (((clean - mean) ** 2) * w).sum((0, 1)) / np.maximum(count, 1e-12)
        scale = np.sqrt(np.maximum(variance, 0))
        scale[(scale < 1e-6) | (count == 0)] = 1
        return cls(mean, scale, count > 0)

    def transform(self, x, m, d, available=None):
        x, m, d = np.asarray(x, float), np.asarray(m, bool), np.asarray(d, float)
        eligible = np.ones(len(x), bool) if available is None else np.asarray(available, bool)
        mask = m & self.observed[None, None, :] & eligible[:, None, None]
        values = (np.where(mask, x, self.mean) - self.mean) / self.scale
        # Unseen FIT channels must never affect another channel through gaps.
        gap = np.where(self.observed[None, None, :] & eligible[:, None, None], d, 0)
        features = np.concatenate((values, mask.astype(float), np.log1p(gap)), axis=2)
        token = mask.any(axis=2)
        return features.astype(np.float32), token


@dataclass
class _HistoryScale:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    observed: np.ndarray

    @classmethod
    def fit(cls, x, weights):
        x = np.asarray(x, float)
        observed = np.isfinite(x).any(axis=0)
        med = np.array([_weighted_median(x[:, j], weights) if observed[j] else 0
                        for j in range(x.shape[1])])
        filled = np.where(np.isfinite(x), x, med)
        mean = np.average(filled, weights=weights, axis=0)
        variance = np.average((filled - mean) ** 2, weights=weights, axis=0)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1
        return cls(med, mean, scale, observed)

    def transform(self, x):
        x = np.asarray(x, float)
        mask = np.isfinite(x) & self.observed
        filled = np.where(mask, x, self.median)
        values = np.where(self.observed, (filled - self.mean) / self.scale, 0)
        return np.concatenate((values, mask.astype(float)), axis=1).astype(np.float32)


class _CorrectionGRU(nn.Module):
    def __init__(self, channels, history_dim, hidden=16):
        super().__init__()
        self.gru = nn.GRUCell(3 * channels, hidden)
        self.head = nn.Linear(hidden + 2 * history_dim, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.hidden = hidden

    def forward(self, sequence, tokens, history):
        h = sequence.new_zeros((len(sequence), self.hidden))
        for t in range(sequence.shape[1]):
            update = self.gru(sequence[:, t], h)
            # Missing/padded events do not advance state. Gaps accompany the next
            # observed event; no state survives between independent anchor rows.
            h = torch.where(tokens[:, t, None], update, h)
        return self.head(torch.cat((h, history), dim=1))


def _tensor(a):
    # Own storage even if a parent pandas/NumPy view is non-writable.
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _run(model, features, tokens, history, batch_size=256):
    model.eval()
    result = []
    with torch.no_grad():
        for offset in range(0, len(features), batch_size):
            part = slice(offset, offset + batch_size)
            result.append(model(_tensor(features[part]),
                                torch.tensor(tokens[part], dtype=torch.bool),
                                _tensor(history[part])).numpy())
    return np.concatenate(result) if result else np.empty((0, 4), np.float32)


def _fit_correction(model, features, tokens, history, offset, y, groups,
                    epochs, seed, batch_size, learning_rate):
    weights = _label_weights(y, groups)
    active = weights.sum(axis=0) > 0
    clean = np.where(np.isfinite(y), y, 0)
    base = np.where(np.isfinite(offset), offset, 0)
    normalizer = weights.sum(axis=0) / len(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    trace = {"loss": [], "first_updates": [], "supervised_domains": active.tolist(),
             "objective": "equal-domain mean of participant-total-weighted L1",
             "epochs": epochs, "seed": seed}
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            correction = model(_tensor(features[idx]), torch.tensor(tokens[idx], dtype=torch.bool),
                               _tensor(history[idx]))
            error = torch.abs(_tensor(base[idx]) + correction - _tensor(clean[idx]))
            per_domain = (error * _tensor(weights[idx])).sum(0) / _tensor(
                np.maximum(len(idx) * normalizer, 1e-12))
            loss = per_domain[torch.tensor(active)].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite sequence L1 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if len(trace["first_updates"]) < 2:
                trace["first_updates"].append({
                    "head_gradient_norm": float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.head.parameters()))),
                    "recurrent_gradient_norm": float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.gru.parameters()))),
                })
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        prediction = base + _run(model, features, tokens, history)
        per_domain = (np.abs(prediction - clean) * weights).sum(0) / np.maximum(weights.sum(0), 1e-12)
        trace["loss"].append(float(per_domain[active].mean()))
    model.eval()
    return trace


@dataclass
class SequenceBundle:
    he_model: _CorrectionGRU
    sensor_model: _CorrectionGRU | None
    ema_scale: _SequenceScale
    sensor_scale: _SequenceScale | None
    history_scale: _HistoryScale
    history_names: tuple[str, ...]
    fallback: np.ndarray
    traces: dict

    def predict(self, batch: SeqBatch, use_sensor=False, *, return_raw=False):
        batch.validate(self.history_names)
        if tuple(batch.history_names) != self.history_names:
            raise ValueError("History schema/order changed after fitting")
        eligible = _history_eligible(batch)
        if not eligible.all():
            result = np.full((len(batch), 4), np.nan)
            if eligible.any():
                result[eligible] = self.predict(batch.subset(np.flatnonzero(eligible)),
                                                use_sensor=use_sensor, return_raw=return_raw)
            return result
        features, tokens = self.ema_scale.transform(batch.ema_values, batch.ema_mask, batch.ema_delta)
        history = self.history_scale.transform(batch.history)
        baseline = np.where(np.isfinite(batch.persistence), batch.persistence, self.fallback)
        raw = baseline + _run(self.he_model, features, tokens, history)
        if use_sensor and self.sensor_model is not None:
            features, tokens = self.sensor_scale.transform(
                batch.sensor_values, batch.sensor_mask, batch.sensor_delta, batch.sensor_available)
            quality = _sensor_quality(batch, self.sensor_scale)
            raw = raw + quality[:, None] * _run(self.sensor_model, features, tokens, history)
        # A domain with no observed training labels cannot publish random heads,
        # even when its persistence happens to be supplied at prediction time.
        raw[:, ~np.isfinite(self.fallback)] = np.nan
        return raw if return_raw else np.clip(raw, 0, 10)


def _sensor_quality(batch, scale):
    effective = np.asarray(batch.sensor_mask, bool) & scale.observed[None, None, :]
    denominator = batch.sensor_mask.shape[1] * max(int(scale.observed.sum()), 1)
    return effective.sum((1, 2)) / denominator * np.asarray(batch.sensor_available, bool)


def fit_sequence(batch: SeqBatch, y, groups, seed=42, epochs=20, hidden=16, *,
                 allowed_history_names: Sequence[str] = DEFAULT_HISTORY_NAMES,
                 batch_size=64, learning_rate=.01):
    """Fixed-budget C then frozen-C sensor residual fit; never uses held-out data.

    The exact history allowlist must be supplied for any new history fields.
    Per-domain subject-total weights handle missing outcomes independently.
    All train preprocessing and fallback medians use only the supplied arrays.
    """
    batch.validate(allowed_history_names)
    if not len(batch):
        raise ValueError("Cannot fit an empty batch")
    if not _history_eligible(batch).all():
        raise ValueError("Invalid required history: fit is rejected before training statistics")
    y, groups = np.asarray(y, float), np.asarray(groups)
    if y.shape != (len(batch), 4) or groups.shape != (len(batch),):
        raise ValueError("Aligned [N,4] targets and [N] groups are required")
    if np.isinf(y).any() or np.any(np.isfinite(y) & ((y < 0) | (y > 10))):
        raise ValueError("Observed EMA targets must be recorded 0--10 scores")
    if not np.isfinite(y).any():
        raise ValueError("No observed training outcomes")
    if not (1 <= epochs <= 20 and 1 <= hidden <= 16 and batch_size >= 1 and
            0 < learning_rate <= .1 and isinstance(seed, (int, np.integer))):
        raise ValueError("Sequence comparator exceeds its fixed small-model budget")
    row_weights = _person_weights(groups)
    label_weights = _label_weights(y, groups)
    fallback = np.array([_weighted_median(y[:, j], label_weights[:, j]) for j in range(4)])
    baseline = np.where(np.isfinite(batch.persistence), batch.persistence, fallback)
    ema_scale = _SequenceScale.fit(batch.ema_values, batch.ema_mask, row_weights)
    history_scale = _HistoryScale.fit(batch.history, row_weights)
    features, tokens = ema_scale.transform(batch.ema_values, batch.ema_mask, batch.ema_delta)
    history = history_scale.transform(batch.history)
    # CPU-only and local RNG restoration: separate fit calls cannot carry state
    # across participants/months or choose a seed from validation performance.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        he_model = _CorrectionGRU(4, len(batch.history_names), hidden)
        he_trace = _fit_correction(he_model, features, tokens, history, baseline, y, groups,
                                   epochs, int(seed), batch_size, learning_rate)
        own_train_c = baseline + _run(he_model, features, tokens, history)
        for parameter in he_model.parameters():
            parameter.requires_grad_(False)
        eligible = np.asarray(batch.sensor_available, bool) & np.asarray(batch.sensor_mask, bool).any((1, 2))
        eligible &= np.isfinite(y).any(1)
        sensor_model, sensor_scale = None, None
        sensor_trace = {"status": "no_sensor_training_observations"}
        if eligible.any():
            sensor_batch = batch.subset(np.flatnonzero(eligible))
            sensor_groups, sensor_y = groups[eligible], y[eligible]
            sensor_scale = _SequenceScale.fit(sensor_batch.sensor_values, sensor_batch.sensor_mask,
                                               _person_weights(sensor_groups))
            features, tokens = sensor_scale.transform(sensor_batch.sensor_values, sensor_batch.sensor_mask,
                                                       sensor_batch.sensor_delta, sensor_batch.sensor_available)
            quality = _sensor_quality(sensor_batch, sensor_scale)
            sensor_model = _CorrectionGRU(7, len(batch.history_names), hidden)
            # Quality is in the sensor correction's computation graph, including
            # during training. A zero quality observation contributes no update.
            sensor_trace = _fit_sensor_correction(sensor_model, features, tokens, history[eligible],
                own_train_c[eligible], sensor_y, sensor_groups, quality, epochs,
                int(seed) + 1, batch_size, learning_rate)
            sensor_trace["status"] = "fitted"
    traces = {"he": he_trace, "sensor": sensor_trace, "seed": int(seed),
              "history_names": list(batch.history_names), "hidden": hidden,
              "sensor_residual_reference": "frozen_C_own_train_predictions_not_OOF",
              "sensor_shrink": "observed_fraction_of_fit_observed_channels",
              "normalization": {"ema_observed": ema_scale.observed.tolist(),
                "ema_mean": ema_scale.mean.tolist(), "ema_scale": ema_scale.scale.tolist(),
                "history_observed": history_scale.observed.tolist(),
                "history_mean": history_scale.mean.tolist(), "history_scale": history_scale.scale.tolist(),
                "sensor_observed": None if sensor_scale is None else sensor_scale.observed.tolist()},
              "no_validation_selection": True, "clipping": "prediction_only_0_10"}
    return SequenceBundle(he_model, sensor_model, ema_scale, sensor_scale, history_scale,
                          tuple(batch.history_names), fallback, traces)


class _QualityCorrection(nn.Module):
    """Training-only wrapper; persistent model is the ordinary correction GRU."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def head(self):
        return self.base.head

    @property
    def gru(self):
        return self.base.gru

    def forward(self, sequence, tokens, history):
        return self.base(sequence, tokens, history[:, :-1]) * history[:, -1:]


def _fit_sensor_correction(model, features, tokens, history, offset, y, groups,
                           quality, epochs, seed, batch_size, learning_rate):
    wrapper = _QualityCorrection(model)
    return _fit_correction(wrapper, features, tokens,
        np.column_stack((history, quality)), offset, y, groups,
        epochs, seed, batch_size, learning_rate)


@dataclass
class MedianSequenceEnsemble:
    bundles: tuple[SequenceBundle, ...]

    def predict(self, batch, use_sensor=False):
        if not self.bundles:
            raise ValueError("No trained ensemble members")
        # Same all-NaN domain masks across members; avoid nanmedian's undefined
        # support change. Clipping commutes with the fixed odd median.
        return np.median(np.stack([b.predict(batch, use_sensor=use_sensor)
                                    for b in self.bundles]), axis=0)


def fit_sequence_ensemble(batch, y, groups, seeds=DEFAULT_SEEDS, **kwargs):
    if tuple(seeds) != DEFAULT_SEEDS:
        raise ValueError("The comparator uses prespecified seeds 42/2026/2027")
    return MedianSequenceEnsemble(tuple(fit_sequence(batch, y, groups, seed=s, **kwargs)
                                        for s in seeds))
