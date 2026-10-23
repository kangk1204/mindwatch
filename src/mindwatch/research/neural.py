"""Small local neural comparators; no pretrained downloads or inference services.

The caller owns the participant/time split. This module never splits rows, reads
patient files, tunes on a test set, or silently substitutes a named architecture.
Torch is an optional dependency: import this module only for neural experiments.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import os
import random

import numpy as np
from .sequence_arrays import past_only_delta, flatten_sequence
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import Tensor, nn
from torch.nn import functional as F


LIMITS = {"phq9": 27.0, "gad7": 21.0, "isi": 28.0}
THRESHOLDS = {"phq9": 10.0, "gad7": 10.0, "isi": 15.0}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device: str = "cpu") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(device)


def participant_weights(groups: np.ndarray) -> np.ndarray:
    """Weights sum to one per participant, with no identifier in model inputs."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("groups must be a nonempty one-dimensional array")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float32)




@dataclass
class NeuralBatch:
    """Local arrays. Audio/text are frozen per-visit representations, not labels.

    Optional embedding availability defaults to fully finite rows. An explicit
    recording_available=False suppresses BOTH audio and text, including stale
    cached embeddings. Quality is [N,3] for sensor/audio/text in [0,1].
    """
    values: np.ndarray | None = None
    mask: np.ndarray | None = None
    delta: np.ndarray | None = None
    audio: np.ndarray | None = None
    text: np.ndarray | None = None
    demographics: np.ndarray | None = None
    audio_available: np.ndarray | None = None
    text_available: np.ndarray | None = None
    recording_available: np.ndarray | None = None
    quality: np.ndarray | None = None

    def __len__(self) -> int:
        for value in vars(self).values():
            if value is not None:
                return len(value)
        raise ValueError("At least one input array is required")

    def validate(self, n_features: int, audio_dim: int = 0,
                 text_dim: int = 0, demo_dim: int = 0) -> None:
        n = len(self)
        if n == 0:
            raise ValueError("Empty batches are unsupported")
        for name, value in vars(self).items():
            if value is not None and len(value) != n:
                raise ValueError(f"{name}: inconsistent sample count")
        if self.values is not None:
            x = np.asarray(self.values)
            m = np.asarray(self.mask) if self.mask is not None else None
            d = np.asarray(self.delta) if self.delta is not None else None
            if x.ndim != 3 or x.shape[-1] != n_features or x.shape[1] < 1:
                raise ValueError("values must have shape [N,T,n_features]")
            if m is None or d is None or m.shape != x.shape or d.shape != x.shape:
                raise ValueError("values, mask and delta must have equal shapes")
            if not np.isin(m, [0, 1]).all():
                raise ValueError("mask must be binary")
            if not np.isfinite(x[m.astype(bool)]).all():
                raise ValueError("Observed sensor values must be finite")
            if not np.isfinite(d).all() or (d < 0).any():
                raise ValueError("delta must be finite and nonnegative")
        elif self.mask is not None or self.delta is not None:
            raise ValueError("mask/delta cannot be supplied without sensor values")
        for name, dim in [("audio", audio_dim), ("text", text_dim),
                          ("demographics", demo_dim)]:
            a = getattr(self, name)
            if a is not None:
                if name != "demographics" and dim == 0:
                    raise ValueError(f"{name} was supplied but its configured dimension is zero")
                if np.asarray(a).ndim != 2 or np.asarray(a).shape != (n, dim):
                    raise ValueError(f"{name} must have shape [N,{dim}]")
                if name == "demographics" and not np.isfinite(a).all():
                    raise ValueError("Demographics must be finite encoded covariates")
        for name in ("audio_available", "text_available", "recording_available"):
            a = getattr(self, name)
            if a is not None and (np.asarray(a).shape != (n,) or
                                  not np.isin(a, [0, 1]).all()):
                raise ValueError(f"{name} must be binary [N]")
        for name in ("audio", "text"):
            a, available = getattr(self, name), getattr(self, name + "_available")
            effective = None if available is None else np.asarray(available, bool)
            if effective is not None and self.recording_available is not None:
                effective = effective & np.asarray(self.recording_available, bool)
            if effective is not None and np.any(effective):
                if a is None:
                    raise ValueError(f"{name}: available embedding is absent")
                if not np.isfinite(np.asarray(a)[effective]).all():
                    raise ValueError(f"{name}: available embeddings must be finite")
        if self.quality is not None:
            q = np.asarray(self.quality)
            if q.shape != (n, 3) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
                raise ValueError("quality must be finite [N,3] in [0,1]")

    def subset(self, indices: np.ndarray) -> NeuralBatch:
        return NeuralBatch(**{k: None if v is None else np.asarray(v)[indices]
                              for k, v in vars(self).items()})

    def tensors(self, device: torch.device) -> dict[str, Tensor]:
        return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
                for k, v in vars(self).items() if v is not None}


@dataclass
class NeuralTargets:
    phq9: np.ndarray | None = None
    gad7: np.ndarray | None = None
    isi: np.ndarray | None = None
    phq_binary: np.ndarray | None = None

    def validate(self, n: int) -> None:
        for name, value in vars(self).items():
            if value is None:
                continue
            a = np.asarray(value, dtype=float)
            if a.shape != (n,) or np.isinf(a).any():
                raise ValueError(f"{name}: invalid target shape or infinity")
            valid = a[np.isfinite(a)]
            if name == "phq_binary":
                if not np.isin(valid, [0, 1]).all():
                    raise ValueError("phq_binary must be 0, 1 or NaN")
            elif ((valid < 0) | (valid > LIMITS[name])).any():
                raise ValueError(f"{name}: target outside scale bounds")
        if self.phq9 is not None and self.phq_binary is not None:
            total, binary = np.asarray(self.phq9), np.asarray(self.phq_binary)
            known = np.isfinite(total) & np.isfinite(binary)
            if np.any((total[known] >= 10) != binary[known]):
                raise ValueError("phq_binary contradicts PHQ-9 total")

    def subset(self, indices: np.ndarray) -> NeuralTargets:
        return NeuralTargets(**{k: None if v is None else np.asarray(v)[indices]
                                for k, v in vars(self).items()})

    def tensors(self, device: torch.device) -> dict[str, Tensor]:
        return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
                for k, v in vars(self).items() if v is not None}


class SensorEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        if n_features < 1 or hidden_dim < 1:
            raise ValueError("Sensor dimensions must be positive")
        self.n_features, self.hidden_dim = n_features, hidden_dim
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_scale", torch.ones(n_features))
        self.register_buffer("feature_observed", torch.zeros(n_features, dtype=torch.bool))

    def set_training_statistics(self, values: np.ndarray, mask: np.ndarray,
                                sample_weight: np.ndarray | None = None) -> None:
        """Fit weighted moments on training observations only; no interpolation."""
        x, m = np.asarray(values, float), np.asarray(mask, bool)
        if x.shape != m.shape or x.ndim != 3 or x.shape[-1] != self.n_features:
            raise ValueError("Invalid training sensor shape")
        if not np.isfinite(x[m]).all():
            raise ValueError("Observed training values must be finite")
        w = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, float)
        if w.shape != (len(x),) or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("Invalid training weights")
        observed_weight = m * w[:, None, None]
        counts = observed_weight.sum(axis=(0, 1))
        clean = np.where(m, x, 0.0)
        mean = (clean * observed_weight).sum(axis=(0, 1)) / np.maximum(counts, 1e-12)
        var = (((clean - mean) ** 2) * observed_weight).sum(axis=(0, 1)) / np.maximum(counts, 1e-12)
        scale = np.sqrt(np.maximum(var, 0))
        scale[(scale < 1e-6) | (counts == 0)] = 1.0
        with torch.no_grad():
            self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32,
                                                    device=self.feature_mean.device))
            self.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32,
                                                     device=self.feature_scale.device))
            self.feature_observed.copy_(torch.as_tensor(counts > 0,
                                                        device=self.feature_mean.device))

    def prepare(self, values: Tensor, mask: Tensor, delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Never propagate missing NaNs through subtraction or recurrent branches.
        m = mask.bool() & self.feature_observed[None, None, :]
        clean = torch.where(m, values, self.feature_mean)
        x = (clean - self.feature_mean) / self.feature_scale
        # A channel absent from fit data cannot influence another channel's
        # hidden decay via its held-out collection/elapsed-time pattern.
        d = delta * self.feature_observed[None, None, :]
        return x, m.to(x.dtype), d


class GRUDEncoder(SensorEncoder):
    """One-layer GRU-D: diagonal input decay, hidden decay and mask input.

    The normalized training empirical mean is zero. Missing inputs decay from
    their last observed value toward that mean, never from a future observation.
    """
    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__(n_features, hidden_dim)
        self.decay_x_weight = nn.Parameter(torch.full((n_features,), 0.1))
        self.decay_x_bias = nn.Parameter(torch.full((n_features,), 0.1))
        self.decay_h = nn.Linear(n_features, hidden_dim)
        nn.init.constant_(self.decay_h.weight, 0.1 / n_features)
        nn.init.constant_(self.decay_h.bias, 0.1)
        self.cell = nn.GRUCell(2 * n_features, hidden_dim)

    def forward(self, values: Tensor, mask: Tensor, delta: Tensor,
                return_sequence: bool = False) -> Tensor:
        x, m, d = self.prepare(values, mask, delta)
        h = x.new_zeros(x.shape[0], self.hidden_dim)
        last = x.new_zeros(x.shape[0], self.n_features)
        states = []
        for t in range(x.shape[1]):
            gamma_x = torch.exp(-F.relu(d[:, t] * self.decay_x_weight + self.decay_x_bias))
            gamma_h = torch.exp(-F.relu(self.decay_h(d[:, t])))
            filled = m[:, t] * x[:, t] + (1 - m[:, t]) * gamma_x * last
            h = self.cell(torch.cat((filled, m[:, t]), dim=1), gamma_h * h)
            last = m[:, t] * x[:, t] + (1 - m[:, t]) * last
            states.append(h)
        sequence = torch.stack(states, dim=1)
        available = m.any(dim=(1, 2))
        if return_sequence:
            return sequence
        return h * available[:, None]


class _CausalBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(input_dim, hidden_dim, 3, dilation=dilation)
        self.residual = nn.Conv1d(input_dim, hidden_dim, 1) if input_dim != hidden_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.dropout(self.conv(F.pad(x, (self.pad, 0)))) + self.residual(x))


class MaskAwareTCN(SensorEncoder):
    """Causal dilated TCN over value, mask and log(1+elapsed days)."""
    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__(n_features, hidden_dim)
        self.blocks = nn.Sequential(
            _CausalBlock(3 * n_features, hidden_dim, 1, dropout),
            _CausalBlock(hidden_dim, hidden_dim, 2, dropout),
            _CausalBlock(hidden_dim, hidden_dim, 4, dropout),
        )

    def forward(self, values: Tensor, mask: Tensor, delta: Tensor,
                return_sequence: bool = False) -> Tensor:
        x, m, d = self.prepare(values, mask, delta)
        # Features never seen in training remain unavailable even in evaluation.
        d = d * self.feature_observed[None, None, :]
        inputs = torch.cat((x * m, m, torch.log1p(d)), dim=-1)
        states = self.blocks(inputs.transpose(1, 2)).transpose(1, 2)
        if return_sequence:
            return states
        return states[:, -1] * m.any(dim=(1, 2))[:, None]


@dataclass
class NeuralConfig:
    n_features: int
    sensor_architecture: str = "grud"
    audio_dim: int = 0
    text_dim: int = 0
    demo_dim: int = 0
    hidden_dim: int = 64
    dropout: float = 0.2
    multitask: bool = False
    modality_dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip: float = 5.0
    score_loss_weight: float = 0.2
    auxiliary_loss_weight: float = 0.3
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.n_features < 1 or self.hidden_dim < 1:
            raise ValueError("n_features and hidden_dim must be positive")
        if min(self.audio_dim, self.text_dim, self.demo_dim) < 0:
            raise ValueError("Embedding dimensions must be nonnegative")
        if self.sensor_architecture not in {"grud", "tcn"}:
            raise ValueError("sensor_architecture must be grud or tcn; no MTM substitute exists")
        if not 0 <= self.dropout < 1 or not 0 <= self.modality_dropout < 1:
            raise ValueError("Dropout must lie in [0,1)")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Training counts must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.score_loss_weight < 0 or self.auxiliary_loss_weight < 0:
            raise ValueError("Loss weights must be nonnegative")


class GatedMultimodalModel(nn.Module):
    """Small availability-aware fusion; target scores never enter forward()."""
    def __init__(self, config: NeuralConfig):
        super().__init__()
        config.validate()
        self.config = deepcopy(config)
        h = config.hidden_dim
        self.sensor = (GRUDEncoder(config.n_features, h) if config.sensor_architecture == "grud"
                       else MaskAwareTCN(config.n_features, h, config.dropout))
        self.audio_projection = (nn.Sequential(nn.LayerNorm(config.audio_dim), nn.Linear(config.audio_dim, h),
                                                nn.GELU()) if config.audio_dim else None)
        self.text_projection = (nn.Sequential(nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, h),
                                               nn.GELU()) if config.text_dim else None)
        self.gates = nn.ModuleList([nn.Linear(h + 1, 1) for _ in range(3)])
        self.fusion = nn.Sequential(nn.Linear(h + 3 + 3 + config.demo_dim, h), nn.GELU(),
                                    nn.Dropout(config.dropout), nn.Linear(h, 32), nn.GELU(),
                                    nn.Dropout(config.dropout))
        self.heads = nn.ModuleDict()
        for name in (LIMITS if config.multitask else {"phq9": 27.0}):
            self.heads[name + "_logit"] = nn.Linear(32, 1)
            self.heads[name + "_score"] = nn.Linear(32, 1)
        # An unlabelled auxiliary or total-score head must never publish a random
        # initialized risk/score. These flags persist in state_dict checkpoints.
        self.register_buffer("supervised_heads", torch.zeros(len(self.heads), dtype=torch.bool))

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        first = next(iter(batch.values()))
        n, h = len(first), self.config.hidden_dim
        zero = first.new_zeros((n, h))
        reps, available = [], []
        if "values" in batch:
            reps.append(self.sensor(batch["values"], batch["mask"], batch["delta"]))
            sensor_mask = batch["mask"].bool() & self.sensor.feature_observed[None, None, :]
            available.append(sensor_mask.any(dim=(1, 2)))
        else:
            reps.append(zero)
            available.append(torch.zeros(n, dtype=torch.bool, device=first.device))
        for name, projection in [("audio", self.audio_projection), ("text", self.text_projection)]:
            if name in batch and projection is not None:
                raw = batch[name]
                seen = batch.get(name + "_available", torch.isfinite(raw).all(dim=1)).bool()
                seen = seen & batch.get("recording_available", torch.ones(n, device=first.device)).bool()
                # Mask before projection: excluded embeddings have zero gradients.
                clean = torch.where(seen[:, None], raw, torch.zeros_like(raw))
                reps.append(projection(clean) * seen[:, None])
                available.append(seen)
            else:
                reps.append(zero)
                available.append(torch.zeros(n, dtype=torch.bool, device=first.device))
        observed = torch.stack(available, dim=1)
        active = observed.clone()
        if self.training and self.config.modality_dropout:
            p = self.config.modality_dropout
            sensor_keep = torch.rand(n, device=first.device) >= p
            recording_keep = torch.rand(n, device=first.device) >= p
            # Audio and text derive from one recording: drop them together.
            active &= torch.stack((sensor_keep, recording_keep, recording_keep), dim=1)
            # Keep one available source if dropout removed everything, avoiding
            # training a demographic-only score that inference must abstain on.
            restore = observed.any(1) & ~active.any(1)
            active[restore] = observed[restore]
        quality = batch.get("quality", observed.to(first.dtype)) * active
        logits = torch.cat([gate(torch.cat((rep, quality[:, i:i+1]), dim=1))
                            for i, (rep, gate) in enumerate(zip(reps, self.gates))], dim=1)
        logits = logits.masked_fill(~active, -1e9)
        gates = torch.softmax(logits, dim=1) * active
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused = sum(rep * gates[:, i:i+1] for i, rep in enumerate(reps))
        demo = batch.get("demographics", first.new_zeros((n, self.config.demo_dim)))
        shared = self.fusion(torch.cat((fused, active.to(first.dtype), quality, demo), dim=1))
        outputs = {name: head(shared).squeeze(-1) for name, head in self.heads.items()}
        for name, limit in LIMITS.items():
            if name + "_score" in outputs:
                outputs[name + "_score"] = limit * torch.sigmoid(outputs[name + "_score"])
        outputs["abstain"] = ~observed.any(dim=1)
        outputs["availability"] = observed
        outputs["gates"] = gates
        return outputs


def _head_targets(targets: Mapping[str, Tensor], multitask: bool) -> dict[str, Tensor]:
    heads: dict[str, Tensor] = {}
    for name, threshold in THRESHOLDS.items():
        if name != "phq9" and not multitask:
            continue
        if name in targets:
            total = targets[name]
            heads[name + "_score"] = total
            heads[name + "_logit"] = torch.where(torch.isfinite(total),
                                                 (total >= threshold).to(total.dtype), total)
    if "phq_binary" in targets:
        binary = targets["phq_binary"]
        fallback = heads.get("phq9_logit", torch.full_like(binary, float("nan")))
        heads["phq9_logit"] = torch.where(torch.isfinite(binary), binary, fallback)
    return heads


def masked_multitask_loss(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor],
                          sample_weight: Tensor | None = None, *,
                          score_loss_weight: float = 0.2, auxiliary_loss_weight: float = 0.3,
                          head_normalizers: Mapping[str, float] | None = None) -> Tensor:
    """Masked BCE and bounded-total Huber loss; no gradient from missing labels.

    With global head_normalizers (weighted labelled count / training N), the
    minibatch estimator preserves the desired whole-training-set weights. Without
    them this returns a weighted mean for the supplied batch, for evaluation.
    """
    base = outputs["phq9_logit"]
    weights = torch.ones_like(base) if sample_weight is None else sample_weight
    loss = base.sum() * 0.0
    heads = _head_targets(targets, "gad7_logit" in outputs)
    for key, target in heads.items():
        if key not in outputs:
            continue
        valid = torch.isfinite(target) & ~outputs["abstain"] & (weights > 0)
        clean = torch.where(valid, target, torch.zeros_like(target))
        if key.endswith("_logit"):
            point_loss = F.binary_cross_entropy_with_logits(outputs[key], clean, reduction="none")
            head_weight = 1.0
        else:
            limit = LIMITS[key.removesuffix("_score")]
            point_loss = F.smooth_l1_loss(outputs[key] / limit, clean / limit,
                                        reduction="none", beta=0.1)
            head_weight = score_loss_weight
        if not key.startswith("phq9"):
            head_weight *= auxiliary_loss_weight
        if head_normalizers is None:
            denominator = (weights * valid).sum().clamp_min(1e-12)
        else:
            denominator = base.new_tensor(len(base) * head_normalizers.get(key, 0.0)).clamp_min(1e-12)
        loss = loss + head_weight * (point_loss * weights * valid).sum() / denominator
    return loss


@dataclass
class NeuralValidation:
    batch: NeuralBatch
    targets: NeuralTargets
    groups: np.ndarray
    sample_weight: np.ndarray | None = None


@dataclass
class NeuralFitResult:
    model: GatedMultimodalModel
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    used_validation: bool = False
    seed: int = 42


def _weights(n: int, groups: np.ndarray | None, supplied: np.ndarray | None) -> np.ndarray:
    if groups is not None and np.asarray(groups).shape != (n,):
        raise ValueError("groups must match sample count")
    w = (participant_weights(groups) if supplied is None and groups is not None else
         np.ones(n, dtype=np.float32) if supplied is None else np.asarray(supplied, np.float32))
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive sum")
    return w


def _available_numpy(batch: NeuralBatch, feature_observed: np.ndarray) -> np.ndarray:
    available = np.zeros(len(batch), dtype=bool)
    if batch.values is not None:
        available |= (np.asarray(batch.mask, bool) & feature_observed).any(axis=(1, 2))
    for name in ("audio", "text"):
        raw = getattr(batch, name)
        if raw is not None:
            seen = getattr(batch, name + "_available")
            seen = np.isfinite(raw).all(axis=1) if seen is None else np.asarray(seen, bool).copy()
            if batch.recording_available is not None:
                seen &= np.asarray(batch.recording_available, bool)
            available |= seen
    return available


def fit_neural(batch: NeuralBatch, targets: NeuralTargets, config: NeuralConfig, *,
               groups: np.ndarray | None = None, sample_weight: np.ndarray | None = None,
               validation: NeuralValidation | None = None) -> NeuralFitResult:
    """Fit using training arrays and an optional disjoint-participant validation.

    No validation: exactly ``epochs`` epochs, no implicit split/early stopping.
    With validation, groups are mandatory on both sides and overlap is rejected.
    No calibrator is fitted here; the caller reserves a distinct calibration set.
    """
    config.validate()
    batch.validate(config.n_features, config.audio_dim, config.text_dim, config.demo_dim)
    targets.validate(len(batch))
    weights = _weights(len(batch), groups, sample_weight)
    if validation is not None:
        if groups is None:
            raise ValueError("Training groups are mandatory with validation")
        validation.batch.validate(config.n_features, config.audio_dim, config.text_dim, config.demo_dim)
        validation.targets.validate(len(validation.batch))
        val_weights = _weights(len(validation.batch), validation.groups, validation.sample_weight)
        if np.intersect1d(groups, validation.groups).size:
            raise ValueError("Training and validation participants overlap")
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = GatedMultimodalModel(config).to(device)
    if batch.values is not None:
        model.sensor.set_training_statistics(batch.values, batch.mask, weights)
    feature_observed = model.sensor.feature_observed.cpu().numpy()
    available = _available_numpy(batch, feature_observed)
    target_tensors = targets.tensors(torch.device("cpu"))
    heads = _head_targets(target_tensors, config.multitask)
    if "phq9_logit" not in heads:
        raise ValueError("At least one available PHQ label is required")
    if not (np.isfinite(heads["phq9_logit"].numpy()) & available & (weights > 0)).any():
        raise ValueError("No observed modality with a usable PHQ label")
    normalizers = {key: float((weights * np.isfinite(value.numpy()) * available).sum() / len(batch))
                   for key, value in heads.items()}
    with torch.no_grad():
        model.supervised_heads.copy_(torch.tensor(
            [normalizers.get(key, 0) > 0 and
             (not key.endswith("_score") or config.score_loss_weight > 0) and
             (key.startswith("phq9") or config.auxiliary_loss_weight > 0)
             for key in model.heads], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(batch))
        total = 0.0
        for offset in range(0, len(batch), config.batch_size):
            idx = order[offset:offset + config.batch_size]
            tb = batch.subset(idx).tensors(device)
            ty = targets.subset(idx).tensors(device)
            tw = torch.as_tensor(weights[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_multitask_loss(model(tb), ty, tw,
                                         score_loss_weight=config.score_loss_weight,
                                         auxiliary_loss_weight=config.auxiliary_loss_weight,
                                         head_normalizers=normalizers)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite neural training loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                                 error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
        record = {"epoch": float(epoch), "train_loss": total / len(batch),
                  "last_gradient_norm": float(grad_norm.detach().cpu())}
        if validation is not None:
            val_loss = evaluate_neural_loss(model, validation.batch, validation.targets,
                                           val_weights, batch_size=config.batch_size)
            if not np.isfinite(val_loss):
                raise ValueError("Validation has no usable PHQ observations or finite loss")
            record["validation_loss"] = val_loss
            if val_loss < best - config.min_delta:
                best, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    model.eval()
    return NeuralFitResult(model, history, best_epoch, validation is not None, config.seed)


def predict_neural(model: GatedMultimodalModel, batch: NeuralBatch,
                   batch_size: int = 128) -> dict[str, np.ndarray]:
    """Uncalibrated probabilities and totals; absent inputs/untrained heads => NaN."""
    c = model.config
    batch.validate(c.n_features, c.audio_dim, c.text_dim, c.demo_dim)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    predictions: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(batch), batch_size):
            idx = np.arange(offset, min(offset + batch_size, len(batch)))
            output = model(batch.subset(idx).tensors(device))
            abstain = output["abstain"].cpu().numpy()
            for name, tensor in output.items():
                supervised = (name not in model.heads or
                              bool(model.supervised_heads[list(model.heads).index(name)]))
                if name.endswith("_logit"):
                    key = name.removesuffix("_logit") + "_probability"
                    values = torch.sigmoid(tensor).cpu().numpy()
                    values[abstain] = np.nan
                else:
                    key, values = name, tensor.cpu().numpy()
                    if name.endswith("_score"):
                        values[abstain] = np.nan
                if not supervised:
                    values[:] = np.nan
                predictions.setdefault(key, []).append(values)
    return {k: np.concatenate(v) for k, v in predictions.items()}


def evaluate_neural_loss(model: GatedMultimodalModel, batch: NeuralBatch,
                         targets: NeuralTargets, sample_weight: np.ndarray | None = None,
                         *, batch_size: int = 128) -> float:
    """Whole-set weighted head means; no minibatch-mean averaging distortion."""
    c = model.config
    batch.validate(c.n_features, c.audio_dim, c.text_dim, c.demo_dim)
    targets.validate(len(batch))
    weights = _weights(len(batch), None, sample_weight)
    device = next(model.parameters()).device
    available = _available_numpy(batch, model.sensor.feature_observed.cpu().numpy())
    head_labels = _head_targets(targets.tensors(torch.device("cpu")), c.multitask)
    if "phq9_logit" not in head_labels or not (np.isfinite(head_labels["phq9_logit"].numpy()) & available & (weights > 0)).any():
        return float("nan")
    normalizers = {key: float((weights * np.isfinite(value.numpy()) * available).sum() / len(batch))
                   for key, value in head_labels.items()}
    model.eval()
    total = 0.0
    with torch.no_grad():
        for offset in range(0, len(batch), batch_size):
            idx = np.arange(offset, min(offset + batch_size, len(batch)))
            loss = masked_multitask_loss(model(batch.subset(idx).tensors(device)),
                                         targets.subset(idx).tensors(device),
                                         torch.as_tensor(weights[idx], device=device),
                                         score_loss_weight=c.score_loss_weight,
                                         auxiliary_loss_weight=c.auxiliary_loss_weight,
                                         head_normalizers=normalizers)
            total += float(loss.cpu()) * len(idx)
    return total / len(batch)


def save_private_checkpoint(result: NeuralFitResult, path: str | Path) -> None:
    """Save only model/config/history to a local private file; no patient arrays."""
    from mindwatch.pipeline import private_output
    path = private_output(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Checkpoint destination must be a regular local file")
    # Open with 0600 before writing, rather than chmod after a world-readable save.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as stream:
        torch.save({"config": asdict(result.model.config),
                    "state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
                    "history": result.history, "best_epoch": result.best_epoch,
                    "seed": result.seed, "used_validation": result.used_validation}, stream)




class SensorNeuralClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn adapter. X packs flattened values, then mask, then elapsed days.

    ``validation_data=(X_val, y_val, groups_val[, weights_val])`` is optional.
    Groups are required for early stopping; never pass temporal-test arrays here.
    Without it the full requested number of epochs is used. ``predict`` returns
    NaN for absent sensor data, preserving explicit abstention.
    """
    def __init__(self, n_features: int | None = None, sequence_length: int = 14,
                 n_demographics: int = 0,
                 architecture: str = "grud", hidden_dim: int = 64, dropout: float = 0.2,
                 epochs: int = 40, batch_size: int = 64, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 8, seed: int = 42,
                 device: str = "cpu"):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.n_demographics = n_demographics
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.device = device

    def _unpack(self, X: np.ndarray) -> NeuralBatch:
        X = np.asarray(X, np.float32)
        sequence_columns = X.shape[1] - self.n_demographics if X.ndim == 2 else 0
        if (X.ndim != 2 or self.sequence_length < 1 or self.n_demographics < 0 or
                sequence_columns < 1 or sequence_columns % (3 * self.sequence_length)):
            raise ValueError("X must pack values/mask/delta with 3*T*F columns plus demographic tail")
        features = sequence_columns // (3 * self.sequence_length)
        if features < 1 or (self.n_features is not None and features != self.n_features):
            raise ValueError("Packed feature count differs from n_features")
        arrays = [a.reshape(len(X), self.sequence_length, features)
                  for a in np.split(X[:, :sequence_columns], 3, axis=1)]
        demographics = X[:, sequence_columns:] if self.n_demographics else None
        return NeuralBatch(*arrays, demographics=demographics)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            groups: np.ndarray | None = None,
            validation_data: tuple | None = None) -> SensorNeuralClassifier:
        batch = self._unpack(X)
        self.n_features_in_ = np.asarray(X).shape[1]
        self.classes_ = np.array([0, 1])
        config = NeuralConfig(n_features=batch.values.shape[-1], sensor_architecture=self.architecture,
                              demo_dim=self.n_demographics,
                              hidden_dim=self.hidden_dim, dropout=self.dropout,
                              modality_dropout=0.0, epochs=self.epochs, batch_size=self.batch_size,
                              learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                              patience=self.patience, seed=self.seed, device=self.device)
        validation = None
        if validation_data is not None:
            if len(validation_data) not in {3, 4}:
                raise ValueError("validation_data requires X,y,groups[,weights]")
            validation = NeuralValidation(self._unpack(validation_data[0]),
                                           NeuralTargets(phq_binary=validation_data[1]),
                                           validation_data[2],
                                           validation_data[3] if len(validation_data) == 4 else None)
        self.fit_result_ = fit_neural(batch, NeuralTargets(phq_binary=np.asarray(y)), config,
                                      groups=groups, sample_weight=sample_weight, validation=validation)
        self.model_ = self.fit_result_.model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        if np.asarray(X).ndim != 2 or np.asarray(X).shape[1] != self.n_features_in_:
            raise ValueError("Packed feature count changed after fit")
        p = predict_neural(self.model_, self._unpack(X))["phq9_probability"]
        return np.column_stack((1 - p, p))

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return np.where(np.isfinite(p), (p >= 0.5).astype(float), np.nan)


def external_model_status(name: str, *, minute_actigraphy_confirmed: bool = False,
                           upstream_revision: str | None = None,
                           local_implementation: str | Path | None = None,
                           local_checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Prerequisite status, never fabricated performance.

    MTM has an exact-source adapter and trains from scratch; verify its pinned
    source with the dedicated adapter. PAT still requires minute-level input and
    a public pretrained checkpoint. Daily aggregates never qualify as PAT input.
    """
    name = name.lower()
    references = {"mtm": "https://www.cse.ust.hk/~gchan/papers/KDD25_MTM.pdf",
                  "pat": "https://theivyzhang.github.io/PAT-website/"}
    if name not in references:
        raise ValueError("External model must be MTM or PAT")
    reasons = []
    if name == "pat" and not minute_actigraphy_confirmed:
        reasons.append("minute_resolution_actigraphy_not_confirmed")
    if not upstream_revision:
        reasons.append("exact_upstream_revision_missing")
    if local_implementation is None or not Path(local_implementation).exists():
        reasons.append("audited_upstream_implementation_missing")
    if name == "pat" and (local_checkpoint is None or not Path(local_checkpoint).is_file()):
        reasons.append("local_public_checkpoint_missing")
    if name == "mtm" and local_implementation is not None and Path(local_implementation).exists():
        from .mtm_adapter import MTM_REVISION, verify_mtm_source
        if upstream_revision != MTM_REVISION:
            reasons.append("upstream_revision_does_not_match_pinned_adapter")
        try:
            verify_mtm_source(local_implementation)
        except ValueError:
            reasons.append("official_source_verification_failed")
    return {"model": name, "reference": references[name], "implemented": name == "mtm",
            "input_and_source_prerequisites_met": not reasons,
            "status": ("ready_from_scratch" if name == "mtm" else "external_adapter_required")
                      if not reasons else "not_eligible",
            "reasons": reasons, "upstream_revision": upstream_revision}
