"""Pinned, unmodified official MTM source adapter for academic research only.

Source: Zhong et al., KDD 2025, doi:10.1145/3711896.3737058.
The downloaded source remains subject to the HKUST Academic Software License,
including its required publication acknowledgement. No code is substituted for
the upstream token mixing, masked concat pooling, or attention implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import urllib.request

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.nn import functional as F

from .neural import (NeuralBatch, SensorEncoder, SensorNeuralClassifier,
                     _weights, resolve_device, set_seed)


MTM_REVISION = "5cc68179b98647a836aaf75c265d36e777ba56ca"
MTM_REPOSITORY = "https://github.com/zshhans/MTM"
MTM_ACKNOWLEDGEMENT = (
    "The Software used in this research was created by [Shuhan Zhong] of HKUST. © 2024 HKUST"
)
MTM_HASHES = {
    "LICENCE.txt": "d61c6172513cff22d9ce98fbeef0f0ce152439262a142e30a3217e966519c05b",
    "COPYRIGHT.txt": "1be453b3baee5a76f4695cdfd000433810fb88cc2d3215446056287e3c1aae8b",
    "mtm/mtm.py": "790797d57b944c6197e4a5eb13e73604b035c81864df486607ce1806b5c3f87c",
    "mtm/sampling.py": "43263a194dab4723f7d4c510585f4ef9020b71a6942c20e63de1c0f887b82908",
    "mtm/utils.py": "37d77b8e80040caece4a71f927f08e4fe5887e7a97ba879763df644833c565ad",
}


def verify_mtm_source(source_dir: str | Path) -> dict:
    """Verify code AND license notices before any import or execution."""
    directory = Path(source_dir)
    for relative, expected in MTM_HASHES.items():
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Official MTM source member missing/unsafe: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Official MTM source hash mismatch: {relative}")
    return {"repository": MTM_REPOSITORY, "revision": MTM_REVISION,
            "sha256": dict(MTM_HASHES), "license": "HKUST Academic Software License",
            "required_publication_acknowledgement": MTM_ACKNOWLEDGEMENT}


def prepare_mtm_source(destination: str | Path) -> dict:
    """Explicit public-code download only; URLs contain no cohort data.

    Existing files are verified rather than overwritten. This function fetches
    five pinned small files, preserves both notices, and writes provenance. Fit
    never downloads automatically; call once during public dependency setup.
    """
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Source destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    for relative, expected in MTM_HASHES.items():
        target = destination / relative
        if target.parent.is_symlink() or target.is_symlink():
            raise ValueError("Source members cannot use symlinks")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing source hash mismatch: {relative}")
            continue
        url = f"https://raw.githubusercontent.com/zshhans/MTM/{MTM_REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read(1_000_001)
        if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Downloaded source hash mismatch: {relative}")
        with target.open("xb") as stream:
            stream.write(content)
    metadata = verify_mtm_source(destination)
    provenance = destination / "UPSTREAM_PROVENANCE.json"
    if not provenance.exists():
        provenance.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def load_official_mtm(source_dir: str | Path):
    """Import hash-verified unchanged official modules under a private namespace."""
    verify_mtm_source(source_dir)
    if importlib.util.find_spec("einops") is None or importlib.util.find_spec("timm") is None:
        raise ImportError("Official MTM requires local einops and timm dependencies")
    namespace = "_klosdom_official_mtm_" + MTM_REVISION[:12]
    module_name = namespace + ".mtm"
    if module_name in sys.modules:
        return sys.modules[module_name].MTM
    package = types.ModuleType(namespace)
    package.__path__ = [str(Path(source_dir).resolve() / "mtm")]
    package.__package__ = namespace
    sys.modules[namespace] = package
    spec = importlib.util.spec_from_file_location(module_name, Path(source_dir) / "mtm" / "mtm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for key in list(sys.modules):
            if key == namespace or key.startswith(namespace + "."):
                del sys.modules[key]
        raise
    return module.MTM


class _OfficialMTMNetwork(SensorEncoder):
    def __init__(self, n_features: int, d_model: int, source_dir: str | Path,
                 ratios: tuple[int, ...], dropout: float, n_demographics: int = 0):
        super().__init__(n_features, d_model)
        official_class = load_official_mtm(source_dir)
        self.upstream = official_class(num_chn=n_features, d_static=n_demographics, num_cls=2,
                                        ratios=list(ratios), d_model=d_model,
                                        r_hid=2, drop=dropout, norm_first=True,
                                        down_mode="concat")

    def forward(self, values, mask, delta, demographics=None):
        x, observed, _ = self.prepare(values, mask, delta)
        n, t, _ = x.shape
        # Upstream mask=True means missing, opposite to the research NPZ mask.
        missing = ~observed.bool()
        times = torch.arange(t, device=x.device, dtype=torch.long).expand(n, -1)
        with torch.sparse.check_sparse_tensor_invariants(enable=True):
            return self.upstream(x, missing, times,
                                 demographics if demographics is not None else x.new_empty(n, 0))


class OfficialMTMClassifier(ClassifierMixin, BaseEstimator):
    """Train the exact official MTM core from scratch on local daily sequences.

    X uses the same 3*T*F packing as SensorNeuralClassifier. Padding/missing masks
    are converted explicitly. No pretrained checkpoint is required: random
    initialization and all training happen inside the supplied participant split.
    The historical window may use noncausal attention entirely before index time.
    Research-license notices are preserved beside the source and in provenance_.
    """
    def __init__(self, source_dir: str | Path | None = None,
                 n_features: int | None = None, sequence_length: int = 14,
                 n_demographics: int = 0,
                 hidden_dim: int = 64, ratios: tuple[int, ...] = (2, 2),
                 dropout: float = 0.2, epochs: int = 40, batch_size: int = 32,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-4,
                 patience: int = 8, seed: int = 42, device: str = "cpu"):
        self.source_dir = source_dir
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.n_demographics = n_demographics
        self.hidden_dim = hidden_dim
        self.ratios = ratios
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.device = device

    def _unpack(self, X):
        batch = SensorNeuralClassifier(n_features=self.n_features,
                                       sequence_length=self.sequence_length,
                                       n_demographics=self.n_demographics)._unpack(X)
        batch.validate(batch.values.shape[-1], demo_dim=self.n_demographics)
        return batch

    @staticmethod
    def _labels(y, n):
        y = np.asarray(y)
        if y.shape != (n,) or not np.isin(y, [0, 1]).all():
            raise ValueError("MTM requires one observed binary label per supplied row")
        return y.astype(np.int64)

    def fit(self, X, y, sample_weight=None, groups=None, validation_data=None):
        if self.source_dir is None:
            raise ValueError("prepare_mtm_source and an explicit local source_dir are required")
        if self.sequence_length < 1 or self.sequence_length > 638:
            raise ValueError("Official MTM positional embeddings require 1<=T<=638")
        if self.hidden_dim < 2 or self.hidden_dim % 2:
            raise ValueError("Official MTM rotary encoding requires an even hidden_dim")
        if not self.ratios or any(not isinstance(r, int) or r < 2 for r in self.ratios):
            raise ValueError("At least one integer downsampling ratio >=2 is required")
        if not 0 <= self.dropout < 1 or min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("Invalid MTM training settings")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid MTM optimizer settings")
        device = resolve_device(self.device)
        if device.type == "mps":
            raise ValueError("Unmodified official MTM uses complex/sparse operators; use CPU or CUDA")
        batch = self._unpack(X)
        y = self._labels(y, len(batch))
        weights = _weights(len(batch), groups, sample_weight)
        val = None
        if validation_data is not None:
            if groups is None or len(validation_data) not in {3, 4}:
                raise ValueError("MTM validation requires groups and X,y,groups[,weights]")
            vb = self._unpack(validation_data[0])
            vy = self._labels(validation_data[1], len(vb))
            vw = _weights(len(vb), validation_data[2], validation_data[3]
                          if len(validation_data) == 4 else None)
            if np.intersect1d(groups, validation_data[2]).size:
                raise ValueError("Training and validation participants overlap")
            val = vb, vy, vw
        self.provenance_ = verify_mtm_source(self.source_dir)
        self._source_dir_resolved_ = str(Path(self.source_dir).resolve())
        self.provenance_.update({"initialization": "random_training_split_only",
                                 "downsampling_ratios": list(self.ratios),
                                 "probability": "softmax_two_CE_logits",
                                 "input_resolution": "daily_summary_features",
                                 "upstream_source_modified": False})
        set_seed(self.seed)
        self.model_ = _OfficialMTMNetwork(batch.values.shape[-1], self.hidden_dim,
                                          self.source_dir, self.ratios, self.dropout,
                                          self.n_demographics).to(device)
        self.model_.set_training_statistics(batch.values, batch.mask, weights)
        eligible = self._eligible(batch)
        eligible &= weights > 0
        if not eligible.any():
            raise ValueError("MTM has no observed sensor data with positive training weight")
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        self.history_, self.best_epoch_ = [], 0
        best, state, stale = float("inf"), None, 0
        selected = np.flatnonzero(eligible)
        normalizer = weights[selected].sum() / len(selected)
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            order = rng.permutation(selected)
            total = 0.0
            for offset in range(0, len(order), self.batch_size):
                idx = order[offset:offset + self.batch_size]
                tensors = batch.subset(idx).tensors(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(tensors["values"], tensors["mask"], tensors["delta"],
                                      tensors.get("demographics"))
                point_loss = F.cross_entropy(logits, torch.as_tensor(y[idx], device=device), reduction="none")
                loss = (point_loss * torch.as_tensor(weights[idx], device=device)).mean() / normalizer
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite official MTM loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(idx)
            record = {"epoch": epoch, "train_loss": total / len(order)}
            if val is not None:
                val_loss = self._evaluation_loss(*val)
                record["validation_loss"] = val_loss
                if val_loss < best - 1e-4:
                    best, self.best_epoch_, stale = val_loss, epoch, 0
                    state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    stale += 1
            self.history_.append(record)
            if val is not None and stale >= self.patience:
                break
        if state is not None:
            self.model_.load_state_dict(state)
        else:
            self.best_epoch_ = len(self.history_)
        self.model_.eval()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def _eligible(self, batch: NeuralBatch):
        seen = self.model_.feature_observed.detach().cpu().numpy()
        return (np.asarray(batch.mask, bool) & seen).any(axis=(1, 2))

    def _evaluation_loss(self, batch, y, weights):
        selected = np.flatnonzero(self._eligible(batch) & (weights > 0))
        if not len(selected):
            raise ValueError("MTM validation has no evaluable sensor observations")
        device = next(self.model_.parameters()).device
        self.model_.eval()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(selected), self.batch_size):
                idx = selected[offset:offset + self.batch_size]
                tensors = batch.subset(idx).tensors(device)
                logits = self.model_(tensors["values"], tensors["mask"], tensors["delta"],
                                      tensors.get("demographics"))
                losses = F.cross_entropy(logits, torch.as_tensor(y[idx], device=device), reduction="none")
                total += float((losses * torch.as_tensor(weights[idx], device=device)).sum().cpu())
        result = total / weights[selected].sum()
        if not np.isfinite(result):
            raise FloatingPointError("Non-finite official MTM validation loss")
        return float(result)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        if np.asarray(X).shape[1] != self.n_features_in_:
            raise ValueError("MTM feature count changed after fit")
        batch = self._unpack(X)
        selected = np.flatnonzero(self._eligible(batch))
        probabilities = np.full((len(batch), 2), np.nan, np.float32)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            for offset in range(0, len(selected), self.batch_size):
                idx = selected[offset:offset + self.batch_size]
                tensors = batch.subset(idx).tensors(device)
                logits = self.model_(tensors["values"], tensors["mask"], tensors["delta"],
                                      tensors.get("demographics"))
                probabilities[idx] = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return np.where(np.isfinite(p), (p >= 0.5).astype(float), np.nan)

    def __getstate__(self):
        # Dynamic upstream classes are not importable in a fresh Python process.
        # Save tensor state and reconstruct the hash-verified source on reload.
        state = dict(self.__dict__)
        model = state.pop("model_", None)
        if model is not None:
            state["_official_model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            state["_official_n_features"] = model.n_features
        return state

    def __setstate__(self, state):
        state = dict(state)
        parameters = state.pop("_official_model_state", None)
        n_features = state.pop("_official_n_features", None)
        self.__dict__.update(state)
        if parameters is not None:
            self.model_ = _OfficialMTMNetwork(n_features, self.hidden_dim,
                                              getattr(self, "_source_dir_resolved_", self.source_dir),
                                              self.ratios, self.dropout, self.n_demographics)
            self.model_.load_state_dict(parameters)
            self.model_.to(resolve_device(self.device)).eval()
