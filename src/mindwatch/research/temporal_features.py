"""Past-window sensor features and fit-only static covariates for neural models.

No speech, raw data loading, file I/O, participant split, or test selection occurs
here. Arrays must already be aligned to an observation window ending before the
assessment. Feature summaries and collection/missingness information are returned
separately so a finite count cannot masquerade as an observed sensor signal.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .evaluation import participant_weights


_TARGET = re.compile(r"(?:^|_)(?:phq_?9|gad_?7|isi|target|label|outcome|y)(?:_|$)")
_IDENTIFIER = re.compile(
    r"(?:^|_)(?:id|subject|participant|patient|sample|recording|person|site|institution)(?:_|$)"
)
_PAST = re.compile(r"(?:^|_)(?:previous|prev|prior|past|lag\d*|baseline)(?:_|$)")


def _channel_names(channels: Sequence[str], n_features: int) -> list[str]:
    if isinstance(channels, (str, bytes)) or len(channels) != n_features:
        raise ValueError("One unique sensor channel name is required per feature")
    names = []
    for channel in channels:
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("Sensor channel names must be nonempty strings")
        name = channel.lower()
        if name.startswith("sensor__"):
            name = name.removeprefix("sensor__")
        name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
        if not name or _TARGET.search(name) or _IDENTIFIER.search(name):
            raise ValueError("Sensor channels must exclude targets and identifiers")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Sensor channel names collide after canonicalization")
    return names


def temporal_features(values: np.ndarray, mask: np.ndarray, channels: Sequence[str], *,
                      windows: tuple[int, ...] = (1, 3, 7, 14),
                      end_index: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic features from exact trailing calendar positions, no filling.

    Inputs are [N,T,F], mask=1 for observed. The default uses the last 14 days.
    ``end_index`` is an exclusive prefix bound for explicit as-of calculations;
    values/masks after it are never inspected, even for value/mask validation.
    At least 14 historical positions must exist before that bound. Window means
    need >=1 observed day. Change is mean(last7)-mean(preceding7), with observations
    in both. Slope is OLS on observed day indices, preserving calendar gaps.
    Lag1 correlation uses only truly adjacent observed pairs, requires >=3 pairs,
    and is NaN for a constant member vector. No interpolation is performed.

    Returns signal columns ``sensor__temporal_<channel>__...`` and separate quality
    columns ``quality__temporal_<channel>__...``. Empty signals remain NaN; zero
    observation counts remain in the quality table only. The caller owns index
    dates/sample IDs and must verify that this historical window is time-valid.
    """
    x, m = np.asarray(values), np.asarray(mask)
    if x.ndim != 3 or x.shape != m.shape or x.shape[2] < 1:
        raise ValueError("values and mask must have equal [N,T,F] shapes with F>=1")
    if (not windows or len(set(windows)) != len(windows) or
            any(isinstance(w, bool) or not isinstance(w, (int, np.integer)) or w < 1 for w in windows)):
        raise ValueError("Window lengths must be distinct positive integer days")
    end = x.shape[1] if end_index is None else end_index
    if isinstance(end, bool) or not isinstance(end, (int, np.integer)) or not max(14, *windows) <= end <= x.shape[1]:
        raise ValueError("end_index must leave at least 14 past calendar days and requested windows")
    names = _channel_names(channels, x.shape[2])
    # Slice before any numeric inspection: future values and masks are irrelevant.
    start = end - max(14, *windows)
    historical_x = np.asarray(x[:, start:end], dtype=float)
    historical_m = np.asarray(m[:, start:end])
    if not np.isin(historical_m, [0, 1]).all():
        raise ValueError("Historical masks must be binary")
    historical_m = historical_m.astype(bool)
    if not np.isfinite(historical_x[historical_m]).all():
        raise ValueError("Observed historical sensor values must be finite")
    n = len(x)
    signal, quality = {}, {}
    recent14 = historical_x[:, -14:]
    seen14 = historical_m[:, -14:]
    day = np.arange(14, dtype=float)[None, :]
    for j, name in enumerate(names):
        signal_prefix = f"sensor__temporal_{name}__"
        quality_prefix = f"quality__temporal_{name}__"
        for window in windows:
            observed = historical_m[:, -window:, j]
            clean = np.where(observed, historical_x[:, -window:, j], 0.0)
            count = observed.sum(axis=1)
            mean = np.divide(clean.sum(axis=1), count, out=np.full(n, np.nan), where=count > 0)
            signal[signal_prefix + f"mean_{window}d"] = mean
            quality[quality_prefix + f"observed_days_{window}d"] = count
            quality[quality_prefix + f"observed_fraction_{window}d"] = count / window
        xj, mj = recent14[:, :, j], seen14[:, :, j]
        means = []
        for sl in (slice(0, 7), slice(7, 14)):
            observed = mj[:, sl]
            count = observed.sum(axis=1)
            mean = np.divide(np.where(observed, xj[:, sl], 0.0).sum(axis=1), count,
                             out=np.full(n, np.nan), where=count > 0)
            means.append(mean)
        signal[signal_prefix + "change_7d_vs_previous_7d"] = means[1] - means[0]
        quality[quality_prefix + "observed_days_previous_7d"] = mj[:, :7].sum(axis=1)
        count = mj.sum(axis=1)
        clean = np.where(mj, xj, 0.0)
        mean_x = np.divide(clean.sum(axis=1), count, out=np.zeros(n), where=count > 0)
        mean_t = np.divide((day * mj).sum(axis=1), count, out=np.zeros(n), where=count > 0)
        centered_t = day - mean_t[:, None]
        centered_x = np.where(mj, xj - mean_x[:, None], 0.0)
        denominator = (centered_t ** 2 * mj).sum(axis=1)
        slope = np.divide((centered_t * centered_x).sum(axis=1), denominator,
                          out=np.full(n, np.nan), where=(count >= 2) & (denominator > 0))
        signal[signal_prefix + "slope_14d"] = slope
        pair_mask = mj[:, :-1] & mj[:, 1:]
        pairs = pair_mask.sum(axis=1)
        a, b = np.where(pair_mask, xj[:, :-1], 0.0), np.where(pair_mask, xj[:, 1:], 0.0)
        mean_a = np.divide(a.sum(axis=1), pairs, out=np.zeros(n), where=pairs > 0)
        mean_b = np.divide(b.sum(axis=1), pairs, out=np.zeros(n), where=pairs > 0)
        da, db = (np.where(pair_mask, a - mean_a[:, None], 0.0),
                  np.where(pair_mask, b - mean_b[:, None], 0.0))
        scale = np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))
        correlation = np.divide((da * db).sum(axis=1), scale, out=np.full(n, np.nan),
                                where=(pairs >= 3) & (scale > 0))
        correlation = np.clip(correlation, -1.0, 1.0)
        signal[signal_prefix + "lag1_autocorrelation_14d"] = correlation
        quality[quality_prefix + "lag1_observed_pairs_14d"] = pairs
    signals = pd.DataFrame(signal, index=pd.RangeIndex(n))
    qualities = pd.DataFrame(quality, index=pd.RangeIndex(n))
    provenance = {"history_end_exclusive": int(end), "calendar_resolution": "daily",
                  "interpolation": "none", "slope_window_days": 14,
                  "lag1_minimum_adjacent_pairs": 3}
    signals.attrs.update(provenance, feature_kind="observed_signal_summary")
    qualities.attrs.update(provenance, feature_kind="collection_and_missingness_only")
    return signals, qualities


def _static_names(names, n_static):
    if n_static == 0:
        if names is not None and len(names):
            raise ValueError("Static feature names were supplied for an empty static tail")
        return ()
    if names is None or isinstance(names, (str, bytes)) or len(names) != n_static or len(set(names)) != n_static:
        raise ValueError("Explicit unique static_feature_names are required for the static tail")
    for name in names:
        if not isinstance(name, str) or not name.startswith(("ema__", "history__", "demo__")):
            raise ValueError("Static covariates must use ema__, history__ or demo__ prefixes")
        prefix, suffix = name.split("__", 1)
        canonical = re.sub(r"[^a-z0-9_]+", "_", suffix.lower()).strip("_")
        if not canonical or _IDENTIFIER.search(canonical):
            raise ValueError("Static covariates must exclude participant/site identifiers")
        if _TARGET.search(canonical) and not (prefix == "history" and _PAST.search(canonical)):
            raise ValueError("Current/raw target columns cannot enter static covariates")
    return tuple(names)


def _classifier(architecture, mtm_source_dir, **parameters):
    # Temporal feature extraction and module import need no optional Torch/MTM.
    if architecture == "mtm":
        from .mtm_adapter import OfficialMTMClassifier
        return OfficialMTMClassifier(source_dir=mtm_source_dir, **parameters)
    from .neural import SensorNeuralClassifier
    return SensorNeuralClassifier(architecture=architecture, **parameters)


class TemporalFusionClassifier(ClassifierMixin, BaseEstimator):
    """Small sensor GRU-D/TCN or official MTM plus fit-only static covariates.

    X = [flatten(values), flatten(mask), flatten(delta), static EMA/history/demo].
    Static median imputation and scaling are learned inside each fit call only;
    the sequence is passed unchanged to the neural adapter for its own train-only
    normalization. No row split/early stopping or target-derived preprocessing is
    performed. Groups/weights are forwarded. A missing sensor window abstains
    even if the static tail is complete. Arrays cannot establish temporal meaning:
    upstream code must guarantee that history/EMA precede the index time.
    Entirely unobserved static training features remain ignored at prediction,
    rather than activating random untrained neural weights on a new value.
    """
    def __init__(self, n_static_features: int = 0,
                 static_feature_names: Sequence[str] | None = None,
                 n_features: int | None = None, sequence_length: int = 14,
                 architecture: str = "grud", hidden_dim: int = 64,
                 epochs: int = 20, batch_size: int = 64,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 dropout: float = 0.2, seed: int = 42, device: str = "cpu",
                 mtm_source_dir: str | None = None):
        self.n_static_features = n_static_features
        self.static_feature_names = static_feature_names
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self.device = device
        self.mtm_source_dir = mtm_source_dir

    def _arrays(self, X, fitting=False):
        if self.architecture not in {"grud", "tcn", "mtm"}:
            raise ValueError("Temporal fusion architecture must be grud, tcn or official mtm")
        if self.sequence_length < 1 or self.n_static_features < 0:
            raise ValueError("Invalid sequence length or static feature count")
        names = _static_names(self.static_feature_names, self.n_static_features)
        if isinstance(X, pd.DataFrame):
            if X.columns.duplicated().any():
                raise ValueError("Duplicate packed input column names")
            sequence_names = list(X.columns[:-self.n_static_features] if self.n_static_features else X.columns)
            if not all(isinstance(c, str) and c.startswith("sensor__seq_") for c in sequence_names):
                raise ValueError("DataFrame sequence columns must use the sensor__seq_ prefix")
            if self.n_static_features and tuple(X.columns[-self.n_static_features:]) != names:
                raise ValueError("Static DataFrame columns must match the declared tail schema")
            if not fitting and hasattr(self, "feature_names_in_") and tuple(X.columns) != tuple(self.feature_names_in_):
                raise ValueError("Packed feature order changed after fit")
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[0] < 1:
            raise ValueError("Packed temporal input must be a nonempty two-dimensional matrix")
        width = array.shape[1] - self.n_static_features
        if width < 1 or width % (3 * self.sequence_length):
            raise ValueError("Packed temporal width must equal 3*T*F plus the static tail")
        n_features = width // (3 * self.sequence_length)
        if self.n_features is not None and n_features != self.n_features:
            raise ValueError("Packed sensor feature count does not match n_features")
        if not fitting and array.shape[1] != self.n_features_in_:
            raise ValueError("Packed temporal feature count changed after fit")
        sequence, tail = array[:, :width], array[:, width:]
        if np.isinf(tail).any():
            raise ValueError("Infinite static covariates must be resolved upstream")
        return sequence, tail, n_features, names

    def fit(self, X, y, sample_weight=None, groups=None):
        sequence, tail, n_features, names = self._arrays(X, fitting=True)
        n = len(sequence)
        target = np.asarray(y)
        if target.shape != (n,) or not np.isin(target, [0, 1]).all():
            raise ValueError("Temporal fusion requires aligned observed binary training labels")
        if groups is not None and (np.asarray(groups).shape != (n,) or pd.isna(groups).any()):
            raise ValueError("Aligned nonmissing participant groups are required")
        weights = (participant_weights(groups) if sample_weight is None and groups is not None
                   else np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float))
        if weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Training weights must be finite nonnegative with positive sum")
        self.static_preprocessor_ = None
        self.static_observed_ = np.isfinite(tail[weights > 0]).any(axis=0)
        if self.n_static_features:
            self.static_preprocessor_ = Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ])
            self.static_preprocessor_.fit(tail[weights > 0], scale__sample_weight=weights[weights > 0])
            allowed_tail = np.where(self.static_observed_[None, :], tail, np.nan)
            transformed_tail = self.static_preprocessor_.transform(allowed_tail)
        else:
            transformed_tail = tail
        self.classifier_ = _classifier(
            self.architecture, self.mtm_source_dir, n_features=n_features,
            sequence_length=self.sequence_length, n_demographics=self.n_static_features,
            hidden_dim=self.hidden_dim, epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            dropout=self.dropout, seed=self.seed, device=self.device,
        )
        self.classifier_.fit(np.concatenate((sequence, transformed_tail), axis=1), target,
                              sample_weight=weights, groups=groups)
        self.n_features_in_, self.n_sensor_features_ = sequence.shape[1] + tail.shape[1], n_features
        self.static_feature_names_ = names
        self.classes_ = self.classifier_.classes_
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        sequence, tail, _, _ = self._arrays(X)
        if self.static_preprocessor_ is not None:
            tail = np.where(self.static_observed_[None, :], tail, np.nan)
            tail = self.static_preprocessor_.transform(tail)
        return self.classifier_.predict_proba(np.concatenate((sequence, tail), axis=1))

    def predict(self, X):
        p = self.predict_proba(X)[:, list(self.classes_).index(1)]
        return np.where(np.isfinite(p), (p >= 0.5).astype(float), np.nan)
