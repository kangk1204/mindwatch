"""Portable, frozen PHQ inference using prior PHQ and recent EMA.

This module contains no fitted parameters and performs no fitting. Supply your
own development-fitted numeric heads. Saved releases, including participant
membership, must stay in a private directory outside every Git checkout.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

DOMAINS = ("ema_depression", "ema_anxiety", "ema_sleep", "ema_stress")
FEATURES = ("h_previous_phq9", *(f"e_{d}_{s}" for d in DOMAINS for s in ("last", "median")))
KEYS = ("sample_id", "subject_id", "index_date")


def _target_free(frame):
    if not frame.columns.is_unique:
        raise ValueError("Unique input column names required")
    blocked = {"target", "total", "phq9", "gad7", "isi", "phq_total", "phq_change"}
    for column in frame.columns:
        name = str(column).lower()
        if name in blocked or name.startswith(("target_", "current_")) or (
            name.startswith("label_") and name != "label_available_at"
        ):
            raise ValueError("Current outcomes are not inference inputs")


def _local_dates(series):
    dates = pd.to_datetime(series, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Complete timezone-naive local-calendar dates required")
    return dates


def _metadata(frame):
    _target_free(frame)
    if not set(KEYS) <= set(frame):
        raise ValueError("Participant, sample and prediction date required")
    out = frame.loc[:, list(KEYS)].copy()
    for key in ("sample_id", "subject_id"):
        if not out[key].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("Nonempty pseudonymous string keys required")
    if out.sample_id.duplicated().any():
        raise ValueError("Unique prediction sample keys required")
    out["index_date"] = _local_dates(out.index_date)
    return out


def features_from_history(queries, surveys, ema):
    """Construct nine inputs from actual observations before the anchor day.

    Queries require only sample_id, subject_id and index_date. Surveys require
    subject_id, index_date and phq9; EMA requires subject_id, date and DOMAINS.
    The most recent valid prior PHQ is used. EMA last/median values use the
    preceding 14 calendar days, never any response on the prediction date.
    Enrollment criteria belong to the study protocol, not this inference API.
    """
    meta = _metadata(queries)
    if not surveys.columns.is_unique or not ema.columns.is_unique:
        raise ValueError("Unique history column names required")
    if not {"subject_id", "index_date", "phq9"} <= set(surveys) or not {
        "subject_id", "date", *DOMAINS
    } <= set(ema):
        raise ValueError("History source columns missing")
    sv = surveys[["subject_id", "index_date", "phq9"]].reset_index(drop=True).copy()
    ev = ema[["subject_id", "date", *DOMAINS]].reset_index(drop=True).copy()
    for data, date_column in ((sv, "index_date"), (ev, "date")):
        if not data.subject_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("History participant keys must be pseudonymous strings")
        data[date_column] = _local_dates(data[date_column])
    rows, survey_dates, ema_dates, reliability = [], [], [], []
    for row in meta.itertuples():
        anchor = row.index_date.normalize()
        prior = sv[sv.subject_id.eq(row.subject_id) & sv.index_date.lt(anchor)].sort_values("index_date")
        recent = ev[ev.subject_id.eq(row.subject_id) & ev.date.ge(anchor-pd.Timedelta(days=14)) & ev.date.lt(anchor)].sort_values("date")
        if prior.index_date.dt.normalize().duplicated().any() or recent.date.dt.normalize().duplicated().any():
            raise ValueError("Resolve duplicate prior daily records before inference")
        phq = pd.to_numeric(prior.phq9, errors="raise").dropna()
        if not np.isfinite(phq).all() or ((phq < 0) | (phq > 27) | (phq != np.round(phq))).any():
            raise ValueError("Invalid prior PHQ total")
        features = {"h_previous_phq9": float(phq.iloc[-1]) if len(phq) else np.nan}
        survey_dates.append(prior.loc[phq.index[-1], "index_date"] if len(phq) else pd.NaT)
        for domain in DOMAINS:
            values = pd.to_numeric(recent[domain], errors="raise").dropna().to_numpy(float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 10)).any():
                raise ValueError("Invalid prior EMA score")
            features[f"e_{domain}_last"] = values[-1] if len(values) else np.nan
            features[f"e_{domain}_median"] = float(np.median(values)) if len(values) else np.nan
        observed = recent[recent[list(DOMAINS)].notna().any(axis=1)]
        ema_dates.append(observed.date.max() if len(observed) else pd.NaT)
        phq_date = survey_dates[-1]
        phq_age = float((anchor-phq_date.normalize()).days) if pd.notna(phq_date) else np.nan
        ema_age = float((anchor-ema_dates[-1].normalize()).days) if len(observed) else np.nan
        counts = {f"{domain}_observed_days_14d": int(recent[domain].notna().sum())
                  for domain in DOMAINS}
        domains = sum(count > 0 for count in counts.values())
        reliability.append({
            "input_provenance_status": "history_derived",
            "prior_phq_age_days": phq_age,
            "prior_phq_age_band": ("missing_prior_phq" if np.isnan(phq_age) else
                "within_90_days" if phq_age <= 90 else "91_to_180_days" if phq_age <= 180 else "over_180_days"),
            "ema_observed_days_14d": len(observed),
            "ema_observed_domains_14d": domains,
            "ema_last_age_days_14d": ema_age,
            "ema_history_status": "none_in_window" if domains == 0 else "all_domains" if domains == len(DOMAINS) else "partial_domains",
            **counts,
        })
        rows.append(features)
    x = pd.DataFrame(rows, columns=FEATURES, index=meta.index, dtype=float)
    provenance = pd.DataFrame({"latest_survey_input_date": survey_dates,
                               "latest_ema_input_date": ema_dates}, index=meta.index)
    descriptive = pd.DataFrame(reliability, index=meta.index)
    for name in _unavailable_provenance(meta.index):
        provenance[name] = descriptive.get(name, pd.Series(index=meta.index, dtype=object))
    return meta, x, provenance


def _unavailable_provenance(index):
    """Prepared arrays cannot establish source ages or actual response counts."""
    return pd.DataFrame({
        "input_provenance_status": "not_provided",
        "prior_phq_age_days": np.nan,
        "prior_phq_age_band": "unknown",
        "ema_observed_days_14d": np.nan,
        "ema_observed_domains_14d": np.nan,
        "ema_last_age_days_14d": np.nan,
        "ema_history_status": "unknown",
        **{f"{domain}_observed_days_14d": np.nan for domain in DOMAINS},
    }, index=index)


@dataclass(frozen=True)
class NumericHead:
    """Caller-supplied fitted linear parameters with training-only transforms.

    Coefficients correspond to the ordered columns in ``features``. A channel
    unseen during training is always replaced with its frozen imputation value,
    even when observed at inference. Scaled values are clipped to [-12, 12].
    ``intercept`` is the linear head's intercept, not probability calibration.
    """

    features: tuple[str, ...]
    seen: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def __post_init__(self):
        names = tuple(self.features)
        if not names or len(set(names)) != len(names) or not set(names) <= set(FEATURES):
            raise ValueError("Nonempty unique head feature subset required")
        object.__setattr__(self, "features", names)
        for name in ("seen", "median", "mean", "scale", "coef"):
            a = np.asarray(getattr(self, name))
            if a.shape != (len(names),):
                raise ValueError("Head array dimensions must match its features")
            if name == "seen":
                if a.dtype.kind != "b":
                    raise ValueError("Training observation mask must be boolean")
                a = a.astype(bool, copy=True)
            else:
                if a.dtype.kind == "c":
                    raise ValueError("Real numeric parameters required")
                a = a.astype(float, copy=True)
                if not np.isfinite(a).all() or (name == "scale" and (a <= 0).any()):
                    raise ValueError("Finite parameters and positive scales required")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not np.isscalar(self.intercept) or not np.isfinite(self.intercept):
            raise ValueError("Finite scalar intercept required")
        object.__setattr__(self, "intercept", float(self.intercept))

    def linear(self, features):
        x = features.loc[:, list(self.features)].to_numpy(float, copy=True)
        x[:, ~self.seen] = np.nan
        x = np.where(np.isfinite(x), x, self.median)
        return np.clip((x-self.mean)/self.scale, -12, 12) @ self.coef + self.intercept


@dataclass(frozen=True)
class FrozenLightweight:
    """Frozen classifier plus a separate linear PHQ change regressor.

    Training/calibration membership is supplied explicitly for exposure labels.
    These labels do not assert independent or prospective validation. Own past
    responses are legitimate inputs for population-unseen people. No active
    clinical or study-specific model is bundled with this API.
    """

    classifier: NumericHead
    change_regressor: NumericHead
    training_people: frozenset[str]
    calibration_people: frozenset[str]
    model_version: str
    data_version: str
    calibration_intercept: float = 0.
    threshold: float | None = None
    calibration_status: str = "unavailable"
    calibration_slope: float = 1.

    def __post_init__(self):
        if not isinstance(self.classifier, NumericHead) or not isinstance(self.change_regressor, NumericHead):
            raise ValueError("Frozen numeric heads required")
        for field in ("training_people", "calibration_people"):
            if isinstance(getattr(self, field), (str, bytes)):
                raise ValueError("Membership must be a collection of participant keys")
            people = frozenset(getattr(self, field))
            if not all(isinstance(v, str) and bool(v.strip()) for v in people):
                raise ValueError("Pseudonymous string membership keys required")
            object.__setattr__(self, field, people)
        for field in ("model_version", "data_version", "calibration_status"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError("Version and calibration status strings required")
        if not np.isscalar(self.calibration_intercept) or not np.isfinite(self.calibration_intercept):
            raise ValueError("Finite calibration intercept required")
        if (not isinstance(self.calibration_slope, (int, float, np.integer, np.floating))
                or isinstance(self.calibration_slope, (bool, complex, np.bool_, np.complexfloating))
                or not np.isfinite(self.calibration_slope) or self.calibration_slope <= 0):
            raise ValueError("Positive finite real calibration slope required")
        object.__setattr__(self, "calibration_slope", float(self.calibration_slope))
        if self.threshold is not None and (not np.isscalar(self.threshold) or not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1):
            raise ValueError("Threshold must be absent or between zero and one")

    def predict(self, queries, features, policy="auto"):
        """Predict from already causal, exactly ordered and row-aligned inputs.

        Prepared inputs cannot prove chronology; prefer predict_from_history
        for raw local records. A missing prior PHQ yields insufficient_data.
        Missing EMA uses training imputation in the same frozen model.
        """
        meta = _metadata(queries)
        _target_free(features)
        if tuple(features.columns) != FEATURES or len(meta) != len(features) or not meta.index.equals(features.index):
            raise ValueError("Aligned metadata and ordered nine-feature schema required")
        if policy not in ("auto", "known", "new"):
            raise ValueError("Unknown user policy")
        values = features.to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("Infinite feature values forbidden")
        prior = values[:, 0]
        valid = np.isfinite(prior)
        if ((prior[valid] < 0) | (prior[valid] > 27) | (prior[valid] != np.round(prior[valid]))).any():
            raise ValueError("Invalid prior PHQ total")
        ema = values[:, 1:]
        observed = np.isfinite(ema)
        if ((ema[observed] < 0) | (ema[observed] > 10)).any():
            raise ValueError("Invalid EMA score")
        retained = {name for head in (self.classifier, self.change_regressor)
                    for name, seen in zip(head.features, head.seen) if seen}
        retained_ema = np.array([name in retained for name in FEATURES[1:]], dtype=bool)
        effective = observed & retained_ema
        uses_ema = effective.any(axis=1)
        missing_ema = bool(retained_ema.any()) & ~uses_ema
        trained = meta.subject_id.isin(self.training_people).to_numpy()
        calibrated = meta.subject_id.isin(self.calibration_people).to_numpy()
        if policy == "new" and (trained | calibrated).any():
            raise ValueError("Population-seen participant cannot be declared new")
        if policy == "known" and not trained.all():
            raise ValueError("Known policy requires population training membership")
        raw = expit(self.classifier.linear(features))
        p = raw.copy() if self.calibration_intercept == 0 and self.calibration_slope == 1 else expit(
            self.calibration_slope*logit(np.clip(raw, 1e-7, 1-1e-7)) + self.calibration_intercept)
        total = np.clip(prior + self.change_regressor.linear(features), 0, 27)
        out = meta.copy()
        out["policy"] = np.where(trained, "known", np.where(calibrated, "calibration_only", "new")) if policy == "auto" else policy
        out["population_exposure"] = np.where(trained, "training", np.where(calibrated, "calibration_only", "population_unseen"))
        out["raw_prediction"] = np.where(valid, raw, np.nan)
        out["prediction"] = np.where(valid, p, np.nan)
        out["threshold"] = np.where(valid, self.threshold if self.threshold is not None else np.nan, np.nan)
        out["predicted_phq9"] = np.where(valid, total, np.nan)
        out["predicted_phq9_change"] = out.predicted_phq9-prior
        # Available inputs and effective model inputs are different: a supplied
        # channel may have been removed or never observed during fitting.
        out["ema_observed_features"] = observed.sum(axis=1)
        out["ema_retained_features"] = int(retained_ema.sum())
        out["ema_used_features"] = np.where(valid, effective.sum(axis=1), 0)
        out["ema_used"] = uses_ema & valid
        out["used_modalities"] = np.where(~valid, "none", np.where(uses_ema, "past_phq+ema", "past_phq"))
        out["status"] = np.where(~valid, "insufficient_data", np.where(missing_ema, "ok_ema_missing_imputed", "ok"))
        out["fallback_reason"] = np.where(~valid, "missing_prior_phq", np.where(missing_ema, "ema_missing_frozen_training_imputation", ""))
        out["calibration_status"] = np.where(valid, self.calibration_status, "not_applicable")
        out["model_version"] = self.model_version
        out["data_version"] = self.data_version
        for column, values in _unavailable_provenance(meta.index).items():
            out[column] = values
        out["input_reliability_scope"] = "descriptive_availability_not_validated_confidence"
        return out

    def predict_from_history(self, queries, surveys, ema, policy="auto"):
        meta, features, provenance = features_from_history(queries, surveys, ema)
        out = self.predict(meta, features, policy)
        for column in provenance:
            out[column] = provenance[column]
        return out


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def save_release(model, directory):
    """Save caller-supplied fitted arrays privately, without pickle or overwrites."""
    from .pipeline import private_output

    if not isinstance(model, FrozenLightweight):
        raise TypeError("FrozenLightweight required")
    folder = private_output(directory)
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    folder.chmod(0o700)
    arrays, heads = {}, {}
    for prefix, head in (("classifier", model.classifier), ("change_regressor", model.change_regressor)):
        heads[prefix] = {"features": list(head.features), "intercept": head.intercept}
        for field in ("seen", "median", "mean", "scale", "coef"):
            arrays[prefix+"_"+field] = getattr(head, field)
    spec = {"format_version": 2, "features": list(FEATURES), "window_days": 14,
            "heads": heads, "training_people": sorted(model.training_people),
            "calibration_people": sorted(model.calibration_people),
            "model_version": model.model_version, "data_version": model.data_version,
            "calibration_intercept": float(model.calibration_intercept),
            "calibration_slope": model.calibration_slope,
            "threshold": None if model.threshold is None else float(model.threshold),
            "calibration_status": model.calibration_status}
    fd = os.open(folder/"weights.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    _write_json(folder/"model.json", spec)
    _write_json(folder/"manifest.json", {"source_sha256": _digest(__file__),
                "files": {name: _digest(folder/name) for name in ("model.json", "weights.npz")}})


def load_release(directory):
    """Load a local numeric release, checking hashes and schema before inference.

    Checksums detect accidental changes; they are not authenticity signatures.
    Keep releases private and obtain them from a trusted source.
    """
    from .pipeline import private_output

    folder = private_output(directory)
    if any((folder/name).is_symlink() for name in ("manifest.json", "model.json", "weights.npz")):
        raise ValueError("Release member symlinks are forbidden")
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if set(manifest.get("files", {})) != {"model.json", "weights.npz"} or manifest.get("source_sha256") != _digest(__file__):
        raise ValueError("Frozen source or release member schema changed")
    for name, digest in manifest["files"].items():
        if digest != _digest(folder/name):
            raise ValueError("Frozen release checksum changed")
    spec = json.loads((folder/"model.json").read_text(encoding="utf-8"))
    if spec.get("format_version") not in (1, 2) or tuple(spec.get("features", ())) != FEATURES or spec.get("window_days") != 14:
        raise ValueError("Unexpected release format or feature schema")
    heads = {}
    expected = {prefix+"_"+field for prefix in ("classifier", "change_regressor") for field in ("seen", "median", "mean", "scale", "coef")}
    with np.load(folder/"weights.npz", allow_pickle=False) as stored:
        if set(stored.files) != expected:
            raise ValueError("Unexpected numeric release member")
        for prefix in ("classifier", "change_regressor"):
            heads[prefix] = NumericHead(**spec["heads"][prefix], **{
                field: stored[prefix+"_"+field] for field in ("seen", "median", "mean", "scale", "coef")})
    if spec["format_version"] == 2 and "calibration_slope" not in spec:
        raise ValueError("Calibration slope missing from release")
    return FrozenLightweight(**heads, calibration_slope=spec.get("calibration_slope", 1.), **{field: spec[field] for field in (
        "training_people", "calibration_people", "model_version", "data_version",
        "calibration_intercept", "threshold", "calibration_status")})
