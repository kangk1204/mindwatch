"""Local known-user monitoring models with explicit availability routing.

Only the supplied fit/calibration outcomes are read. No patient I/O, remote
inference, split creation, validation selection or implicit model substitution
occurs here. Availability/history provenance is an upstream data contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import inspect
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import BenchmarkConfig, _calibrated, _calibrator, _estimator, _probability
from .evaluation import participant_weights, select_sensitivity_threshold


BRANCHES = {"A", "B", "C", "D"}
_IDENTIFIER = re.compile(r"(?:^|_)(?:id|subject|participant|patient|sample|recording|person|site|institution)(?:_|$)")
_TARGET = re.compile(r"(?:^|_)(?:phq_?9|gad_?7|isi|target|label|outcome|y)(?:_|$)")
_PAST = re.compile(r"(?:^|_)(?:previous|prev|prior|past|lag\d*|baseline)(?:_|$)")


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def _fingerprint(payload):
    # Sample identifiers occur only inside this in-memory hash payload, never in
    # the returned provenance or model_version. Ordering is deliberately binding.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                         default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _dataset_manifest(data):
    provenance = getattr(data, "provenance", {}) or {}
    for key in ("dataset_manifest_sha256", "dataset_manifest_sha", "data_manifest_sha256", "data_manifest_sha"):
        if provenance.get(key):
            return str(provenance[key])
    return None


def _positions(rows, n, *, allow_empty=True):
    positions = np.asarray(rows)
    if positions.size == 0 and allow_empty:
        return np.empty(0, dtype=np.int64)
    if (positions.ndim != 1 or positions.dtype.kind not in "iu" or
            positions.size == 0 or len(np.unique(positions)) != len(positions) or
            positions.min() < 0 or positions.max() >= n):
        raise ValueError("Monitoring rows must be unique in-range integer positions")
    return positions.astype(np.int64)


def _frame(data):
    frame = data.frame
    if not {"sample_id", "subject_id", "index_date", "history_available", "sensor_available", "ema_available"}.issubset(frame):
        raise ValueError("Monitoring metadata/availability columns are missing")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any() or frame.subject_id.isna().any():
        raise ValueError("Monitoring samples must be unique and participants present")
    for name in ("history_available", "sensor_available", "ema_available"):
        if not np.isin(frame[name].to_numpy(), [0, 1]).all():
            raise ValueError("Monitoring availability flags must be explicit binary values")
    return frame


def branch_available(frame, branch):
    """A=history, B=history+sensor, C=history+EMA, D=history+sensor+EMA."""
    if branch not in BRANCHES:
        raise ValueError("Monitoring branch must be A/B/C/D")
    available = frame.history_available.to_numpy(dtype=bool).copy()
    if branch in {"B", "D"}:
        available &= frame.sensor_available.to_numpy(dtype=bool)
    if branch in {"C", "D"}:
        available &= frame.ema_available.to_numpy(dtype=bool)
    return available


def _names(names, allowed_prefixes=("history__", "demo__", "sensor__", "ema__", "quality__"), *, allow_forecast_current=False):
    if not names or len(set(names)) != len(names):
        raise ValueError("Monitoring predictors need nonempty unique names")
    for name in names:
        if not isinstance(name, str) or not name.startswith(allowed_prefixes):
            raise ValueError("Unapproved monitoring predictor prefix")
        prefix, suffix = name.split("__", 1)
        clean = re.sub(r"[^a-z0-9_]+", "_", suffix.lower()).strip("_")
        if not clean or _IDENTIFIER.search(clean):
            raise ValueError("Participant/site identifiers cannot be monitoring predictors")
        allowed_history = prefix == "history" and (_PAST.search(clean) or
                             (allow_forecast_current and clean.startswith("forecast_current_")))
        if _TARGET.search(clean) and not allowed_history:
            raise ValueError("Current/raw targets cannot be monitoring predictors")
    return tuple(names)


def _design(data, branch, model, rows, *, expected_names=None, expected_width=None, allow_forecast_current=False):
    if branch not in data.designs:
        raise ValueError("Monitoring design branch is missing")
    design = data.designs[branch]
    neural = model in {"grud", "tcn"}
    if neural:
        if branch not in {"B", "D"} or "sequence" not in design:
            raise ValueError("Sensor neural models require B/D sequence designs")
        names = _names(design.get("static_names", []), ("history__", "demo__", "ema__"), allow_forecast_current=allow_forecast_current)
        matrix = design["sequence"]
        width = int(design.get("sequence_columns", 0))
        if width < 1 or np.shape(matrix) != (len(data.frame), width + len(names)):
            raise ValueError("Monitoring sequence/static design dimensions disagree")
    else:
        names = _names(design.get("tabular_names", []), allow_forecast_current=allow_forecast_current)
        matrix = design.get("tabular")
        width = len(names)
        if np.shape(matrix) != (len(data.frame), width):
            raise ValueError("Monitoring tabular names and dimensions disagree")
    if expected_names is not None and names != tuple(expected_names):
        raise ValueError("Monitoring predictor order changed after fit")
    if expected_width is not None and width != expected_width:
        raise ValueError("Monitoring predictor width changed after fit")
    # Select rows before numeric inspection; unrelated validation/test records do
    # not enter preprocessing, fitting, or target conversion.
    x = np.asarray(matrix)[rows].astype(float, copy=True)
    if np.isinf(x).any():
        raise ValueError("Infinite monitoring predictors must be resolved upstream")
    return x, names, width


def _targets(frame, rows, target_column, task):
    if task not in {"classification", "regression"}:
        raise ValueError("Monitoring task must be classification or regression")
    if target_column not in frame:
        raise ValueError("Monitoring target column is absent")
    values = pd.to_numeric(frame.iloc[rows][target_column], errors="coerce").to_numpy(dtype=float)
    if np.isinf(values).any() or (task == "classification" and np.any(np.isfinite(values) & ~np.isin(values, [0, 1]))):
        raise ValueError("Monitoring target values violate the declared task")
    return values


def _dates(frame, rows, column):
    if not len(rows):
        return pd.DatetimeIndex([])
    if column not in frame:
        raise ValueError("Monitoring label/date metadata is absent")
    dates = pd.to_datetime(frame.iloc[rows][column], errors="coerce")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        raise ValueError("Monitoring date metadata must be complete local timestamps")
    return pd.DatetimeIndex(dates)


def _fit_boundaries(frame, rows, config):
    index = _dates(frame, rows, "index_date")
    label_column = _get(config, "label_time_column", "label_available_at")
    released = _dates(frame, rows, label_column)
    if len(rows) and (released < index).any():
        raise ValueError("Label availability cannot precede its assessment")
    end = _get(config, "fit_end_exclusive")
    if end is not None:
        end = pd.Timestamp(end)
        if (index >= end).any() or (released >= end).any():
            raise ValueError("Fit data/labels cross the supplied availability boundary")


def _weighted_fit(estimator, x, y, groups):
    weights = participant_weights(groups)
    if isinstance(estimator, Pipeline):
        kwargs = {estimator.steps[-1][0] + "__sample_weight": weights}
        if "scale" in estimator.named_steps:
            kwargs["scale__sample_weight"] = weights
        estimator.fit(x, y, **kwargs)
    else:
        kwargs = {"sample_weight": weights}
        if "groups" in inspect.signature(estimator.fit).parameters:
            kwargs["groups"] = np.asarray(groups)
        estimator.fit(x, y, **kwargs)
    return estimator


def _regressor(model, parameters, seed):
    if model == "ridge":
        fitted = Ridge(alpha=parameters.get("alpha", 10.0))
        transforms = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                      ("scale", StandardScaler())]
    elif model == "catboost":
        from catboost import CatBoostRegressor
        fitted = CatBoostRegressor(iterations=300, learning_rate=.05, loss_function="MAE",
                                    depth=parameters.get("depth", 3),
                                    l2_leaf_reg=parameters.get("l2_leaf_reg", 10.0),
                                    random_seed=seed, thread_count=1, verbose=False,
                                    allow_writing_files=False)
        unsupported = set(parameters) - {"depth", "l2_leaf_reg"}
        if unsupported:
            raise ValueError("Unsupported fixed-budget CatBoost regression parameter")
        transforms = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    else:
        raise ValueError("Regression monitoring supports ridge/catboost only")
    return Pipeline([*transforms, ("model", fitted)])


@dataclass
class BaseEstimatorFit:
    branch: str
    model_name: str
    task: str
    estimators: list[Any]
    feature_names: tuple[str, ...]
    design_width: int
    metadata: dict[str, Any]


def fit_base_estimator(data, branch, model, parameters, fit_rows, config,
                        target_column="y_phq9", task="classification", seeds=None):
    """Fit exactly the passed rows; no filtering or implicit family fallback.

    For GRU-D/TCN, default seeds are 42/2026/2027, with probabilities averaged
    before any later calibration. The returned object records the actual family
    and seeds. A failed neural fit raises; it never relabels an elastic net as DL.
    """
    frame = _frame(data)
    rows = _positions(fit_rows, len(frame), allow_empty=False)
    _fit_boundaries(frame, rows, config)
    if not branch_available(frame, branch)[rows].all():
        raise ValueError("fit_base_estimator received rows unavailable for its branch")
    allow_forecast = bool(_get(config, "allow_current_state_for_forecast", False))
    if allow_forecast and target_column in {"y_phq9", "y_gad7", "y_isi", "phq9", "gad7", "isi"}:
        raise ValueError("Current-state targets cannot enable forecast-current covariates")
    x, names, width = _design(data, branch, model, rows, allow_forecast_current=allow_forecast)
    y = _targets(frame, rows, target_column, task)
    if not np.isfinite(y).all():
        raise ValueError("fit_base_estimator cannot silently discard unlabelled rows")
    if task == "classification" and len(np.unique(y)) != 2:
        raise ValueError("A classification fit requires both classes")
    if task == "regression" and len(y) < 2:
        raise ValueError("A regression fit requires at least two labelled rows")
    neural = task == "classification" and model in {"grud", "tcn"}
    # Explicit call override > canonical YAML `seeds` > legacy alias > defaults.
    # The ordering is essential for the runner's deliberately different-seed
    # control; silently ignoring the canonical field would repeat the same fit.
    actual_seeds = (tuple(seeds if seeds is not None else
                          _get(config, "seeds", _get(config, "neural_seeds", (42, 2026, 2027))))
                    if neural else (int(_get(config, "seed", 42)),))
    if (not actual_seeds or len(set(actual_seeds)) != len(actual_seeds) or
            any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in actual_seeds)):
        raise ValueError("Monitoring ensemble seeds must be nonempty and distinct")
    estimators = []
    parameters = dict(parameters)
    if task == "classification" and model not in {"elasticnet", "catboost", "grud", "tcn"}:
        raise ValueError("Unknown monitoring classifier family")
    for seed in actual_seeds:
        if neural:
            from .temporal_features import TemporalFusionClassifier
            estimator = TemporalFusionClassifier(
                n_static_features=len(names), static_feature_names=names,
                sequence_length=int(_get(config, "window", 14)), architecture=model,
                batch_size=int(_get(config, "neural_batch_size", 64)),
                device=_get(config, "neural_device", "cpu"), seed=int(seed), **parameters)
        elif task == "classification":
            estimator = _estimator(model, parameters, int(seed), BenchmarkConfig())
        else:
            estimator = _regressor(model, parameters, int(seed))
        estimators.append(_weighted_fit(estimator, x, y, frame.iloc[rows].subject_id.to_numpy()))
    fit_fingerprint = _fingerprint({"target_column": target_column, "task": task,
                                    "branch": branch, "model": model, "parameters": parameters,
                                    "seeds": [int(s) for s in actual_seeds],
                                    "fit_sample_ids": frame.iloc[rows].sample_id.astype(str).tolist(),
                                    "feature_names": names, "design_width": width,
                                    "dataset_manifest_sha256": _dataset_manifest(data)})
    version = f"{_get(config, 'version', 'monitoring_v6')}:{task}:{target_column}:{branch}:{model}:{fit_fingerprint}"
    metadata = {"status": "fitted", "branch": branch, "actual_model": model, "task": task,
                "parameters": parameters, "seeds": [int(s) for s in actual_seeds],
                "ensemble_size": len(estimators), "n_fit_rows": len(rows),
                "n_fit_participants": int(frame.iloc[rows].subject_id.nunique()),
                "probability_averaging": "before_calibration" if neural else "single_estimator",
                "model_version": version, "target_column": target_column,
                "fit_fingerprint": fit_fingerprint, "dataset_manifest_sha256": _dataset_manifest(data),
                "allow_current_state_for_forecast": allow_forecast,
                "fit_index_max": _dates(frame, rows, "index_date").max().isoformat(),
                "label_availability_max": _dates(frame, rows, _get(config, "label_time_column", "label_available_at")).max().isoformat()}
    return BaseEstimatorFit(branch, model, task, estimators, names, width, metadata)


def predict_base(estimator: BaseEstimatorFit, data, branch, rows, task="classification"):
    """Raw probabilities/scores, aligned to passed rows; never accesses outcomes."""
    frame = _frame(data)
    positions = _positions(rows, len(frame))
    if estimator.branch != branch or estimator.task != task:
        raise ValueError("Monitoring estimator branch/task does not match prediction")
    x, _, _ = _design(data, branch, estimator.model_name, positions,
                       expected_names=estimator.feature_names, expected_width=estimator.design_width,
                       allow_forecast_current=estimator.metadata.get("allow_current_state_for_forecast", False))
    if not len(positions):
        return np.empty(0, dtype=float)
    if task == "classification":
        predictions = np.stack([_probability(model, x) for model in estimator.estimators])
    else:
        predictions = np.stack([np.asarray(model.predict(x), dtype=float) for model in estimator.estimators])
    # All members must predict: do not silently average a smaller ensemble if one
    # seed abstains or fails. Routing can then select an explicitly lower branch.
    valid = np.isfinite(predictions).all(axis=0)
    result = np.full(len(positions), np.nan)
    result[valid] = predictions[:, valid].mean(axis=0)
    if task == "classification" and np.any(np.isfinite(result) & ((result < 0) | (result > 1))):
        raise ValueError("Classifier returned invalid probabilities")
    return result


@dataclass
class BranchFit:
    base: BaseEstimatorFit
    calibrator: Any = None
    threshold: float | None = None
    calibration_status: str = "insufficient_calibration"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringBundle:
    feature_set: str
    task: str
    branches: dict[str, BranchFit]
    metadata: dict[str, Any]

    def predict(self, data, rows):
        frame = _frame(data)
        positions = _positions(rows, len(frame))
        name = "probability" if self.task == "classification" else "score"
        result = pd.DataFrame({name: np.full(len(positions), np.nan),
                               "used_modalities": "none", "fallback_reason": "insufficient_history",
                               "calibration_status": "unavailable_no_model", "threshold": np.nan,
                               "model_version": self.metadata["model_version"]}, index=pd.RangeIndex(len(positions)))
        if not len(positions):
            return result
        history = frame.iloc[positions].history_available.to_numpy(dtype=bool)
        sensor = frame.iloc[positions].sensor_available.to_numpy(dtype=bool)
        ema = frame.iloc[positions].ema_available.to_numpy(dtype=bool)
        reasons = [[] for _ in positions]
        for i in np.flatnonzero(history):
            if self.feature_set in {"B", "D"} and not sensor[i]:
                reasons[i].append("sensor_unavailable")
            if self.feature_set in {"C", "D"} and not ema[i]:
                reasons[i].append("ema_unavailable")
        order = {"A": ["A"], "B": ["B", "A"], "C": ["C", "A"], "D": ["D", "B", "C", "A"]}[self.feature_set]
        unresolved = history.copy()
        for branch in order:
            attempt = np.flatnonzero(unresolved & branch_available(frame, branch)[positions])
            if not len(attempt):
                continue
            fitted = self.branches.get(branch)
            if fitted is None:
                for i in attempt:
                    reasons[i].append("branch_unfitted_" + branch)
                continue
            try:
                values = predict_base(fitted.base, data, branch, positions[attempt], self.task)
                valid = np.isfinite(values)
                if self.task == "classification" and fitted.calibrator is not None and valid.any():
                    values[valid] = _calibrated(fitted.calibrator, values[valid])
                valid &= np.isfinite(values)
            except Exception as exc:
                for i in attempt:
                    reasons[i].append("branch_prediction_failed_" + branch + ":" + type(exc).__name__)
                continue
            for i in attempt[~valid]:
                reasons[i].append("branch_prediction_unavailable_" + branch)
            predicted = attempt[valid]
            result.loc[predicted, name] = values[valid]
            result.loc[predicted, "used_modalities"] = branch
            result.loc[predicted, "fallback_reason"] = [";".join(reasons[i]) if reasons[i] else "none" for i in predicted]
            result.loc[predicted, "calibration_status"] = fitted.calibration_status
            result.loc[predicted, "threshold"] = fitted.threshold if fitted.threshold is not None else np.nan
            result.loc[predicted, "model_version"] = fitted.metadata["model_version"]
            unresolved[predicted] = False
        for i in np.flatnonzero(unresolved):
            result.at[i, "fallback_reason"] = ";".join(reasons[i] + ["no_fitted_branch"])
        return result


def fit_monitoring_bundle(data, fit_rows, cal_rows, selection, config,
                            target_column="y_phq9", task="classification"):
    """Fit routed branches and independent branch-specific calibration only.

    The selected full branch uses its selected family/parameters. Every needed
    lower branch is an explicitly recorded fixed elastic net (C=.01,l1_ratio=0)
    or Ridge(alpha=10). Insufficient/failed upper fits are retained in metadata
    and predictions route downward with reasons, never retaining a false DL name.
    """
    frame = _frame(data)
    fit = _positions(fit_rows, len(frame), allow_empty=False)
    cal = _positions(cal_rows, len(frame))
    if np.intersect1d(fit, cal).size:
        raise ValueError("Monitoring fit/calibration row overlap is forbidden")
    _fit_boundaries(frame, fit, config)
    if len(cal):
        cal_index = _dates(frame, cal, "index_date")
        fit_released = _dates(frame, fit, _get(config, "label_time_column", "label_available_at"))
        if fit_released.max() >= cal_index.min():
            raise ValueError("Fit labels must be released strictly before calibration begins")
        cal_end = _get(config, "calibration_end_exclusive")
        if cal_end is not None and (_dates(frame, cal, _get(config, "label_time_column", "label_available_at")) >= pd.Timestamp(cal_end)).any():
            raise ValueError("Calibration labels cross the supplied availability boundary")
    requested = selection.get("feature_set")
    if requested not in BRANCHES:
        raise ValueError("Selected monitoring feature_set must be A/B/C/D")
    model = selection.get("model")
    parameters = selection.get("parameters", {})
    needed = {"A": ["A"], "B": ["A", "B"], "C": ["A", "C"], "D": ["A", "B", "C", "D"]}[requested]
    # A malformed feature schema is an implementation/data error, not ordinary
    # missingness. Reject it before fitting any fallback branch.
    for branch in needed:
        family = model if branch == requested else "elasticnet" if task == "classification" else "ridge"
        _design(data, branch, family, np.empty(0, dtype=np.int64),
                  allow_forecast_current=bool(_get(config, "allow_current_state_for_forecast", False)))
    # Targets are accessed only after selecting the exact fit/cal rows.
    fit_targets = _targets(frame, fit, target_column, task)
    cal_targets = _targets(frame, cal, target_column, task) if len(cal) else np.empty(0)
    fitted_branches, statuses = {}, {}
    for branch in needed:
        available = branch_available(frame, branch)
        branch_fit = fit[available[fit] & np.isfinite(fit_targets)]
        actual_model = model if branch == requested else "elasticnet" if task == "classification" else "ridge"
        actual_parameters = parameters if branch == requested else {"C": .01, "l1_ratio": 0.0} if task == "classification" else {"alpha": 10.0}
        status = {"requested_model": actual_model, "n_fit_available": len(branch_fit),
                  "n_fit_missing_label": int((available[fit] & ~np.isfinite(fit_targets)).sum()),
                  "full_selected_branch": branch == requested}
        yfit = _targets(frame, branch_fit, target_column, task)
        subjects = frame.iloc[branch_fit].subject_id
        neural = task == "classification" and actual_model in {"grud", "tcn"}
        minimum_people = int(_get(config, "min_neural_people" if neural else "min_fit_people", 0))
        minimum_per_class = int(_get(config, "min_neural_per_class" if neural else "min_fit_per_class", 1))
        status["n_fit_participants"] = int(subjects.nunique())
        status["minimum_fit_participants"] = minimum_people
        support = status["n_fit_participants"] >= minimum_people
        if task == "classification":
            class_people = {str(label): int(subjects.iloc[np.flatnonzero(yfit == label)].nunique()) for label in (0, 1)}
            status["fit_class_participants"] = class_people
            status["minimum_fit_per_class"] = minimum_per_class
            support &= all(class_people[str(label)] >= minimum_per_class for label in (0, 1))
        else:
            support &= len(yfit) >= 2
        if not support:
            statuses[branch] = {**status, "status": "insufficient_fit_support"}
            continue
        try:
            base = fit_base_estimator(data, branch, actual_model, actual_parameters, branch_fit,
                                      config, target_column=target_column, task=task)
        except Exception as exc:
            statuses[branch] = {**status, "status": "fit_failed", "exception_type": type(exc).__name__}
            continue
        status.update(status="fitted", actual_model=base.model_name, model_version=base.metadata["model_version"],
                      seeds=base.metadata["seeds"])
        fitted = BranchFit(base)
        if task == "regression":
            fitted.calibration_status = "not_applicable_regression"
        else:
            branch_cal = cal[available[cal] & np.isfinite(cal_targets)]
            raw = predict_base(base, data, branch, branch_cal, task)
            ycal = _targets(frame, branch_cal, target_column, task)
            keep = np.isfinite(raw)
            counts = {str(label): int((ycal[keep] == label).sum()) for label in (0, 1)}
            status["calibration_class_counts"] = counts
            status["n_calibration_predicted"] = int(keep.sum())
            if all(counts[str(label)] >= 10 for label in (0, 1)):
                try:
                    groups = frame.iloc[branch_cal[keep]].subject_id.to_numpy()
                    fitted.calibrator = _calibrator(ycal[keep], raw[keep], groups)
                    calibrated = _calibrated(fitted.calibrator, raw[keep])
                    operating = select_sensitivity_threshold(ycal[keep], calibrated, groups,
                                                               float(_get(config, "target_sensitivity", .8)))
                    if operating["threshold"] is None:
                        raise ValueError("Monitoring calibration threshold is unavailable")
                    fitted.threshold = float(operating["threshold"])
                    fitted.calibration_status = "platt_calibrated"
                    status["operating_point"] = operating
                except Exception as exc:
                    fitted.calibrator, fitted.threshold = None, None
                    fitted.calibration_status = "calibration_failed_identity"
                    status["calibration_exception_type"] = type(exc).__name__
        status["calibration_status"] = fitted.calibration_status
        fitted.metadata = status
        fitted_branches[branch], statuses[branch] = fitted, status
    bundle_fingerprint = _fingerprint({
        "target_column": target_column, "task": task, "feature_set": requested,
        "fit_sample_ids": frame.iloc[fit].sample_id.astype(str).tolist(),
        "calibration_sample_ids": frame.iloc[cal].sample_id.astype(str).tolist(),
        "dataset_manifest_sha256": _dataset_manifest(data),
        "target_sensitivity": float(_get(config, "target_sensitivity", .8)),
        "branch_dispositions": statuses,
    })
    for branch, fitted in fitted_branches.items():
        fitted.metadata["base_model_version"] = fitted.base.metadata["model_version"]
        fitted.metadata["model_version"] = fitted.base.metadata["model_version"] + ":bundle-" + bundle_fingerprint
    version = f"{_get(config, 'version', 'monitoring_v6')}:{task}:{target_column}:{requested}:{bundle_fingerprint}"
    metadata = {"feature_set": requested, "requested_model": model, "task": task,
                "target_column": target_column, "selection_status": selection.get("selection_status"),
                "model_version": version, "branches": statuses,
                "bundle_fingerprint": bundle_fingerprint, "dataset_manifest_sha256": _dataset_manifest(data),
                "fit_rows": len(fit), "calibration_rows": len(cal),
                "fit_calibration_subject_overlap_allowed": True,
                "prediction_rows_filtered": False,
                "interpretation": "Local questionnaire monitoring; clinical validity not established"}
    return MonitoringBundle(requested, task, fitted_branches, metadata)


def monitoring_regression_metrics(y, score, subjects):
    """Equal-participant MAE/RMSE with an all-labelled coverage denominator."""
    y, score, subjects = np.asarray(y, float), np.asarray(score, float), np.asarray(subjects)
    if y.shape != score.shape or y.shape != subjects.shape or y.ndim != 1:
        raise ValueError("Monitoring regression metrics need aligned vectors")
    labelled = np.isfinite(y)
    valid = labelled & np.isfinite(score)
    total_weights = participant_weights(subjects[labelled]) if labelled.any() else np.empty(0)
    result = {"n_labelled_visits": int(labelled.sum()), "n_predicted_visits": int(valid.sum()),
              "coverage": float(total_weights[np.isfinite(score[labelled])].sum() / total_weights.sum()) if len(total_weights) else None,
              "mae": None, "rmse": None, "status": "no_predictions"}
    if valid.any():
        weights = participant_weights(subjects[valid])
        errors = y[valid] - score[valid]
        result.update(status="ok", mae=float(np.average(np.abs(errors), weights=weights)),
                      rmse=float(np.sqrt(np.average(errors ** 2, weights=weights))))
    return result
