"""Leakage-controlled classical benchmarks with an explicitly sealed test set.

All prediction tables and checkpoints are confidential local artifacts. Nothing
in the development runner reads temporal-test outcomes or makes clinical claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import binary_metrics, participant_weights, select_sensitivity_threshold


@dataclass
class BenchmarkConfig:
    models: tuple[str, ...] = ("prevalence", "demographic", "elasticnet", "missingness", "catboost", "ebm", "tabpfn")
    outer_folds: int = 5
    inner_folds: int = 3
    calibration_fraction: float = 0.20
    target_sensitivity: float = 0.80
    seed: int = 42
    bootstrap_replicates: int = 2000
    observation_feature_columns: tuple[str, ...] | None = None
    protocol_verified: bool = False
    protocol_unresolved: tuple[str, ...] = ("Consent/task/speaker and device metadata require verification.",)
    parameter_grids: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # TabPFN must have public weights cached locally; no inference API is used.
    tabpfn_model_path: str | None = None
    sequence_length: int = 14
    neural_epochs: int = 20
    neural_batch_size: int = 64
    neural_device: str = "cpu"
    n_demographics: int = 0
    neural_demographics: int | None = None
    mtm_source_dir: str | None = None


def _config(config: BenchmarkConfig | Mapping[str, Any] | None) -> BenchmarkConfig:
    result = config if isinstance(config, BenchmarkConfig) else BenchmarkConfig(**(dict(config) if config else {}))
    if result.neural_demographics is not None:
        if result.n_demographics not in (0, result.neural_demographics):
            raise ValueError("Demographic size aliases disagree.")
        result = replace(result, n_demographics=result.neural_demographics)
    if result.outer_folds < 2 or result.inner_folds < 2 or not 0 < result.calibration_fraction < 0.5:
        raise ValueError("Fold counts and calibration fraction are invalid.")
    if not 0 < result.target_sensitivity <= 1:
        raise ValueError("Target sensitivity must lie in (0, 1].")
    if result.sequence_length < 1 or result.neural_epochs < 1 or result.neural_batch_size < 1:
        raise ValueError("Sequence length, epochs and batch size must be positive.")
    if result.n_demographics < 0 or result.bootstrap_replicates < 1:
        raise ValueError("Demographic count and bootstrap count are invalid.")
    return result


def _directory(path: Path) -> None:
    from mindwatch.pipeline import private_output
    path = private_output(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _json(path: Path, data: Any) -> None:
    from mindwatch.pipeline import private_output
    path = private_output(path)
    # Atomic replacement avoids partially written selection/checkpoint metadata.
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    temporary.chmod(0o600)
    temporary.replace(path)


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    from mindwatch.pipeline import private_output
    path = private_output(path)
    # Create privately before a library opens the path (rather than chmod later).
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.close(descriptor)
    frame.to_parquet(path, index=False)
    path.chmod(0o600)


def _checkpoint(path: Path, model: Any) -> str:
    from mindwatch.pipeline import private_output
    path = private_output(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.close(descriptor)
    joblib.dump(model, path)
    path.chmod(0o600)
    return sha256(path.read_bytes()).hexdigest()


def _target(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if column in {"phq9", "gad7", "isi"}:
        maximum = {"phq9": 27, "gad7": 21, "isi": 28}[column]
        valid = np.isfinite(values) & (values >= 0) & (values <= maximum) & (values == np.floor(values))
        threshold = 15 if column == "isi" else 10
        return np.where(valid, (values >= threshold).astype(float), np.nan)
    if np.any(np.isfinite(values) & ~np.isin(values, [0, 1])):
        raise ValueError("The specified target is not binary or a recognized score.")
    return values


def _validate_columns(frame, feature_columns, target_column):
    required = {"sample_id", "subject_id", "index_date", "partition", target_column, *feature_columns}
    if not required.issubset(frame.columns):
        raise ValueError("Required modeling columns are absent.")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Provide a nonempty list of unique feature columns.")
    prefixes = ("sensor__", "audio__", "text__", "ema__", "history__", "demo__", "quality__")
    if target_column in feature_columns or any(not name.startswith(prefixes) for name in feature_columns):
        raise ValueError("Predictors must use approved feature prefixes and exclude targets/identifiers.")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any() or frame["subject_id"].isna().any():
        raise ValueError("Samples must be unique and participant identifiers present.")
    # Partition membership is safe to inspect; outcomes are not read here.
    if frame.groupby("subject_id")["partition"].nunique().gt(1).any():
        # excluded_after_cutoff are later visits of development participants.
        partitions = frame.groupby("subject_id")["partition"].agg(set)
        if any("development" in values and "temporal_test" in values for values in partitions):
            raise ValueError("Participant overlap across development and temporal test is forbidden.")


def _observation_columns(feature_columns: Sequence[str], config: BenchmarkConfig) -> list[str]:
    if config.observation_feature_columns is not None:
        selected = list(config.observation_feature_columns)
        if not set(selected).issubset(feature_columns):
            raise ValueError("Observation columns must be a subset of selected features.")
        return selected
    # Coverage/count/mask summaries describe collection, not an observed signal.
    # In particular, a finite zero coverage cannot turn an empty window into a
    # predictable sample. An explicit override remains available for experiments.
    surrogate_suffixes = ("__coverage", "__count", "__n", "__mask", "__missing", "__missingness", "__availability", "__available", "__observed_fraction", "__quality")
    return [name for name in feature_columns if name.startswith(("sensor__", "audio__", "text__", "ema__", "history__")) and not name.endswith(surrogate_suffixes)]


def _eligible(frame: pd.DataFrame, feature_columns, config) -> np.ndarray:
    study_gate = frame["quality__sensor_eligible"].eq(1).to_numpy() if "quality__sensor_eligible" in frame else np.ones(len(frame), dtype=bool)
    sequence_names = list(feature_columns[:-config.n_demographics]) if config.n_demographics else list(feature_columns)
    if config.observation_feature_columns is None and sequence_names and all(name.startswith("sensor__seq_") for name in sequence_names):
        if len(sequence_names) % (3 * config.sequence_length):
            raise ValueError("Packed sequence width must be 3 * sequence_length * n_features.")
        packed = frame[sequence_names].to_numpy(dtype=float)
        width = packed.shape[1] // 3
        values, mask = packed[:, :width], packed[:, width:2 * width]
        return study_gate & (np.isfinite(values) & np.isfinite(mask) & (mask > 0)).any(axis=1)
    observations = _observation_columns(feature_columns, config)
    if not observations:
        # Explicitly demo-only experiments can opt in via observation columns.
        return np.zeros(len(frame), dtype=bool)
    values = frame[observations].to_numpy(dtype=float)
    return study_gate & np.isfinite(values).any(axis=1)


def _fingerprint(frame: pd.DataFrame, feature_columns, target_column) -> str:
    columns = ["sample_id", "subject_id", "index_date", target_column, *feature_columns]
    sorted_frame = frame[columns].sort_values("sample_id").reset_index(drop=True)
    return sha256(pd.util.hash_pandas_object(sorted_frame, index=False).to_numpy().tobytes()).hexdigest()


def adaptive_group_folds(y, subjects, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Try fewer participant-grouped folds until every train/validation has both classes."""
    y, subjects = np.asarray(y, dtype=int), np.asarray(subjects)
    if len(np.unique(y)) < 2:
        return []
    n_groups = len(np.unique(subjects))
    supports = [len(np.unique(subjects[y == label])) for label in (0, 1)]
    maximum = min(n_splits, n_groups, *supports)
    for folds in range(maximum, 1, -1):
        for attempt in range(10):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + attempt)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                splits = list(splitter.split(np.zeros(len(y)), y, groups=subjects))
            if all(len(np.unique(y[train])) == 2 and len(np.unique(y[valid])) == 2 for train, valid in splits):
                return splits
    return []


def participant_calibration_split(y, subjects, fraction=0.2, seed=42):
    """Disjoint fit/calibration participants, both classes on each side."""
    y, subjects = np.asarray(y), np.asarray(subjects)
    if len(np.unique(subjects)) < 4 or len(np.unique(y)) < 2:
        return None
    splitter = GroupShuffleSplit(n_splits=100, test_size=fraction, random_state=seed)
    for train, calibration in splitter.split(np.zeros(len(y)), y, groups=subjects):
        if len(np.unique(y[train])) == 2 and len(np.unique(y[calibration])) == 2:
            return train, calibration
    return None


class MissingnessTransformer(BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return (~np.isfinite(np.asarray(X, dtype=float))).astype(float)


def _grids(model: str, config: BenchmarkConfig):
    if model in config.parameter_grids:
        return config.parameter_grids[model]
    return {
        "prevalence": [{}], "demographic": [{"C": 0.1}, {"C": 1.0}],
        "elasticnet": [{"C": c, "l1_ratio": ratio} for c in (0.01, 0.1, 1.0) for ratio in (0.0, 0.5)],
        "missingness": [{"C": 0.1}, {"C": 1.0}],
        "catboost": [{"depth": depth, "l2_leaf_reg": regularization} for depth in (3, 5) for regularization in (3.0, 10.0)],
        "ebm": [{"interactions": 0}, {"interactions": 3}],
        "tabpfn": [{}],
        "grud": [{"hidden_dim": 64, "dropout": 0.2}],
        "tcn": [{"hidden_dim": 64, "dropout": 0.2}],
        "mtm": [{"hidden_dim": 64, "dropout": 0.2}],
    }.get(model, [])


def model_status(model: str, config: BenchmarkConfig) -> dict[str, Any]:
    if model in {"grud", "tcn", "mtm"}:
        if importlib.util.find_spec("torch") is None:
            return {"status": "dependency_unavailable", "dependency": "torch"}
        if model == "mtm":
            if importlib.util.find_spec("mindwatch.research.mtm_adapter") is None:
                return {"status": "official_adapter_unavailable", "dependency": "official_MTM"}
            if not config.mtm_source_dir or not Path(config.mtm_source_dir).is_dir():
                return {"status": "official_source_unavailable", "dependency": "official_MTM"}
            missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
            if missing:
                return {"status": "dependency_unavailable", "dependency": ",".join(missing)}
            try:
                from .mtm_adapter import OfficialMTMClassifier, verify_mtm_source
                provenance = verify_mtm_source(config.mtm_source_dir)
            except Exception as exc:
                return {"status": "official_source_or_adapter_unusable", "dependency": "official_MTM", "exception_type": type(exc).__name__}
            return {"status": "available", "dependency": "official_MTM", "provenance": provenance}
        try:
            from .neural import SensorNeuralClassifier
            if "groups" not in inspect.signature(SensorNeuralClassifier.fit).parameters:
                return {"status": "participant_group_interface_unavailable", "dependency": "torch"}
        except Exception as exc:
            return {"status": "dependency_import_failed", "dependency": "torch", "exception_type": type(exc).__name__}
        return {"status": "available", "dependency": "torch", "version": importlib.metadata.version("torch")}
    optional = {"catboost": "catboost", "ebm": "interpret", "tabpfn": "tabpfn"}
    if model not in {"prevalence", "demographic", "elasticnet", "missingness", *optional}:
        return {"status": "unknown_model"}
    if model in optional:
        package = optional[model]
        if importlib.util.find_spec(package) is None:
            return {"status": "dependency_unavailable", "dependency": package}
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # interpret-core supplies the `interpret` import without installing
            # the optional visualization metapackage of that name.
            try:
                version = importlib.metadata.version("interpret-core" if package == "interpret" else package)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        if model == "tabpfn":
            if not config.tabpfn_model_path or not Path(config.tabpfn_model_path).is_file():
                return {"status": "local_weights_unavailable", "dependency": package, "version": version}
            try:
                from tabpfn import TabPFNClassifier
            except Exception as exc:
                return {"status": "dependency_import_failed", "dependency": package, "version": version, "exception_type": type(exc).__name__}
            if "sample_weight" not in inspect.signature(TabPFNClassifier.fit).parameters:
                return {"status": "participant_weighting_unsupported", "dependency": package, "version": version}
        return {"status": "available", "dependency": package, "version": version}
    return {"status": "available", "dependency": "scikit-learn", "version": importlib.metadata.version("scikit-learn")}


def _estimator(model: str, params: dict, seed: int, config: BenchmarkConfig):
    if model in {"grud", "tcn"}:
        from .neural import SensorNeuralClassifier
        return SensorNeuralClassifier(sequence_length=config.sequence_length, n_demographics=config.n_demographics, architecture=model, epochs=config.neural_epochs, batch_size=config.neural_batch_size, seed=seed, device=config.neural_device, **params)
    if model == "mtm":
        from .mtm_adapter import OfficialMTMClassifier
        return OfficialMTMClassifier(source_dir=config.mtm_source_dir, sequence_length=config.sequence_length, n_demographics=config.n_demographics, epochs=config.neural_epochs, batch_size=config.neural_batch_size, seed=seed, device=config.neural_device, **params)
    if model == "prevalence":
        classifier = DummyClassifier(strategy="prior")
    elif model in {"elasticnet", "demographic", "missingness"}:
        # scikit-learn >=1.8 specifies regularization through l1_ratio.
        classifier = LogisticRegression(solver="saga" if model == "elasticnet" else "lbfgs", l1_ratio=params.get("l1_ratio", 0.0), C=params.get("C", 1.0), max_iter=3000, random_state=seed)
    elif model == "catboost":
        from catboost import CatBoostClassifier
        classifier = CatBoostClassifier(iterations=300, learning_rate=0.05, loss_function="Logloss", random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False, **params)
    elif model == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        classifier = ExplainableBoostingClassifier(max_bins=64, outer_bags=4, max_rounds=1000, n_jobs=1, random_state=seed, **params)
    elif model == "tabpfn":
        from tabpfn import TabPFNClassifier
        classifier = TabPFNClassifier(model_path=config.tabpfn_model_path, device="cpu", random_state=seed)
    else:
        raise ValueError("Unknown benchmark model.")
    preprocess = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if model == "missingness":
        preprocess = [("missingness", MissingnessTransformer())]
    if model in {"elasticnet", "demographic", "missingness"}:
        preprocess.append(("scale", StandardScaler()))
    return Pipeline([*preprocess, ("model", classifier)])


def _fit(estimator, X, y, subjects):
    weights = participant_weights(subjects)
    if isinstance(estimator, Pipeline):
        kwargs = {estimator.steps[-1][0] + "__sample_weight": weights}
        if "groups" in inspect.signature(estimator.steps[-1][1].fit).parameters:
            kwargs[estimator.steps[-1][0] + "__groups"] = np.asarray(subjects)
        estimator.fit(X, y, **kwargs)
    else:
        kwargs = {"sample_weight": weights}
        if "groups" in inspect.signature(estimator.fit).parameters:
            kwargs["groups"] = np.asarray(subjects)
        estimator.fit(X, y, **kwargs)
    return estimator


def _probability(estimator, X):
    probabilities = estimator.predict_proba(X)
    classes = estimator.classes_ if hasattr(estimator, "classes_") else estimator.named_steps["model"].classes_
    if 1 not in classes:
        return np.zeros(len(X))
    return np.asarray(probabilities[:, list(classes).index(1)], dtype=float)


def _tune(X, y, subjects, model, config, seed):
    folds = adaptive_group_folds(y, subjects, config.inner_folds, seed)
    if not folds:
        return None, {"status": "insufficient_inner_class_support", "folds": 0}
    scored = []
    for params in _grids(model, config):
        scores = []
        try:
            for fit, valid in folds:
                estimator = _fit(_estimator(model, params, seed, config), X[fit], y[fit], subjects[fit])
                score = binary_metrics(y[valid], _probability(estimator, X[valid]), subjects[valid], threshold=None)["auroc"]
                scores.append(score)
        except Exception as exc:
            # Never include exception strings: libraries may echo source values.
            scored.append({"parameters": params, "status": "fit_failed", "exception_type": type(exc).__name__})
            continue
        scored.append({"parameters": params, "status": "ok", "mean_auroc": float(np.mean(scores))})
    successes = [record for record in scored if record["status"] == "ok"]
    if not successes:
        return None, {"status": "all_candidates_failed", "folds": len(folds), "candidates": scored}
    best = max(successes, key=lambda record: record["mean_auroc"])
    return best["parameters"], {"status": "ok", "folds": len(folds), "candidates": scored}


def _calibrator(y, p, subjects):
    from scipy.special import logit
    transformed = logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    weights = participant_weights(subjects)
    model.fit(transformed, y, sample_weight=weights)
    return model


def _calibrated(model, p):
    from scipy.special import logit
    return model.predict_proba(logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1]


def one_se_selection(records: Sequence[Mapping[str, Any]], complexity: Sequence[str] = ("prevalence", "demographic", "missingness", "elasticnet", "catboost", "ebm", "tabpfn", "grud", "tcn", "mtm")) -> dict[str, Any]:
    """Choose the simplest model within one standard error of the best mean AUROC."""
    valid = [record for record in records if record.get("status") == "ok" and record.get("fold_aurocs") and all(score is not None and np.isfinite(score) for score in record["fold_aurocs"])]
    if not valid:
        return {"status": "no_selectable_model", "selected_model": None}
    summaries = []
    for record in valid:
        scores = np.asarray(record["fold_aurocs"], dtype=float)
        summaries.append({"model": record["model"], "mean_auroc": float(scores.mean()), "se": float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0})
    best = max(summaries, key=lambda value: value["mean_auroc"])
    cutoff = best["mean_auroc"] - best["se"]
    eligible = [value for value in summaries if value["mean_auroc"] >= cutoff - 1e-12]
    selected = min(eligible, key=lambda value: (complexity.index(value["model"]) if value["model"] in complexity else len(complexity), -value["mean_auroc"]))
    return {"status": "ok", "selected_model": selected["model"], "best_mean_model": best["model"], "one_se_cutoff": cutoff, "model_summaries": summaries, "interpretation": "CV standard error is a selection heuristic, not an independent confidence interval."}


def run_development(frame: pd.DataFrame, feature_columns: Sequence[str], target_column: str, output_dir: str | Path, config: BenchmarkConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Nested grouped CV, then freeze a development-only fit and operating point.

    The returned object contains aggregate results only. Development OOF tables
    and model checkpoints are saved privately. An existing selection is never
    overwritten; use a new versioned directory for a changed experiment.
    """
    cfg, out = _config(config), Path(output_dir)
    feature_columns = list(feature_columns)
    _validate_columns(frame, feature_columns, target_column)
    if set(cfg.models) & {"grud", "tcn", "mtm"}:
        sequence_names = feature_columns[:-cfg.n_demographics] if cfg.n_demographics else feature_columns
        demographic_names = feature_columns[-cfg.n_demographics:] if cfg.n_demographics else []
        if not sequence_names or not all(name.startswith("sensor__seq_") for name in sequence_names) or len(sequence_names) % (3 * cfg.sequence_length) or not all(name.startswith("demo__") for name in demographic_names):
            raise ValueError("Neural benchmarks require packed sensor__seq_ values/mask/delta and an explicitly sized demo__ tail.")
    if (out / "selection.json").exists():
        raise FileExistsError("This experiment already has a frozen selection; use a new output directory.")
    _directory(out)
    # Deliberately subset before extracting the target or numeric feature values.
    modeling_columns = ["sample_id", "subject_id", "index_date", target_column, *feature_columns]
    if "quality__sensor_eligible" in frame and "quality__sensor_eligible" not in modeling_columns:
        modeling_columns.append("quality__sensor_eligible")
    development = frame.loc[frame["partition"].eq("development"), modeling_columns].copy().reset_index(drop=True)
    if development.empty:
        raise ValueError("No development data are available.")
    fingerprint = _fingerprint(development, feature_columns, target_column)
    y_all = _target(development, target_column)
    eligible = np.isfinite(y_all) & _eligible(development, feature_columns, cfg)
    data = development.loc[eligible].reset_index(drop=True)
    y, subjects = y_all[eligible].astype(int), data["subject_id"].to_numpy()
    X = data[feature_columns].to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    outer = adaptive_group_folds(y, subjects, cfg.outer_folds, cfg.seed)
    report: dict[str, Any] = {"status": "running", "analysis_type": "exploratory" if not cfg.protocol_verified else "prespecified_protocol", "protocol_verified": cfg.protocol_verified, "protocol_unresolved": list(cfg.protocol_unresolved) if not cfg.protocol_verified else [], "target": target_column, "n_development_visits": len(development), "n_eligible_visits": len(data), "n_eligible_participants": int(data["subject_id"].nunique()), "outer_folds_requested": cfg.outer_folds, "outer_folds_used": len(outer), "temporal_test_evaluated": False, "models": [], "config": asdict(cfg), "cohort_eligibility_summary": frame.attrs.get("cohort_eligibility_summary")}
    if not outer:
        report["status"] = "insufficient_outer_class_support"
        _json(out / "development_report.json", report)
        return report
    # Identical calibration/fit and outer validation groups across every model.
    split_cache = []
    for fold, (train, valid) in enumerate(outer):
        calibration = participant_calibration_split(y[train], subjects[train], cfg.calibration_fraction, cfg.seed + fold)
        if calibration is None:
            report["status"] = "insufficient_calibration_class_support"
            _json(out / "development_report.json", report)
            return report
        fit_local, calibration_local = calibration
        split_cache.append((train[fit_local], train[calibration_local], valid))
    predictions, split_rows = [], []
    for fold, (fit, calibration, valid) in enumerate(split_cache):
        for role, indices in (("fit", fit), ("calibration", calibration), ("validation", valid)):
            split_rows.append(pd.DataFrame({"sample_id": data.iloc[indices]["sample_id"].to_numpy(), "subject_id": subjects[indices], "outer_fold": fold, "role": role}))
    _parquet(out / "development_cv_manifest.parquet", pd.concat(split_rows, ignore_index=True))
    for model in cfg.models:
        dependency = model_status(model, cfg)
        record: dict[str, Any] = {"model": model, **dependency, "folds": [], "fold_aurocs": []}
        selected_columns = [i for i, name in enumerate(feature_columns) if name.startswith("demo__")] if model == "demographic" else list(range(len(feature_columns)))
        if not selected_columns:
            record["status"] = "demographic_features_unavailable"
        if dependency["status"] != "available" or not selected_columns:
            report["models"].append(record)
            continue
        xx = X[:, selected_columns]
        for fold, (fit, calibration, valid) in enumerate(split_cache):
            try:
                parameters, tuning = _tune(xx[fit], y[fit], subjects[fit], model, cfg, cfg.seed + fold)
                if parameters is None:
                    record["folds"].append({"fold": fold, **tuning})
                    break
                estimator = _fit(_estimator(model, parameters, cfg.seed + fold, cfg), xx[fit], y[fit], subjects[fit])
                calibration_raw = _probability(estimator, xx[calibration])
                calibrator = _calibrator(y[calibration], calibration_raw, subjects[calibration])
                calibration_p = _calibrated(calibrator, calibration_raw)
                threshold_info = select_sensitivity_threshold(y[calibration], calibration_p, subjects[calibration], cfg.target_sensitivity)
                if threshold_info["threshold"] is None:
                    raise ValueError("Calibration operating point unavailable.")
                p = _calibrated(calibrator, _probability(estimator, xx[valid]))
                metrics = binary_metrics(y[valid], p, subjects[valid], threshold_info["threshold"])
                record["folds"].append({"fold": fold, "status": "ok", "parameters": parameters, "tuning": tuning, "metrics": metrics, "operating_point": threshold_info})
                record["fold_aurocs"].append(metrics["auroc"])
                predictions.append(pd.DataFrame({"sample_id": data.iloc[valid]["sample_id"].to_numpy(), "subject_id": subjects[valid], "model": model, "outer_fold": fold, "target": y[valid], "probability": p, "threshold": threshold_info["threshold"]}))
            except Exception as exc:
                record["folds"].append({"fold": fold, "status": "fit_failed", "exception_type": type(exc).__name__})
                break
        record["status"] = "ok" if len(record["fold_aurocs"]) == len(outer) else "incomplete_cv"
        report["models"].append(record)
        # Persist progress without participant-level results in the returned report.
        _json(out / "development_report.json", report)
    if predictions:
        oof = pd.concat(predictions, ignore_index=True)
        _parquet(out / "oof_predictions.parquet", oof)
        for record in report["models"]:
            if record["status"] == "ok":
                predicted = oof.loc[oof["model"].eq(record["model"])]
                record["pooled_oof_metrics"] = binary_metrics(predicted["target"], predicted["probability"], predicted["subject_id"], predicted["threshold"])
                record["eligible_labelled_coverage"] = float(len(predicted) / np.isfinite(y_all).sum()) if np.isfinite(y_all).sum() else None
    selection = one_se_selection(report["models"])
    report["selection"] = selection
    if selection["status"] != "ok":
        report["status"] = "no_selectable_model"
        _json(out / "development_report.json", report)
        return report
    # Final development refit has a fresh, participant-held-out calibration set.
    full_split = participant_calibration_split(y, subjects, cfg.calibration_fraction, cfg.seed + 10000)
    if full_split is None:
        report["status"] = "insufficient_final_calibration"
        _json(out / "development_report.json", report)
        return report
    fit, calibration = full_split
    model = selection["selected_model"]
    selected_features = [name for name in feature_columns if name.startswith("demo__")] if model == "demographic" else feature_columns
    xx = data[selected_features].to_numpy(dtype=float, copy=True)
    xx[~np.isfinite(xx)] = np.nan
    parameters, tuning = _tune(xx[fit], y[fit], subjects[fit], model, cfg, cfg.seed + 10000)
    if parameters is None:
        report["status"] = "final_development_tuning_failed"
        _json(out / "development_report.json", report)
        return report
    try:
        estimator = _fit(_estimator(model, parameters, cfg.seed, cfg), xx[fit], y[fit], subjects[fit])
        p_calibration_raw = _probability(estimator, xx[calibration])
        calibrator = _calibrator(y[calibration], p_calibration_raw, subjects[calibration])
        operating = select_sensitivity_threshold(y[calibration], _calibrated(calibrator, p_calibration_raw), subjects[calibration], cfg.target_sensitivity)
        if operating["threshold"] is None:
            raise ValueError("Final operating point unavailable.")
    except Exception as exc:
        report.update(status="final_development_fit_failed", exception_type=type(exc).__name__)
        _json(out / "development_report.json", report)
        return report
    frozen = {"model": estimator, "calibrator": calibrator, "threshold": operating["threshold"], "feature_columns": selected_features}
    checkpoint_hash = _checkpoint(out / "frozen_development_model.joblib", frozen)
    final_roles = pd.DataFrame({"sample_id": data["sample_id"], "subject_id": subjects, "role": "fit"})
    final_roles.loc[calibration, "role"] = "calibration"
    _parquet(out / "frozen_fit_manifest.parquet", final_roles)
    selection_artifact = {**selection, "sealed_selection": True, "temporal_test_evaluated": False, "target_column": target_column, "feature_columns": feature_columns, "model_feature_columns": selected_features, "parameters": parameters, "operating_point": operating, "tuning": tuning, "development_fingerprint": fingerprint, "checkpoint_sha256": checkpoint_hash, "config": asdict(cfg), "analysis_type": report["analysis_type"], "protocol_unresolved": report["protocol_unresolved"]}
    # Written last: its presence means the corresponding model and threshold exist.
    _json(out / "selection.json", selection_artifact)
    report.update(status="selection_frozen", selection_artifact="selection.json")
    _json(out / "development_report.json", report)
    return report


def finalize(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    """Explicit, once-only temporal-test evaluation of a frozen selection.

    No fitting, threshold changes, feature selection or model ranking happens
    here. Checkpoints must come from this trusted local experiment directory.
    """
    out = Path(output_dir)
    if not (out / "selection.json").is_file():
        raise ValueError("Development selection must be persisted before temporal evaluation.")
    if (out / "final_results.json").exists():
        raise FileExistsError("Temporal results already exist for this frozen experiment.")
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if not selection.get("sealed_selection"):
        raise ValueError("Development selection is incomplete.")
    cfg, target = _config(selection["config"]), selection["target_column"]
    features = selection["feature_columns"]
    _validate_columns(frame, features, target)
    development = frame.loc[frame["partition"].eq("development")]
    if _fingerprint(development, features, target) != selection["development_fingerprint"]:
        raise ValueError("Development data changed after selection; create a new experiment.")
    checkpoint = out / "frozen_development_model.joblib"
    if not checkpoint.is_file() or sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Frozen model integrity check failed.")
    frozen = joblib.load(checkpoint)
    if frozen["threshold"] != selection["operating_point"]["threshold"] or frozen["feature_columns"] != selection["model_feature_columns"]:
        raise ValueError("Frozen model and selection metadata disagree.")
    # Temporal values and labels are accessed only after the persisted gate.
    temporal = frame.loc[frame["partition"].eq("temporal_test")].copy().reset_index(drop=True)
    if temporal.empty:
        raise ValueError("No temporal-test data are available.")
    y = _target(temporal, target)
    eligible = _eligible(temporal, features, cfg)
    p = np.full(len(temporal), np.nan)
    if eligible.any():
        xx = temporal.loc[eligible, frozen["feature_columns"]].to_numpy(dtype=float, copy=True)
        xx[~np.isfinite(xx)] = np.nan
        p[eligible] = _calibrated(frozen["calibrator"], _probability(frozen["model"], xx))
    metrics = binary_metrics(y, p, temporal["subject_id"], frozen["threshold"])
    # Marginal cluster-bootstrap CIs use the same locked probabilities.
    intervals = {}
    for name in ("auroc", "auprc", "brier"):
        intervals[name] = _single_cluster_interval(y, p, temporal["subject_id"], name, cfg.bootstrap_replicates, cfg.seed)
    status = np.where(eligible, "predicted", "abstain_no_observed_modality")
    if "quality__sensor_eligible" in temporal:
        status = np.where(temporal["quality__sensor_eligible"].eq(1), status, "abstain_study_sensor_eligibility")
    table = pd.DataFrame({"sample_id": temporal["sample_id"], "subject_id": temporal["subject_id"], "target": y, "probability": p, "threshold": frozen["threshold"], "status": status})
    _parquet(out / "temporal_predictions.parquet", table)
    result = {"status": "finalized", "selected_model": selection["selected_model"], "threshold": frozen["threshold"], "metrics": metrics, "confidence_intervals": intervals, "analysis_type": selection["analysis_type"], "protocol_unresolved": selection.get("protocol_unresolved", []), "interpretation": "Questionnaire-defined risk screening; software evaluation does not establish clinical validity.", "model_checkpoint_sha256": selection["checkpoint_sha256"], "temporal_test_evaluated": True, "coverage_denominator": "all_labelled_temporal_visits_supplied_by_finalize_loader", "cohort_eligibility_summary": frame.attrs.get("cohort_eligibility_summary")}
    _json(out / "final_results.json", result)
    return result


def _single_cluster_interval(y, p, subjects, metric, replicates, seed):
    from .evaluation import _metric
    y, p, subjects = np.asarray(y), np.asarray(p), np.asarray(subjects)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p, subjects = y[mask].astype(int), p[mask], subjects[mask]
    codes, unique = pd.factorize(subjects, sort=False)
    if len(unique) < 2:
        return {"status": "insufficient_participants", "lower": None, "upper": None}
    weights, rng, values = participant_weights(subjects), np.random.default_rng(seed), []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        ww = weights * counts[codes]
        included = ww > 0
        value = _metric(y[included], p[included], ww[included], metric)
        if value is not None:
            values.append(value)
    if not values:
        return {"status": "no_valid_resamples", "lower": None, "upper": None}
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"status": "ok", "lower": float(lower), "upper": float(upper), "n_valid": len(values), "n_bootstrap": replicates, "method": "participant_cluster_percentile"}


class OOFStackingClassifier:
    """Training-only, participant-grouped OOF stacking utility.

    Fit this object *inside* each outer fit partition, never on a global OOF
    table used for evaluation. Caller handles separate calibration as above.
    """
    def __init__(self, base_estimators: Mapping[str, Any], inner_folds: int = 3, seed: int = 42):
        self.base_estimators = dict(base_estimators)
        self.inner_folds, self.seed = inner_folds, seed

    def fit(self, X, y, subjects):
        X, y, subjects = np.asarray(X), np.asarray(y), np.asarray(subjects)
        if not self.base_estimators:
            raise ValueError("At least one base estimator is required.")
        splits = adaptive_group_folds(y, subjects, self.inner_folds, self.seed)
        if not splits:
            raise ValueError("Insufficient participant class support for inner OOF stacking.")
        oof = np.full((len(y), len(self.base_estimators)), np.nan)
        self.estimators_ = []
        for column, template in enumerate(self.base_estimators.values()):
            for train, valid in splits:
                model = _fit(clone(template), X[train], y[train], subjects[train])
                oof[valid, column] = _probability(model, X[valid])
            self.estimators_.append(_fit(clone(template), X, y, subjects))
        if not np.isfinite(oof).all():
            raise ValueError("OOF predictions are incomplete.")
        self.meta_model_ = _fit(LogisticRegression(C=0.1, max_iter=1000), oof, y, subjects)
        self.classes_ = self.meta_model_.classes_
        self.oof_predictions_ = oof
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([_probability(model, X) for model in self.estimators_])
        return self.meta_model_.predict_proba(meta_features)
