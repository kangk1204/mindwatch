"""Executable, participant-nested OOF stacking of local modality classifiers.

Inputs are already reviewed numeric features. This module never reads source
records, transcripts or recordings, and never inspects temporal labels during
development. Fixed base/meta settings make this a prespecified comparator, not
an additional adaptive model search.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from .benchmarks import (
    BenchmarkConfig, _calibrated, _calibrator, _checkpoint, _directory,
    _estimator, _fingerprint, _fit, _json, _observation_columns, _parquet,
    _probability, _single_cluster_interval, _target, _validate_columns,
    adaptive_group_folds, model_status, participant_calibration_split,
)
from .evaluation import binary_metrics, paired_cluster_bootstrap, select_sensitivity_threshold


@dataclass
class StackingConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    calibration_fraction: float = 0.20
    target_sensitivity: float = 0.80
    meta_C: float = 0.1
    seed: int = 42
    bootstrap_replicates: int = 2000
    baseline_modality: str = "sensor"
    cohort_mode: str = "available_modalities"
    base_models: dict[str, str] = field(default_factory=dict)
    base_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    protocol_verified: bool = False
    protocol_unresolved: tuple[str, ...] = ("Numeric inputs require reviewed modality/time provenance.",)


def _config(config) -> StackingConfig:
    cfg = config if isinstance(config, StackingConfig) else StackingConfig(**dict(config or {}))
    if cfg.outer_folds < 2 or cfg.inner_folds < 2 or not 0 < cfg.calibration_fraction < .5:
        raise ValueError("Stacking fold counts and calibration fraction are invalid.")
    if cfg.meta_C <= 0 or cfg.bootstrap_replicates < 1 or not 0 < cfg.target_sensitivity <= 1:
        raise ValueError("Stacking regularization, bootstrap count or sensitivity is invalid.")
    if cfg.cohort_mode not in {"available_modalities", "complete_case"}:
        raise ValueError("Stacking cohort mode is invalid.")
    return cfg


class StackingUnavailable(ValueError):
    """A fixed, non-sensitive reason code for unavailable fits."""
    def __init__(self, code, details=None):
        self.code = code
        self.details = details
        super().__init__(code)


def _groups(modality_columns, cfg):
    if len(modality_columns) < 2 or cfg.baseline_modality not in modality_columns:
        raise ValueError("Provide at least two modalities and an explicit baseline modality.")
    result = {name: list(columns) for name, columns in modality_columns.items()}
    if not (set(cfg.base_models) | set(cfg.base_parameters)).issubset(result):
        raise ValueError("Base model/parameter mappings contain an undeclared modality.")
    for name, columns in result.items():
        if name not in {"sensor", "audio", "text", "ema", "history"} or not columns or len(set(columns)) != len(columns):
            raise ValueError("Modality names/feature sets are invalid.")
        if any(not col.startswith((name + "__", "demo__", "quality__")) for col in columns):
            raise ValueError("Each modality may contain only its own features and explicit covariates.")
        if not _observation_columns(columns, BenchmarkConfig()):
            raise ValueError("A modality must contain actual observations, not quality/covariates alone.")
        if cfg.base_models.get(name, "elasticnet") not in {"elasticnet", "catboost", "ebm"}:
            raise ValueError("Stacking supports fixed elasticnet, CatBoost or EBM base classifiers.")
    return result


def _columns(groups):
    return list(dict.fromkeys(col for names in groups.values() for col in names))


def _available(frame, columns, modality):
    observed = _observation_columns(columns, BenchmarkConfig())
    valid = np.isfinite(frame[observed].to_numpy(dtype=float)).any(axis=1)
    # A missing sensor window can coexist with an available reviewed voice.
    # Sensor study eligibility does not discard that voice-only observation.
    if modality == "sensor" and "quality__sensor_eligible" in frame:
        valid &= frame["quality__sensor_eligible"].eq(1).to_numpy()
    return valid


def _matrix(frame, columns):
    values = frame[columns].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _parameters(modality, cfg):
    if modality in cfg.base_parameters:
        return cfg.base_parameters[modality]
    return {"elasticnet": {"C": .1, "l1_ratio": .5}, "catboost": {"depth": 3, "l2_leaf_reg": 10.0}, "ebm": {"interactions": 0}}[cfg.base_models.get(modality, "elasticnet")]


def _meta_features(base_probabilities):
    present = np.isfinite(base_probabilities)
    scores = np.where(present, logit(np.clip(base_probabilities, 1e-7, 1-1e-7)), 0.)
    return np.column_stack([scores, present.astype(float)])


class FrozenModalityStack:
    """Full fit-partition base models and a training-internal OOF meta model."""
    def __init__(self, modality_columns, base_models, meta_model, require_sensor_eligibility=False):
        self.modality_columns = modality_columns
        self.base_models = base_models
        self.meta_model = meta_model
        self.require_sensor_eligibility = require_sensor_eligibility
        self.classes_ = np.array([0, 1])

    def base_probabilities(self, frame):
        if self.require_sensor_eligibility and "quality__sensor_eligible" not in frame:
            raise ValueError("The frozen sensor study eligibility flag is required.")
        result = np.full((len(frame), len(self.modality_columns)), np.nan)
        for column, (name, features) in enumerate(self.modality_columns.items()):
            model = self.base_models.get(name)
            if model is None:
                continue
            observed = _available(frame, features, name)
            if observed.any():
                result[observed, column] = _probability(model, _matrix(frame.loc[observed], features))
        return result

    def raw_probabilities(self, frame):
        base = self.base_probabilities(frame)
        present = np.isfinite(base).any(axis=1)
        stacked = np.full(len(frame), np.nan)
        if present.any():
            stacked[present] = self.meta_model.predict_proba(_meta_features(base[present]))[:, 1]
        return stacked, base

    def predict_proba(self, frame):
        p, _ = self.raw_probabilities(frame)
        return np.column_stack([1-p, p])


def _fit_stack(frame, y, groups, cfg, seed):
    subjects = frame["subject_id"].to_numpy()
    inner = adaptive_group_folds(y, subjects, cfg.inner_folds, seed)
    if not inner:
        raise StackingUnavailable("insufficient_inner_participant_class_support")
    oof = np.full((len(frame), len(groups)), np.nan)
    full_models, statuses, manifests = {}, {}, []
    common_membership = np.full(len(frame), -1, dtype=int)
    for fold, (train, valid) in enumerate(inner):
        common_membership[valid] = fold
        for role, rows in (("fit", train), ("validation", valid)):
            manifests.append(pd.DataFrame({"sample_id": frame.iloc[rows]["sample_id"].to_numpy(), "subject_id": subjects[rows], "inner_fold": fold, "role": role}))
    for column, (name, features) in enumerate(groups.items()):
        x, available = _matrix(frame, features), _available(frame, features, name)
        model_name, params = cfg.base_models.get(name, "elasticnet"), _parameters(name, cfg)
        failed, class_insufficient, exception_types = 0, 0, set()
        for train, valid in inner:
            fit = train[available[train]]
            held = valid[available[valid]]
            if not len(held):
                continue
            if len(np.unique(y[fit])) < 2:
                class_insufficient += 1
                continue
            try:
                estimator = _fit(_estimator(model_name, params, seed, BenchmarkConfig()), x[fit], y[fit], subjects[fit])
                oof[held, column] = _probability(estimator, x[held])
            except Exception as exc:
                failed += 1
                exception_types.add(type(exc).__name__)
        fit = np.flatnonzero(available)
        full_status = "insufficient_modality_class_support"
        if len(np.unique(y[fit])) == 2:
            try:
                full_models[name] = _fit(_estimator(model_name, params, seed, BenchmarkConfig()), x[fit], y[fit], subjects[fit])
                full_status = "ok"
            except Exception as exc:
                full_status = "base_fit_failed"
                exception_types.add(type(exc).__name__)
        statuses[name] = {"model": model_name, "full_fit_status": full_status, "inner_fit_failures": failed, "inner_class_support_failures": class_insufficient, "exception_types": sorted(exception_types), "n_inner_oof_predictions": int(np.isfinite(oof[:, column]).sum())}
        if full_status == "base_fit_failed":
            raise StackingUnavailable("base_full_fit_failure", statuses)
    usable = np.isfinite(oof).any(axis=1)
    if len(np.unique(y[usable])) < 2:
        raise StackingUnavailable("insufficient_meta_oof_class_support")
    if len(full_models) < 2 or int(np.isfinite(oof).any(axis=0).sum()) < 2:
        raise StackingUnavailable("fewer_than_two_trainable_modalities")
    if any(status["inner_fit_failures"] for status in statuses.values()):
        raise StackingUnavailable("base_inner_fit_failure", statuses)
    meta = _fit(LogisticRegression(C=cfg.meta_C, max_iter=3000, solver="lbfgs"), _meta_features(oof[usable]), y[usable], subjects[usable])
    ensemble = FrozenModalityStack(groups, full_models, meta, require_sensor_eligibility="sensor" in groups and "quality__sensor_eligible" in frame)
    proof = pd.DataFrame({"sample_id": frame["sample_id"].to_numpy(), "subject_id": subjects, "target": y, "inner_fold": common_membership})
    for column, name in enumerate(groups):
        proof[f"base_probability__{name}"] = oof[:, column]
    details = {"inner_folds": len(inner), "n_meta_training_visits": int(usable.sum()), "n_meta_training_participants": int(pd.Series(subjects[usable]).nunique()), "base_status": statuses}
    return ensemble, details, proof, pd.concat(manifests, ignore_index=True)


def _calibration(y, raw, subjects, cfg):
    usable = np.isfinite(raw)
    if len(np.unique(y[usable])) < 2:
        return None, {"status": "insufficient_calibration_class_support", "threshold": None}
    calibrator = _calibrator(y[usable], raw[usable], subjects[usable])
    point = select_sensitivity_threshold(y[usable], _calibrated(calibrator, raw[usable]), subjects[usable], cfg.target_sensitivity)
    return calibrator, point


def _fit_bundle(fit_frame, y_fit, calibration_frame, y_calibration, groups, cfg, seed):
    original_fit_count, original_calibration_count = len(fit_frame), len(calibration_frame)
    if cfg.cohort_mode == "complete_case":
        fit_common = np.column_stack([_available(fit_frame, columns, name) for name, columns in groups.items()]).all(axis=1)
        calibration_common = np.column_stack([_available(calibration_frame, columns, name) for name, columns in groups.items()]).all(axis=1)
        fit_frame, y_fit = fit_frame.loc[fit_common], y_fit[fit_common]
        calibration_frame, y_calibration = calibration_frame.loc[calibration_common], y_calibration[calibration_common]
    ensemble, details, proof, membership = _fit_stack(fit_frame, y_fit, groups, cfg, seed)
    details.update(cohort_mode=cfg.cohort_mode, n_parent_fit_visits=original_fit_count, n_base_fit_cohort_visits=len(fit_frame), n_parent_calibration_visits=original_calibration_count, n_calibration_cohort_visits=len(calibration_frame))
    stacked, base = ensemble.raw_probabilities(calibration_frame)
    subjects = calibration_frame["subject_id"].to_numpy()
    calibrators, operating = {}, {}
    baseline_index = list(groups).index(cfg.baseline_modality)
    for name, raw in (("stacked", stacked), ("base_" + cfg.baseline_modality, base[:, baseline_index])):
        calibrators[name], operating[name] = _calibration(y_calibration, raw, subjects, cfg)
    if calibrators["stacked"] is None:
        raise StackingUnavailable("stacked_calibration_class_support_unavailable")
    bundle = {"ensemble": ensemble, "calibrators": calibrators, "operating_points": operating, "baseline_modality": cfg.baseline_modality, "cohort_mode": cfg.cohort_mode}
    return bundle, details, proof, membership


def _score(bundle, frame):
    stacked, base = bundle["ensemble"].raw_probabilities(frame)
    index = list(bundle["ensemble"].modality_columns).index(bundle["baseline_modality"])
    output = {}
    for name, raw in (("stacked", stacked), ("base_" + bundle["baseline_modality"], base[:, index])):
        probability = np.full(len(frame), np.nan)
        threshold = bundle["operating_points"][name]["threshold"]
        available = np.isfinite(raw)
        if bundle["calibrators"][name] is not None and available.any():
            probability[available] = _calibrated(bundle["calibrators"][name], raw[available])
        if bundle["cohort_mode"] == "complete_case":
            probability[~np.isfinite(base).all(axis=1)] = np.nan
        output[name] = (probability, threshold)
    return output, np.isfinite(base)


def _report_predictions(table, baseline, cfg):
    models = []
    for name, rows in table.groupby("model", sort=False):
        threshold = rows["threshold"].to_numpy(dtype=float)
        models.append({"model": name, **binary_metrics(rows["target"], rows["probability"], rows["subject_id"], threshold)})
    left = table.loc[table.model.eq("stacked")]
    right = table.loc[table.model.eq("base_" + baseline)]
    common = left.merge(right, on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(common.subject_id_a, common.subject_id_b) or not np.array_equal(common.outer_fold_a, common.outer_fold_b):
        raise ValueError("Paired stacking predictions must share participant/fold membership.")
    if not np.array_equal(common.target_a, common.target_b, equal_nan=True):
        raise ValueError("Paired stacking predictions must share identical outcomes.")
    usable = np.isfinite(common.target_a) & np.isfinite(common.probability_a) & np.isfinite(common.probability_b)
    paired = common.loc[usable]
    comparisons = {}
    for metric in ("auroc", "auprc", "brier"):
        comparisons[metric] = paired_cluster_bootstrap(paired.target_a, paired.probability_a, paired.probability_b, paired.subject_id_a, metric=metric, n_bootstrap=cfg.bootstrap_replicates, seed=cfg.seed)
    paired_models = []
    for suffix, name in (("a", "stacked"), ("b", "base_" + baseline)):
        paired_models.append({"model": name, **binary_metrics(paired.target_a, paired["probability_" + suffix], paired.subject_id_a, paired["threshold_" + suffix])})
    return {"full_cohort_models": models, "paired_common_models": paired_models, "paired_comparisons": comparisons, "n_common_predicted_visits": len(paired), "n_common_predicted_participants": int(paired.subject_id_a.nunique()), "coverage_estimand": "each_model_on_all_labelled_rows; paired_comparison_on_identical_common_predicted_visits", "training_cohort_policy": "identical_complete_case_fit_and_calibration_rows" if cfg.cohort_mode == "complete_case" else "modality_specific_observed_fit_rows; paired_difference_also_reflects_training_availability", "inference": "AUROC is primary; AUPRC/Brier comparisons exploratory, no confirmatory multiplicity claim."}


def run_stacking_development(frame: pd.DataFrame, modality_columns: Mapping[str, Sequence[str]], target_column: str, output_dir: str | Path, config: StackingConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run honest outer OOF evaluation and freeze one fixed stacking design.

    All modality base predictions used to train the meta model are generated
    by participant-held-out inner fits. Development-only full refit reserves an
    independent participant calibration set; no temporal outcomes are read.
    """
    cfg = _config(config)
    groups, out = _groups(modality_columns, cfg), Path(output_dir)
    features = _columns(groups)
    _validate_columns(frame, features, target_column)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new versioned stacking experiment directory.")
    _directory(out)
    columns = ["sample_id", "subject_id", "index_date", target_column, *features]
    if "quality__sensor_eligible" in frame and "quality__sensor_eligible" not in columns:
        columns.append("quality__sensor_eligible")
    development = frame.loc[frame.partition.eq("development"), columns].copy().reset_index(drop=True)
    if development.empty:
        raise ValueError("No development rows are available.")
    fingerprint_features = list(features)
    if "quality__sensor_eligible" in development and "quality__sensor_eligible" not in fingerprint_features:
        fingerprint_features.append("quality__sensor_eligible")
    fingerprint = _fingerprint(development, fingerprint_features, target_column)
    y_all = _target(development, target_column)
    labelled = np.isfinite(y_all)
    data, y = development.loc[labelled].reset_index(drop=True), y_all[labelled].astype(int)
    subjects = data["subject_id"].to_numpy()
    report = {"status": "running", "kind": "participant_nested_oof_stacking", "analysis_type": "prespecified_fixed_comparator" if cfg.protocol_verified else "exploratory", "protocol_unresolved": [] if cfg.protocol_verified else list(cfg.protocol_unresolved), "config": asdict(cfg), "n_development_labelled_visits": len(data), "n_development_participants": int(data.subject_id.nunique()), "temporal_test_evaluated": False, "base_dependencies": {}}
    for name in groups:
        dependency = model_status(cfg.base_models.get(name, "elasticnet"), BenchmarkConfig())
        report["base_dependencies"][name] = dependency
        if dependency["status"] != "available":
            report["status"] = "requested_base_dependency_unavailable"
    if report["status"] != "running":
        _json(out / "stacking_development_report.json", report)
        return report
    outer = adaptive_group_folds(y, subjects, cfg.outer_folds, cfg.seed)
    report["outer_folds_used"] = len(outer)
    if not outer:
        report["status"] = "insufficient_outer_class_support"
        _json(out / "stacking_development_report.json", report)
        return report
    predictions, proofs, membership_rows, manifests, fold_details = [], [], [], [], []
    try:
        for fold, (outer_train, valid) in enumerate(outer):
            split = participant_calibration_split(y[outer_train], subjects[outer_train], cfg.calibration_fraction, cfg.seed + fold)
            if split is None:
                raise StackingUnavailable("insufficient_outer_calibration_class_support")
            fit, calibration = [outer_train[rows] for rows in split]
            for role, rows in (("fit", fit), ("calibration", calibration), ("validation", valid)):
                manifests.append(pd.DataFrame({"sample_id": data.iloc[rows]["sample_id"].to_numpy(), "subject_id": subjects[rows], "outer_fold": fold, "role": role}))
            bundle, details, proof, membership = _fit_bundle(data.iloc[fit], y[fit], data.iloc[calibration], y[calibration], groups, cfg, cfg.seed + fold)
            proof["outer_fold"], membership["outer_fold"] = fold, fold
            proofs.append(proof)
            membership_rows.append(membership)
            scored, available = _score(bundle, data.iloc[valid])
            fold_details.append({"outer_fold": fold, **details, "operating_points": bundle["operating_points"]})
            for name, (probability, threshold) in scored.items():
                predictions.append(pd.DataFrame({"sample_id": data.iloc[valid]["sample_id"].to_numpy(), "subject_id": subjects[valid], "outer_fold": fold, "partition": "development", "model": name, "target": y[valid], "probability": probability, "threshold": threshold if threshold is not None else np.nan, "n_available_modalities": available.sum(axis=1)}))
        table = pd.concat(predictions, ignore_index=True)
        _parquet(out / "stacking_oof_predictions.parquet", table)
        _parquet(out / "stacking_inner_oof_predictions.parquet", pd.concat(proofs, ignore_index=True))
        _parquet(out / "stacking_inner_manifest.parquet", pd.concat(membership_rows, ignore_index=True))
        _parquet(out / "stacking_outer_manifest.parquet", pd.concat(manifests, ignore_index=True))
        report.update(_report_predictions(table, cfg.baseline_modality, cfg))
        report["folds"] = fold_details
        final_split = participant_calibration_split(y, subjects, cfg.calibration_fraction, cfg.seed + 10000)
        if final_split is None:
            raise StackingUnavailable("insufficient_final_calibration_class_support")
        fit, calibration = final_split
        bundle, details, proof, membership = _fit_bundle(data.iloc[fit], y[fit], data.iloc[calibration], y[calibration], groups, cfg, cfg.seed + 10000)
        _parquet(out / "stacking_frozen_inner_oof.parquet", proof)
        _parquet(out / "stacking_frozen_inner_manifest.parquet", membership)
        role = np.full(len(data), "fit", dtype=object)
        role[calibration] = "calibration"
        _parquet(out / "stacking_frozen_fit_manifest.parquet", pd.DataFrame({"sample_id": data.sample_id, "subject_id": subjects, "role": role}))
        digest = _checkpoint(out / "frozen_stacking_model.joblib", bundle)
        selection = {"kind": "participant_nested_oof_stacking", "sealed_selection": True, "selected_model": "regularized_logistic_modality_stack", "selection_rule": "fixed_base_and_meta_hyperparameters; no_adaptive_model_search", "target_column": target_column, "feature_columns": features, "fingerprint_columns": fingerprint_features, "modality_columns": groups, "baseline_modality": cfg.baseline_modality, "config": asdict(cfg), "development_fingerprint": fingerprint, "checkpoint_sha256": digest, "operating_points": bundle["operating_points"], "final_fit_details": details, "analysis_type": report["analysis_type"], "protocol_unresolved": report["protocol_unresolved"], "temporal_test_evaluated": False}
        _json(out / "stacking_selection.json", selection)
        report["status"] = "stacking_selection_frozen"
    except Exception as exc:
        report.update(status="stacking_unavailable", reason=exc.code if isinstance(exc, StackingUnavailable) else "fit_or_evaluation_failed", exception_type=type(exc).__name__)
        if isinstance(exc, StackingUnavailable) and exc.details is not None:
            report["failure_details"] = exc.details
    _json(out / "stacking_development_report.json", report)
    return report


def finalize_stacking(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    """Explicit once-only test evaluation; all fits and thresholds remain frozen."""
    out = Path(output_dir)
    if not (out / "stacking_selection.json").is_file():
        raise ValueError("Persisted stacking selection is required before finalization.")
    if (out / "stacking_final_results.json").exists():
        raise FileExistsError("This stacking experiment was already finalized.")
    selection = json.loads((out / "stacking_selection.json").read_text())
    if selection.get("sealed_selection") is not True:
        raise ValueError("Frozen stacking selection is incomplete.")
    features, target = selection["feature_columns"], selection["target_column"]
    _validate_columns(frame, features, target)
    development = frame.loc[frame.partition.eq("development")]
    fingerprint_features = selection["fingerprint_columns"]
    if not set(fingerprint_features).issubset(development) or _fingerprint(development, fingerprint_features, target) != selection["development_fingerprint"]:
        raise ValueError("Stacking development inputs changed after selection.")
    checkpoint = out / "frozen_stacking_model.joblib"
    if not checkpoint.is_file() or sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Frozen stacking checkpoint integrity check failed.")
    bundle = joblib.load(checkpoint)
    if bundle["operating_points"] != selection["operating_points"] or bundle["ensemble"].modality_columns != selection["modality_columns"] or bundle["baseline_modality"] != selection["baseline_modality"] or bundle["cohort_mode"] != selection["config"]["cohort_mode"]:
        raise ValueError("Frozen stacking metadata disagree.")
    cfg = _config(selection["config"])
    # No temporal outcome extraction occurs before the persisted gates above.
    temporal = frame.loc[frame.partition.eq("temporal_test")].reset_index(drop=True)
    if temporal.empty:
        raise ValueError("No temporal-test rows are available.")
    y = _target(temporal, target)
    scored, available = _score(bundle, temporal)
    tables = []
    for name, (p, threshold) in scored.items():
        tables.append(pd.DataFrame({"sample_id": temporal.sample_id, "subject_id": temporal.subject_id, "outer_fold": -1, "model": name, "target": y, "probability": p, "threshold": threshold if threshold is not None else np.nan, "n_available_modalities": available.sum(axis=1)}))
    table = pd.concat(tables, ignore_index=True)
    _parquet(out / "stacking_temporal_predictions.parquet", table)
    report = {"status": "stacking_finalized", **_report_predictions(table, cfg.baseline_modality, cfg), "analysis_type": selection["analysis_type"], "protocol_unresolved": selection["protocol_unresolved"], "temporal_test_evaluated": True, "checkpoint_sha256": selection["checkpoint_sha256"], "confidence_intervals": {}}
    for name, rows in table.groupby("model", sort=False):
        report["confidence_intervals"][name] = {metric: _single_cluster_interval(rows.target, rows.probability, rows.subject_id, metric, cfg.bootstrap_replicates, cfg.seed) for metric in ("auroc", "auprc", "brier")}
    _json(out / "stacking_final_results.json", report)
    return report
