"""Development-only nested evaluation of small gated neural fusion models.

The caller supplies aligned local arrays after metadata review. This module does
not read raw data, extract speech, infer consent, or contact external services.
Only explicit ``finalize_fusion`` may inspect temporal-test outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .benchmarks import (
    _calibrated, _calibrator, _checkpoint, _directory, _json, _parquet,
    _single_cluster_interval, adaptive_group_folds, one_se_selection,
    participant_calibration_split,
)
from .evaluation import (binary_metrics, paired_cluster_bootstrap, participant_weights,
                         select_sensitivity_threshold)
from .neural import (LIMITS, THRESHOLDS, NeuralBatch, NeuralConfig, NeuralTargets,
                     _available_numpy, fit_neural, predict_neural)


@dataclass
class FusionRunConfig:
    neural: NeuralConfig
    variants: tuple[str, ...] = ("phq_single", "multitask")
    outer_folds: int = 5
    inner_folds: int = 3
    calibration_fraction: float = 0.20
    target_sensitivity: float = 0.80
    seed: int = 42
    bootstrap_replicates: int = 2000
    parameter_grid: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    require_reviewed_speech: bool = True
    speech_gate_verified: bool = False
    speech_gate_reference: str | None = None
    protocol_verified: bool = False
    protocol_unresolved: tuple[str, ...] = ("Metadata and protocol require verification.",)

    def validate(self):
        self.neural.validate()
        if (not self.variants or len(set(self.variants)) != len(self.variants) or
                set(self.variants) - {"phq_single", "multitask"}):
            raise ValueError("Fusion variants must be unique phq_single/multitask values")
        if min(self.outer_folds, self.inner_folds) < 2 or not 0 < self.calibration_fraction < 0.5:
            raise ValueError("Invalid grouped fold counts or calibration fraction")
        if not 0 < self.target_sensitivity <= 1 or self.bootstrap_replicates < 1:
            raise ValueError("Invalid sensitivity target or bootstrap count")
        # Input dimensions, outcome definitions and variant task heads cannot be
        # changed during hyperparameter tuning. All comparisons share one cohort.
        allowed = {"hidden_dim", "dropout", "modality_dropout", "learning_rate",
                   "weight_decay", "epochs", "batch_size", "score_loss_weight",
                   "auxiliary_loss_weight"}
        if not self.parameter_grid or any(set(p) - allowed for p in self.parameter_grid):
            raise ValueError("Unsupported fusion tuning parameters")
        for params in self.parameter_grid:
            replace(self.neural, **params).validate()


def _config(config):
    if isinstance(config, FusionRunConfig):
        result = config
    else:
        values = dict(config)
        values["neural"] = NeuralConfig(**values["neural"])
        result = FusionRunConfig(**values)
    result.validate()
    return result


def _metadata(frame, batch):
    required = {"sample_id", "subject_id", "index_date", "partition", "phq9"}
    if not required.issubset(frame) or len(frame) != len(batch):
        raise ValueError("Fusion requires aligned frame metadata, PHQ totals and input arrays")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any() or frame.subject_id.isna().any():
        raise ValueError("Fusion sample identifiers must be unique and subjects present")
    if frame.index_date.isna().any():
        raise ValueError("Fusion index dates must be present")
    membership = frame.groupby("subject_id")["partition"].agg(set)
    if any({"development", "temporal_test"}.issubset(x) for x in membership):
        raise ValueError("Participant overlap across development and temporal test")


def _totals(frame):
    targets = {}
    for name, maximum in LIMITS.items():
        raw = (pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
               if name in frame else np.full(len(frame), np.nan))
        valid = np.isfinite(raw) & (raw >= 0) & (raw <= maximum) & (raw == np.floor(raw))
        targets[name] = np.where(valid, raw, np.nan)
    return NeuralTargets(**targets)


def _binary(targets, name="phq9"):
    total = np.asarray(getattr(targets, name), dtype=float)
    return np.where(np.isfinite(total), (total >= THRESHOLDS[name]).astype(float), np.nan)


def _speech_available(batch):
    present = np.zeros(len(batch), dtype=bool)
    for name in ("audio", "text"):
        raw = getattr(batch, name)
        if raw is None:
            continue
        seen = getattr(batch, name + "_available")
        seen = np.isfinite(raw).all(axis=1) if seen is None else np.asarray(seen, bool)
        seen = seen & np.asarray(batch.recording_available, bool)
        present |= seen
    return present


def _check_batch(batch, config):
    c = config.neural
    batch.validate(c.n_features, c.audio_dim, c.text_dim, c.demo_dim)
    if (batch.audio is not None or batch.text is not None) and batch.recording_available is None:
        raise ValueError("Fusion speech requires explicit reviewed recording availability")
    speech = _speech_available(batch)
    if speech.any() and (not config.speech_gate_verified or not config.speech_gate_reference or
                         not config.speech_gate_reference.strip()):
        return "speech_metadata_gate_unverified"
    if config.require_reviewed_speech and not speech.any():
        return "no_reviewed_speech_inputs"
    return None


def _fingerprint(frame, batch):
    h = sha256()
    meta = frame[["sample_id", "subject_id", "index_date"]].copy()
    totals = _totals(frame)
    for name in LIMITS:
        meta[name] = getattr(totals, name)
    h.update(pd.util.hash_pandas_object(meta, index=False).to_numpy().tobytes())
    for name in sorted(vars(batch)):
        array = getattr(batch, name)
        h.update(name.encode())
        if array is None:
            h.update(b"none")
        else:
            values = np.asarray(array, dtype="<f8").copy()
            values[np.isnan(values)] = np.nan  # canonical missing-value bits
            h.update(str(values.shape).encode())
            h.update(values.tobytes())
    return h.hexdigest()


def _train(batch, targets, groups, config, variant, params, seed):
    # Fixed epochs selected inside inner CV. Neither the calibration participants
    # nor outer validation participants may influence early stopping.
    c = replace(config.neural, **params, multitask=variant == "multitask", seed=seed)
    return fit_neural(batch, targets, c, groups=groups).model


def _tune(batch, targets, groups, config, variant, seed):
    y = _binary(targets).astype(int)
    splits = adaptive_group_folds(y, groups, config.inner_folds, seed)
    if not splits:
        return None, {"status": "insufficient_inner_class_support"}
    records = []
    for parameters in config.parameter_grid:
        scores = []
        try:
            for fit, valid in splits:
                model = _train(batch.subset(fit), targets.subset(fit), groups[fit],
                               config, variant, parameters, seed)
                p = predict_neural(model, batch.subset(valid))["phq9_probability"]
                score = binary_metrics(y[valid], p, groups[valid], threshold=None)["auroc"]
                if score is None or not np.isfinite(score):
                    raise ValueError("Inner fold has no evaluable discrimination estimate")
                scores.append(score)
            records.append({"parameters": parameters, "status": "ok",
                            "mean_auroc": float(np.mean(scores))})
        except Exception as exc:
            records.append({"parameters": parameters, "status": "fit_failed",
                            "exception_type": type(exc).__name__})
    valid = [r for r in records if r["status"] == "ok"]
    if not valid:
        return None, {"status": "all_candidates_failed", "candidates": records}
    selected = max(valid, key=lambda r: r["mean_auroc"])
    return selected["parameters"], {"status": "ok", "folds": len(splits), "candidates": records}


def _calibration(model, batch, targets, groups, config):
    raw = predict_neural(model, batch)
    calibrators, operating, statuses = {}, {}, {}
    for name in LIMITS:
        key = name + "_probability"
        if key not in raw:
            statuses[name] = "head_not_created"
            continue
        y, p = _binary(targets, name), raw[key]
        keep = np.isfinite(y) & np.isfinite(p)
        if len(np.unique(y[keep])) != 2:
            statuses[name] = "insufficient_calibration_class_support"
            continue
        calibrators[name] = _calibrator(y[keep], p[keep], groups[keep])
        cp = _calibrated(calibrators[name], p[keep])
        operating[name] = select_sensitivity_threshold(y[keep], cp, groups[keep],
                                                       config.target_sensitivity)
        statuses[name] = "calibrated"
    if "phq9" not in calibrators or operating["phq9"]["threshold"] is None:
        raise ValueError("PHQ calibration requires predicted examples of both classes")
    return calibrators, operating, statuses


def _predict(model, batch, calibrators):
    result = predict_neural(model, batch)
    for name, calibrator in calibrators.items():
        key = name + "_probability"
        p = result[key]
        keep = np.isfinite(p)
        p[keep] = _calibrated(calibrator, p[keep])
    return result


def _metrics(targets, predictions, groups, operating, statuses):
    result = {}
    for name in LIMITS:
        key = name + "_probability"
        if key not in predictions:
            continue
        result[name] = binary_metrics(_binary(targets, name), predictions[key], groups,
                                      operating.get(name, {}).get("threshold"))
        result[name]["probability_scale"] = ("calibrated" if statuses.get(name) == "calibrated"
                                              else "uncalibrated_or_untrained")
        score_key = name + "_score"
        if score_key in predictions:
            total, score = np.asarray(getattr(targets, name)), predictions[score_key]
            valid = np.isfinite(total) & np.isfinite(score)
            regression = {"n_predicted_visits": int(valid.sum()), "mae": None, "rmse": None}
            if valid.any():
                weights = participant_weights(np.asarray(groups)[valid])
                error = total[valid] - score[valid]
                regression.update(mae=float(np.average(np.abs(error), weights=weights)),
                                  rmse=float(np.sqrt(np.average(error ** 2, weights=weights))))
            result[name]["total_score_metrics"] = regression
    return result


def run_fusion_development(frame: pd.DataFrame, batch: NeuralBatch, output_dir: str | Path,
                            config: FusionRunConfig | Mapping[str, Any]) -> dict:
    """Run actual single/multitask nested CV and freeze a development-only model.

    Input rows and batch arrays must align exactly. Targets are scale totals, not
    input features. Temporal rows are never subset into any fitted computation.
    Only aggregate reports are returned; participant artifacts are saved 0600.
    """
    cfg, out = _config(config), Path(output_dir)
    _metadata(frame, batch)
    if (out / "selection.json").exists():
        raise FileExistsError("Fusion selection already frozen; use a new versioned directory")
    _directory(out)
    positions = np.flatnonzero(frame.partition.eq("development").to_numpy())
    if not len(positions):
        raise ValueError("Fusion development data are absent")
    development = frame.iloc[positions].reset_index(drop=True)
    dev_batch = batch.subset(positions)
    report = {"status": "running", "analysis": "gated_fusion_single_vs_multitask",
              "analysis_type": "prespecified_protocol" if cfg.protocol_verified else "exploratory",
              "protocol_unresolved": list(cfg.protocol_unresolved) if not cfg.protocol_verified else [],
              "config": asdict(cfg), "models": [], "temporal_test_evaluated": False,
              "speech_gate_reference_is_caller_assertion": True,
              "prediction_interpretation": "Questionnaire-defined concurrent risk screening"}
    gate = _check_batch(dev_batch, cfg)
    if gate:
        report["status"] = gate
        _json(out / "development_report.json", report)
        return report
    fingerprint = _fingerprint(development, dev_batch)
    totals = _totals(development)
    y_all = _binary(totals)
    eligible = np.isfinite(y_all) & _available_numpy(dev_batch, np.ones(cfg.neural.n_features, bool))
    selected = np.flatnonzero(eligible)
    data, arrays, labels = development.iloc[selected].reset_index(drop=True), dev_batch.subset(selected), totals.subset(selected)
    y, groups = y_all[selected].astype(int), data.subject_id.to_numpy()
    outer = adaptive_group_folds(y, groups, cfg.outer_folds, cfg.seed)
    report.update(n_development_visits=len(development), n_labelled_visits=int(np.isfinite(y_all).sum()),
                  n_eligible_visits=len(data), n_eligible_participants=int(data.subject_id.nunique()),
                  outer_folds_used=len(outer), reviewed_speech_visits=int(_speech_available(dev_batch).sum()))
    if not outer:
        report["status"] = "insufficient_outer_class_support"
        _json(out / "development_report.json", report)
        return report
    splits, manifests = [], []
    for fold, (train, valid) in enumerate(outer):
        split = participant_calibration_split(y[train], groups[train], cfg.calibration_fraction, cfg.seed + fold)
        if split is None:
            report["status"] = "insufficient_calibration_class_support"
            _json(out / "development_report.json", report)
            return report
        fit, calibration = train[split[0]], train[split[1]]
        splits.append((fit, calibration, valid))
        for role, indices in [("fit", fit), ("calibration", calibration), ("validation", valid)]:
            manifests.append(pd.DataFrame({"sample_id": data.iloc[indices].sample_id.to_numpy(),
                                           "subject_id": groups[indices], "outer_fold": fold, "role": role}))
    _parquet(out / "development_cv_manifest.parquet", pd.concat(manifests, ignore_index=True))
    prediction_tables = []
    full_oof = {}
    for variant in cfg.variants:
        record = {"model": variant, "status": "running", "fold_aurocs": [], "folds": []}
        p_all = np.full(len(development), np.nan)
        for fold, (fit, calibration, valid) in enumerate(splits):
            try:
                params, tuning = _tune(arrays.subset(fit), labels.subset(fit), groups[fit],
                                       cfg, variant, cfg.seed + fold)
                if params is None:
                    record["folds"].append({"fold": fold, "status": "tuning_failed", "tuning": tuning})
                    break
                model = _train(arrays.subset(fit), labels.subset(fit), groups[fit], cfg, variant,
                               params, cfg.seed + fold)
                calibrators, operating, statuses = _calibration(model, arrays.subset(calibration),
                                                                  labels.subset(calibration), groups[calibration], cfg)
                pred = _predict(model, arrays.subset(valid), calibrators)
                metrics = _metrics(labels.subset(valid), pred, groups[valid], operating, statuses)
                auroc = metrics["phq9"]["auroc"]
                if auroc is None:
                    raise ValueError("Outer validation has no evaluable PHQ AUROC")
                record["fold_aurocs"].append(auroc)
                record["folds"].append({"fold": fold, "status": "ok", "parameters": params,
                                         "tuning": tuning, "metrics": metrics,
                                         "operating_points": operating, "calibration_status": statuses})
                pp = pred["phq9_probability"]
                p_all[selected[valid]] = pp
                table = pd.DataFrame({"sample_id": data.iloc[valid].sample_id.to_numpy(),
                                       "subject_id": groups[valid], "model": variant, "outer_fold": fold,
                                       "target": y[valid], "probability": pp,
                                       "threshold": operating["phq9"]["threshold"]})
                for name in LIMITS:
                    for suffix in ("_score", "_probability"):
                        if name + suffix in pred:
                            table[name + suffix] = pred[name + suffix]
                prediction_tables.append(table)
            except Exception as exc:
                record["folds"].append({"fold": fold, "status": "fit_failed",
                                         "exception_type": type(exc).__name__})
                break
        record["status"] = "ok" if len(record["fold_aurocs"]) == len(outer) else "incomplete_cv"
        if record["status"] == "ok":
            # Includes no-input labelled development visits in coverage denominator.
            record["population_oof_metrics"] = binary_metrics(y_all, p_all,
                                                                development.subject_id, threshold=None)
            full_oof[variant] = p_all
        report["models"].append(record)
        _json(out / "development_report.json", report)
    if prediction_tables:
        _parquet(out / "oof_predictions.parquet", pd.concat(prediction_tables, ignore_index=True))
    if {"phq_single", "multitask"}.issubset(full_oof):
        report["paired_multitask_minus_single"] = paired_cluster_bootstrap(
            y_all, full_oof["multitask"], full_oof["phq_single"], development.subject_id,
            n_bootstrap=cfg.bootstrap_replicates, seed=cfg.seed)
    selection = one_se_selection(report["models"], complexity=("phq_single", "multitask"))
    report["selection"] = selection
    if selection["status"] != "ok":
        report["status"] = "no_selectable_model"
        _json(out / "development_report.json", report)
        return report
    split = participant_calibration_split(y, groups, cfg.calibration_fraction, cfg.seed + 10000)
    if split is None:
        report["status"] = "insufficient_final_calibration"
        _json(out / "development_report.json", report)
        return report
    fit, calibration = split
    variant = selection["selected_model"]
    params, tuning = _tune(arrays.subset(fit), labels.subset(fit), groups[fit],
                           cfg, variant, cfg.seed + 10000)
    if params is None:
        report["status"] = "final_development_tuning_failed"
        _json(out / "development_report.json", report)
        return report
    try:
        model = _train(arrays.subset(fit), labels.subset(fit), groups[fit], cfg, variant, params, cfg.seed)
        calibrators, operating, statuses = _calibration(model, arrays.subset(calibration),
                                                         labels.subset(calibration), groups[calibration], cfg)
    except Exception as exc:
        report.update(status="final_development_fit_failed", exception_type=type(exc).__name__)
        _json(out / "development_report.json", report)
        return report
    frozen = {"model": model, "calibrators": calibrators, "operating_points": operating,
              "calibration_status": statuses, "variant": variant}
    digest = _checkpoint(out / "frozen_development_model.joblib", frozen)
    final_manifest = data[["sample_id", "subject_id"]].copy()
    final_manifest["role"] = "fit"
    final_manifest.loc[calibration, "role"] = "calibration"
    _parquet(out / "frozen_fit_manifest.parquet", final_manifest)
    artifact = {**selection, "sealed_selection": True, "temporal_test_evaluated": False,
                "development_fingerprint": fingerprint, "checkpoint_sha256": digest,
                "parameters": params, "tuning": tuning, "operating_points": operating,
                "calibration_status": statuses, "config": asdict(cfg),
                "analysis_type": report["analysis_type"], "protocol_unresolved": report["protocol_unresolved"]}
    _json(out / "selection.json", artifact)
    report.update(status="selection_frozen", selection_artifact="selection.json")
    _json(out / "development_report.json", report)
    return report


def _load_frozen(output_dir):
    out = Path(output_dir)
    if not (out / "selection.json").is_file():
        raise ValueError("Fusion development selection must be persisted first")
    selection = json.loads((out / "selection.json").read_text())
    if not selection.get("sealed_selection"):
        raise ValueError("Fusion selection is not complete")
    checkpoint = out / "frozen_development_model.joblib"
    if not checkpoint.is_file() or sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Fusion checkpoint integrity check failed")
    frozen = joblib.load(checkpoint)  # trusted local private experiment only
    if frozen["variant"] != selection["selected_model"] or frozen["operating_points"] != selection["operating_points"]:
        raise ValueError("Fusion checkpoint and selection disagree")
    return out, selection, frozen


def predict_fusion_frozen(batch: NeuralBatch, output_dir: str | Path) -> dict[str, np.ndarray]:
    """Inference with the frozen model/calibrators; absent inputs remain NaN."""
    _, selection, frozen = _load_frozen(output_dir)
    cfg = _config(selection["config"])
    gate = _check_batch(batch, replace(cfg, require_reviewed_speech=False))
    if gate:
        raise ValueError("Fusion speech metadata gate is not verified")
    return _predict(frozen["model"], batch, frozen["calibrators"])


def finalize_fusion(frame: pd.DataFrame, batch: NeuralBatch, output_dir: str | Path) -> dict:
    """One explicit temporal evaluation. No model selection or fitting occurs."""
    out, selection, frozen = _load_frozen(output_dir)
    if (out / "final_results.json").exists():
        raise FileExistsError("Fusion temporal evaluation already exists")
    _metadata(frame, batch)
    cfg = _config(selection["config"])
    development = np.flatnonzero(frame.partition.eq("development").to_numpy())
    if _fingerprint(frame.iloc[development].reset_index(drop=True), batch.subset(development)) != selection["development_fingerprint"]:
        raise ValueError("Fusion development data changed after selection")
    positions = np.flatnonzero(frame.partition.eq("temporal_test").to_numpy())
    if not len(positions):
        raise ValueError("Fusion temporal data are absent")
    temporal = frame.iloc[positions].reset_index(drop=True)
    temporal_batch = batch.subset(positions)
    gate = _check_batch(temporal_batch, replace(cfg, require_reviewed_speech=False))
    if gate:
        raise ValueError("Fusion temporal speech metadata gate is not verified")
    labels = _totals(temporal)
    pred = _predict(frozen["model"], temporal_batch, frozen["calibrators"])
    metrics = _metrics(labels, pred, temporal.subject_id.to_numpy(),
                       frozen["operating_points"], frozen["calibration_status"])
    y, p = _binary(labels), pred["phq9_probability"]
    intervals = {name: _single_cluster_interval(y, p, temporal.subject_id.to_numpy(), name,
                                                 cfg.bootstrap_replicates, cfg.seed)
                 for name in ("auroc", "auprc", "brier")}
    table = temporal[["sample_id", "subject_id"]].copy()
    table["target"], table["probability"] = y, p
    table["threshold"] = frozen["operating_points"]["phq9"]["threshold"]
    table["status"] = np.where(np.isfinite(p), "predicted", "abstain_no_observed_modality")
    for name in LIMITS:
        for suffix in ("_score", "_probability"):
            if name + suffix in pred:
                table[name + suffix] = pred[name + suffix]
    _parquet(out / "temporal_predictions.parquet", table)
    result = {"status": "finalized", "selected_model": selection["selected_model"],
              "metrics": metrics, "phq9_confidence_intervals": intervals,
              "operating_points": frozen["operating_points"], "temporal_test_evaluated": True,
              "checkpoint_sha256": selection["checkpoint_sha256"],
              "analysis_type": selection["analysis_type"],
              "interpretation": "Questionnaire risk screening; no clinical validity established",
              "coverage_denominator": "all_labelled_temporal_visits_supplied_by_caller"}
    _json(out / "final_results.json", result)
    return result
