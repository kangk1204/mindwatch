"""Aggregate-only monitoring reports from locked, private development OOFs.

No training, model selection, threshold fitting, test-set access or record
printing occurs here. Private subgroup category keys are kept out of JSON/CSV
summaries. Participant bootstrap intervals condition on the saved predictions.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from .analyses import subgroup_reports
from .evaluation import binary_metrics, holm_adjust, paired_cluster_bootstrap
from .privacy import clean_json, secure_dir, sha256, write_json, write_parquet
from .reporting import cluster_auc_interval


CONTEXT = ["mode", "endpoint", "task", "feature_set", "model"]
ACCEPTED_STATUS = {"completed_development_only", "completed_with_failures"}
BRANCHES = {"A": "과거 설문·기본정보", "B": "A + 센서", "C": "A + EMA", "D": "A + 센서 + EMA"}


def _normalise(table, *, source):
    if table is None or table.empty:
        return pd.DataFrame()
    result = table.copy()
    required = {"sample_id", "subject_id", "target", "endpoint", "task", "model"}
    if not required.issubset(result) or result.sample_id.isna().any() or result.subject_id.isna().any():
        raise ValueError("OOF schema or participant/sample identifiers are invalid.")
    if "partition" in result and not result.partition.eq("development").all():
        raise ValueError("Only development OOF rows may enter a monitoring report.")
    if source == "prognosis":
        if "mode" not in result: result["mode"] = "known_user"
        if "feature_set" not in result:
            result["feature_set"] = result.model.astype(str).str.extract(r"^([ABCD])_", expand=False).fillna("A")
    if not set(CONTEXT).issubset(result) or not result["mode"].isin(["known_user", "new_user"]).all() or not result.task.isin(["classification", "regression"]).all() or not result.feature_set.isin(list("ABCD")).all():
        raise ValueError("OOF context is incomplete or invalid.")
    allowed_endpoints = {"classification": {"y_phq9", "y_gad7", "y_isi", "future_ge10", "worsening_ge5"}, "regression": {"phq9", "future_phq9"}}
    if any(endpoint not in allowed_endpoints[task] for endpoint,task in zip(result.endpoint, result.task)):
        raise ValueError("Unrecognized endpoint or outcome task.")
    if result.duplicated([*CONTEXT, "sample_id"]).any():
        raise ValueError("A model must have exactly one OOF prediction per sample and context.")
    result["target"] = pd.to_numeric(result.target, errors="coerce")
    for name in ("probability", "score", "threshold"):
        if name not in result: result[name] = np.nan
        result[name] = pd.to_numeric(result[name], errors="coerce")
    binary = result.task.eq("classification") & np.isfinite(result.target)
    if not result.loc[binary,"target"].isin([0,1]).all() or np.isinf(result[["target","probability","score","threshold"]].to_numpy(float)).any():
        raise ValueError("Nonbinary outcomes or infinite OOF values are invalid.")
    if not result.probability.dropna().between(0,1).all() or not result.threshold.dropna().between(0,1).all():
        raise ValueError("Saved probabilities and thresholds must lie in [0, 1].")
    if "outer_fold" not in result: result["outer_fold"] = -1
    return result


def _explicit_weights(subjects):
    _, inverse, counts = np.unique(np.asarray(subjects), return_inverse=True, return_counts=True)
    return 1.0/counts[inverse] if len(inverse) else np.array([])


def _point(table, task):
    if task == "regression":
        from .monitoring_study import regression_metrics
        metrics = regression_metrics(table.target, table.score, table.subject_id)
        metrics["status"] = "ok" if metrics["n_predicted_visits"] else "no_predictions"
        return metrics
    valid = np.isfinite(table.target) & np.isfinite(table.probability)
    thresholds = table.threshold.to_numpy(dtype=float, copy=True)
    usable_thresholds = np.isfinite(thresholds[valid]).all()
    thresholds[~valid] = .5  # Only ignored abstention/unlabelled positions.
    metrics = binary_metrics(table.target, table.probability, table.subject_id, thresholds if usable_thresholds else None)
    metrics["operating_point_status"] = "saved_calibration_or_fallback_thresholds" if usable_thresholds else "predicted_rows_have_missing_thresholds"
    return metrics


def _recalculate(table, task, metrics):
    name = "score" if task == "regression" else "probability"
    keep = np.isfinite(table.target) & np.isfinite(table[name])
    y, p = table.loc[keep, "target"].to_numpy(), table.loc[keep, name].to_numpy()
    w = _explicit_weights(table.loc[keep, "subject_id"])
    estimates = {}
    if len(y):
        if task == "regression":
            estimates = {"mae": float(mean_absolute_error(y, p, sample_weight=w)), "rmse": float(np.sqrt(mean_squared_error(y, p, sample_weight=w)))}
        else:
            estimates["brier"] = float(brier_score_loss(y, p, sample_weight=w))
            if len(np.unique(y)) == 2:
                estimates.update(auroc=float(roc_auc_score(y, p, sample_weight=w)), auprc=float(average_precision_score(y, p, sample_weight=w)))
    for key, value in estimates.items():
        if metrics.get(key) is None or not np.isclose(value, metrics[key], atol=1e-12, rtol=0):
            raise ValueError("Independent sklearn OOF recalculation failed.")
    return {"checks": len(estimates), "all_passed": True, "absolute_tolerance": 1e-12}


def _intervals(table, task, n_boot, seed):
    column = "score" if task == "regression" else "probability"
    valid = table.loc[np.isfinite(table.target) & np.isfinite(table[column])].reset_index(drop=True)
    metrics = ("mae", "rmse") if task == "regression" else ("auroc", "auprc", "brier", "sensitivity", "specificity", "ppv", "npv")
    empty = {key: {"lower": None, "upper": None, "n_valid": 0, "n_bootstrap": n_boot} for key in metrics}
    if valid.subject_id.nunique() < 2:
        return {"status": "insufficient_participants", "metrics": empty}
    y, p = valid.target.to_numpy(), valid[column].to_numpy()
    codes, unique = pd.factorize(valid.subject_id, sort=False)
    weights, rng = _explicit_weights(valid.subject_id), np.random.default_rng(seed)
    draws = {key: [] for key in metrics if key != "auroc"}
    if task == "classification" and len(np.unique(y)) == 2:
        ci, count = cluster_auc_interval(valid[["subject_id", "target", "probability"]], n_boot=n_boot, seed=seed)
        empty["auroc"] = {"lower": ci[0], "upper": ci[1], "n_valid": count, "n_bootstrap": n_boot}
    for _ in range(n_boot):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        w = weights*multiplicity[codes]
        keep = w > 0
        if task == "regression":
            draws["mae"].append(float(np.average(abs(y[keep]-p[keep]), weights=w[keep])))
            draws["rmse"].append(float(np.sqrt(np.average((y[keep]-p[keep])**2, weights=w[keep]))))
        else:
            draws["brier"].append(float(np.average((y[keep]-p[keep])**2, weights=w[keep])))
            if len(np.unique(y[keep])) == 2:
                draws["auprc"].append(float(average_precision_score(y[keep], p[keep], sample_weight=w[keep])))
            thresholds = valid.threshold.to_numpy(float)
            if np.isfinite(thresholds).all():
                positive = p >= thresholds
                tp, fn = w[(y == 1)&positive].sum(), w[(y == 1)&~positive].sum()
                fp, tn = w[(y == 0)&positive].sum(), w[(y == 0)&~positive].sum()
                for key,numerator,denominator in (("sensitivity",tp,tp+fn),("specificity",tn,tn+fp),("ppv",tp,tp+fp),("npv",tn,tn+fn)):
                    if denominator: draws[key].append(float(numerator/denominator))
    for key, values in draws.items():
        if values:
            lo, hi = np.quantile(values, [.025, .975])
            empty[key] = {"lower": float(lo), "upper": float(hi), "n_valid": len(values), "n_bootstrap": n_boot}
    return {"status": "ok", "method": "equal_participant_cluster_percentile_fixed_predictions", "metrics": empty,
            "scope": "predicted labelled participants; saved thresholds held fixed; calibration curves and slopes are point summaries"}


def _public_distribution(table, column, allowed):
    if column not in table: return {"metadata_unavailable": len(table)}
    values = table[column].astype(str).where(table[column].astype(str).isin(allowed), "other_or_unknown")
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}


def _routing_summary(table, task):
    """Actual saved inference paths, including support-triggered fallbacks.

    Unknown version strings are counted opaquely so arbitrary source metadata
    cannot introduce participant identifiers into the aggregate report.
    """
    column = "score" if task == "regression" else "probability"
    predicted = np.isfinite(table.target) & np.isfinite(table[column])
    versions = table.get("model_version", pd.Series("metadata_unavailable", index=table.index)).fillna("metadata_unavailable").astype(str)
    legacy_pattern = re.compile(r"^monitoring_v\d+(?:_\d+)*:(classification|regression):([ABCD]):(elasticnet|ridge|catboost|grud|tcn)$")
    current_pattern = re.compile(r"^monitoring_v\d+(?:_\d+)*:(classification|regression):(?:y_phq9|phq9|y_gad7|y_isi|future_phq9|future_ge10|worsening_ge5):([ABCD]):(elasticnet|ridge|catboost|grud|tcn):[0-9a-f]{16}(?::bundle-[0-9a-f]{16})?$")
    parsed = [current_pattern.fullmatch(value) or legacy_pattern.fullmatch(value) for value in versions]
    recognised = pd.Series([match is not None for match in parsed],index=table.index)
    public_versions = versions.where(recognised | versions.isin(["persistence","persistence_v6"]), "other_or_unknown")
    pathways = table.get("used_modalities", pd.Series("none", index=table.index)).astype(str)
    families = pd.Series([match.group(3) if match else None for match in parsed],index=table.index)
    branches = pd.Series([match.group(2) if match else None for match in parsed],index=table.index)
    neural = predicted & recognised & families.isin(["grud","tcn"])
    d_neural = neural & pathways.eq("D") & branches.eq("D")
    n_predicted, n_d = int(predicted.sum()), int(d_neural.sum())
    return {"used_modalities_counts": _public_distribution(table,"used_modalities",{"A","B","C","D","none"}),
        "model_version_counts": {str(k):int(v) for k,v in public_versions.value_counts().items()},
        "n_predicted_rows": n_predicted, "n_actual_neural_predicted_rows": int(neural.sum()),
        "n_actual_D_neural_predicted_rows": n_d,
        "actual_D_neural_fraction_of_predictions": n_d/n_predicted if n_predicted else None,
        "actual_D_neural_fraction_of_labelled_rows": n_d/int(np.isfinite(table.target).sum()) if np.isfinite(table.target).any() else None,
        "neural_routing_interpretation": "no_actual_D_neural_routing" if n_d == 0 else "mixed_routed_pipeline" if n_d < n_predicted else "all_predictions_use_D_neural_route"}


def _summaries(table, source, n_boot, seed, intervals=False):
    output, checks = [], 0
    if table.empty: return output, checks
    for context, rows in table.groupby(CONTEXT, sort=True, dropna=False):
        row = dict(zip(CONTEXT, context))
        metrics = _point(rows, row["task"])
        checks += _recalculate(rows, row["task"], metrics)["checks"]
        output.append({"source": source, **row, "metrics": metrics,
            "confidence_intervals": _intervals(rows, row["task"], n_boot, seed) if intervals else {"status": "not_requested_for_candidate_point_summary"},
            "folds": int(rows.outer_fold.nunique()),
            "used_input_pathways": _public_distribution(rows, "used_modalities", {"A", "B", "C", "D", "none"}),
            "routing": _routing_summary(rows, row["task"]),
            "calibration_status_counts": _public_distribution(rows, "calibration_status", {"platt_calibrated", "insufficient_calibration", "calibration_failed_identity", "not_applicable_regression", "unavailable_no_model"}),
            "selection_scope": "inner_selected_or_prespecified_warmup" if source == "selected" else "fixed_candidate_descriptive"})
    return output, checks


def _pair(left, right, task, n_boot, seed, cohort="all_labelled", complete_ids=None):
    if left.empty or right.empty:
        return {"status": "model_oof_unavailable", "cohort": cohort, "p_value": None}
    keep_columns = ["sample_id", "subject_id", "target", "probability", "score", "threshold", "outer_fold"]
    if "sensor_available" in left and "sensor_available" in right: keep_columns.append("sensor_available")
    joined = left[keep_columns].merge(right[keep_columns], on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(joined.subject_id_a, joined.subject_id_b) or not np.array_equal(joined.outer_fold_a, joined.outer_fold_b) or not np.array_equal(joined.target_a.to_numpy(float), joined.target_b.to_numpy(float), equal_nan=True):
        raise ValueError("Paired OOF identity, outcome or fold membership differs.")
    full_membership = set(left.sample_id) == set(right.sample_id)
    if cohort == "sensor_common":
        if "sensor_available_a" not in joined:
            return {"status": "sensor_availability_metadata_unavailable", "cohort": cohort, "p_value": None}
        joined = joined.loc[joined.sensor_available_a.eq(True) & joined.sensor_available_b.eq(True)]
    elif cohort == "all_four_complete_case":
        if complete_ids is None:
            return {"status": "all_four_support_unavailable", "cohort": cohort, "p_value": None}
        joined = joined.loc[joined.sample_id.isin(complete_ids)]
    column = "score" if task == "regression" else "probability"
    valid = np.isfinite(joined.target_a) & np.isfinite(joined[column+"_a"]) & np.isfinite(joined[column+"_b"])
    paired = joined.loc[valid]
    result = {"status": "ok" if full_membership else "partial_common_membership", "cohort": cohort,
              "n_left_rows": len(left), "n_right_rows": len(right), "n_common_labelled_rows": int(np.isfinite(joined.target_a).sum()),
              "n_paired_predicted_rows": len(paired), "n_paired_participants": int(paired.subject_id_a.nunique()),
              "pairing": "identical_participants_samples_and_outer_folds", "p_value": None, "common_metrics": {}}
    for suffix, name in (("a", "augmented_or_original"), ("b", "baseline_or_control")):
        view = pd.DataFrame({"target": paired.target_a, "subject_id": paired.subject_id_a, "probability": paired["probability_"+suffix], "score": paired["score_"+suffix], "threshold": paired["threshold_"+suffix]})
        result["common_metrics"][name] = _point(view, task)
    if not full_membership:
        result["inference"] = "descriptive_common_subset_only_missing_model_rows_preclude_primary_inference"
        return result
    if task == "classification":
        inference = paired_cluster_bootstrap(paired.target_a, paired.probability_a, paired.probability_b, paired.subject_id_a, n_bootstrap=n_boot, seed=seed)
        result["inference"] = inference
        result["p_value"] = inference["p_value"]
    else:
        y, a, b = paired.target_a.to_numpy(), paired.score_a.to_numpy(), paired.score_b.to_numpy()
        codes, people = pd.factorize(paired.subject_id_a, sort=False)
        if len(people) < 2:
            result["inference"] = {"status": "insufficient_participants"}
            return result
        weights, rng = _explicit_weights(paired.subject_id_a), np.random.default_rng(seed)
        delta = float(np.average(abs(y-a)-abs(y-b), weights=weights))
        values = []
        for _ in range(n_boot):
            counts = np.bincount(rng.integers(0, len(people), len(people)), minlength=len(people))
            values.append(float(np.average(abs(y-a)-abs(y-b), weights=weights*counts[codes])))
        lo, hi = np.quantile(values, [.025, .975])
        pvalue = float((1+np.count_nonzero(abs(np.asarray(values)-delta) >= abs(delta)-1e-12))/(n_boot+1))
        result.update(p_value=pvalue, inference={"status": "ok", "metric": "mae", "delta_a_minus_b": delta, "ci_lower": float(lo), "ci_upper": float(hi), "p_value": pvalue, "n_bootstrap": n_boot, "n_valid": n_boot, "method": "approximate_centred_participant_cluster_bootstrap"})
    return result


def _complete_ids(groups):
    if not all(branch in groups for branch in "ABCD"): return None
    common = None
    for branch in "ABCD":
        rows = groups[branch]
        column = "score" if rows.task.iloc[0] == "regression" else "probability"
        if not {"sensor_available", "ema_available"}.issubset(rows): return None
        available = np.isfinite(rows.target) & np.isfinite(rows[column]) & rows.sensor_available.eq(True) & rows.ema_available.eq(True)
        ids = set(rows.loc[available, "sample_id"])
        common = ids if common is None else common & ids
    return common


def _comparisons(selected, candidate, control, report, n_boot, seed):
    records = []
    for source, table in (("selected", selected), ("same_family_candidate", candidate)):
        if table.empty: continue
        for context, block in table.groupby(["mode", "endpoint", "task"], sort=True):
            mode, endpoint, task = context
            if source == "same_family_candidate" and (endpoint != "y_phq9" or task != "classification"):
                continue
            families = ["selected"] if source == "selected" else ["elasticnet", "catboost"]
            for family in families:
                groups = {}
                for branch in "ABCD":
                    name = f"selected_{branch}" if family == "selected" else f"{branch}_{family}"
                    rows = block.loc[block.model.eq(name)]
                    if not rows.empty: groups[branch] = rows
                complete = _complete_ids(groups)
                for cohort in ("all_labelled", "sensor_common", "all_four_complete_case"):
                    current = []
                    for augmented, baseline in (("B", "A"), ("C", "A"), ("D", "C")):
                        pair = _pair(groups.get(augmented, pd.DataFrame()), groups.get(baseline, pd.DataFrame()), task, n_boot, seed, cohort, complete)
                        current.append({"source": source, "family": family, "mode": mode, "endpoint": endpoint, "task": task, "comparison": augmented+"-"+baseline,
                            "primary": source == "selected" and mode == "known_user" and endpoint == "y_phq9" and task == "classification" and augmented == "B" and cohort == "all_labelled", **pair})
                    secondary = holm_adjust({row["comparison"]: row["p_value"] for row in current if row["comparison"] in {"C-A", "D-C"}})
                    for row in current:
                        row["p_holm_secondary_family"] = secondary.get(row["comparison"])
                        row["multiplicity"] = "primary_contrast_not_in_secondary_Holm_family" if row["comparison"] == "B-A" else "Holm_C-A_and_D-C_within_mode_endpoint_task_family_cohort"
                        inference = row.get("inference", {})
                        common = row.get("common_metrics", {})
                        augmented_brier = common.get("augmented_or_original", {}).get("brier")
                        baseline_brier = common.get("baseline_or_control", {}).get("brier")
                        eligible = task == "classification" and row["comparison"] == "B-A" and isinstance(inference,dict) and inference.get("ci_lower") is not None and augmented_brier is not None and baseline_brier is not None
                        row["prespecified_sensor_added_value_supported"] = bool(inference["ci_lower"] > 0 and augmented_brier <= baseline_brier) if eligible else None
                    records.extend(current)
    controls = []
    if not control.empty:
        baseline_name = report.get("controls", {}).get("prespecified_candidate", "D_tcn")
        for context, rows in control.groupby(CONTEXT, sort=True):
            values = dict(zip(CONTEXT, context))
            original = candidate.loc[candidate["mode"].eq(values["mode"]) & candidate.endpoint.eq(values["endpoint"]) & candidate.task.eq(values["task"]) & candidate.model.eq(baseline_name)] if not candidate.empty else pd.DataFrame()
            original_routing = _routing_summary(original, values["task"]) if not original.empty else None
            control_routing = _routing_summary(rows, values["task"])
            neural_pairs = bool(original_routing and original_routing["n_actual_D_neural_predicted_rows"] and control_routing["n_actual_D_neural_predicted_rows"])
            controls.append({**values, "original_candidate": baseline_name, "exploratory_conditional_on_prespecified_candidate": True,
                "original_routing": original_routing, "control_routing": control_routing,
                "interpretation": "conditional_routed_pipeline_comparison_not_pure_neural_stability" if neural_pairs else "no_D_neural_routing_in_at_least_one_pipeline_no_neural_stability_claim",
                "delta_direction": "original_minus_control", **_pair(original, rows, values["task"], n_boot, seed)})
    return {"comparisons": records, "controls": controls,
            "complete_case_definition": "all four pipeline predictions finite and sensor+EMA available; not every raw feature observed",
            "bootstrap_interpretation": "conditional on saved development OOF predictions; no independent test or full refit uncertainty"}


def _subgroups(selected):
    output, keys = [], []
    if selected.empty: return {"scope": "selected PHQ binary monitoring only", "groups": []}, pd.DataFrame()
    primary = selected.loc[(selected.endpoint.eq("y_phq9") & selected.task.eq("classification"))]
    for context, rows in primary.groupby(CONTEXT, sort=True):
        values = dict(zip(CONTEXT, context))
        rows = rows.copy()
        rows["partition"] = "development"
        coverage_column = None
        if "quality__sensor_observed_days" in rows:
            rows["_sensor_fraction"] = pd.to_numeric(rows.quality__sensor_observed_days, errors="coerce")/14
            coverage_column = "_sensor_fraction"
        summary = subgroup_reports(rows, target_column="target", probability_column="probability", coverage_column=coverage_column)
        for record in summary["groups"]:
            output.append({**values, **record})
        # Reproduce the helper's opaque key mapping privately; never embed it in
        # aggregate JSON or return it from the filesystem report writer.
        categories = {}
        for grouping in ("age_group", "sex", "region_group"):
            if grouping in rows:
                categories[grouping] = rows[grouping].fillna("missing").astype(str).replace("", "missing")
        if coverage_column:
            categories["_coverage_group"] = pd.cut(rows[coverage_column], [-np.inf,.25,.5,.75,np.inf], labels=["[0,.25)","[.25,.5)","[.5,.75)","[.75,1]"], right=False).astype("string").fillna("missing")
        for grouping, series in categories.items():
            for i, value in enumerate(sorted(series.unique())):
                keys.append({**values, "grouping": grouping, "group": f"group_{i+1:03d}", "category": str(value)})
        if "baseline_phq9" in rows:
            selected_high = rows.loc[pd.to_numeric(rows.baseline_phq9, errors="coerce").ge(10)]
            output.append({**values, "grouping": "baseline_phq9_ge10", "group": "prespecified_ge10", **_point(selected_high, "classification"), "analysis_status": "descriptive_no_subgroup_hypothesis_test"})
    return {"scope": "selected primary PHQ binary monitoring; age/sex/opaque geography/observed-day coverage and baseline>=10",
            "groups": output, "category_key": "private_subgroup_key.parquet", "inference": "descriptive; small subgroup uncertainty and confounding remain"}, pd.DataFrame(keys)


def _selector_rows(report, selected):
    rows = report.get("selectors", [])
    if rows:
        columns = ["mode", "endpoint", "task", "feature_set", "fold", "chosen_name", "selection_status", "status"]
        return [{**{key: record.get(key) for key in columns},
                 **{key:value for key,value in record.get("metrics",{}).items() if not isinstance(value,(dict,list))}} for record in rows]
    result = []
    if not selected.empty:
        for context, block in selected.groupby([*CONTEXT, "outer_fold"], sort=True):
            values = dict(zip([*CONTEXT, "outer_fold"], context))
            result.append({**values, "selection_status": "metadata_unavailable" if "selection_status" not in block else str(block.selection_status.iloc[0]), "status": "OOF_present_selection_record_unavailable"})
    return result


def aggregate_report_tables(report: dict, selected: pd.DataFrame, candidate: pd.DataFrame | None = None,
                            control: pd.DataFrame | None = None, prognosis: pd.DataFrame | None = None, *,
                            preservation_audit: dict | None = None, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Pure aggregation for synthetic tests and privately loaded OOF tables.

    The private_group_keys return value is deliberately separate from every
    public aggregate object. Only the writer persists it, as a restricted file.
    """
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("A positive participant bootstrap count is required.")
    selected, candidate, control, prognosis = [_normalise(table, source=name) for table, name in
        ((selected,"selected"),(candidate,"candidate"),(control,"control"),(prognosis,"prognosis"))]
    if report.get("test_evaluated") is True or report.get("provenance", {}).get("old_test_loaded") is True:
        raise ValueError("Monitoring development reports cannot include evaluated test outcomes.")
    selected_summary, checks = _summaries(selected, "selected", n_boot, seed, intervals=True)
    candidate_summary, number = _summaries(candidate, "candidate", n_boot, seed)
    checks += number
    control_summary, number = _summaries(control, "control", n_boot, seed)
    checks += number
    prognosis_summary, number = _summaries(prognosis, "prognosis", n_boot, seed, intervals=True)
    checks += number
    comparisons = _comparisons(selected, candidate, control, report, n_boot, seed)
    subgroups, private_keys = _subgroups(selected)
    role_checks = [item.get("metadata", {}).get("checks", {}) for item in report.get("folds", [])]
    role_false = sum(value is False for item in role_checks for value in item.values())
    preservation = bool(preservation_audit and preservation_audit.get("all_prior_artifacts_unchanged") is True)
    failures = report.get("failures", [])
    status = "completed_development_report" if report.get("status") == "completed_development_only" and not failures else "completed_with_failures_development_report"
    if report.get("status") not in ACCEPTED_STATUS: status = "source_run_not_completed"
    if not selected_summary: status = "selected_oof_unavailable"
    role_verified = bool(role_checks) and all(item and all(value is True for value in item.values()) for item in role_checks)
    if not preservation or not role_verified: status = "audit_not_verified"
    provenance = report.get("provenance", {})
    whitelist = {"status","entry_rule","entry_threshold","initial_visits_in_current_state","all_later_visits_retained","entry_participants","followup_participants","followup_visits","followup_phq_positive","sensor_available_visits","ema_available_visits","history_available_visits","prognosis_pairs","development_end_exclusive","old_test_loaded","original_test_people_excluded","enrollment_basis","label_release_proxy","as_of_upload_time","sensor_eligibility","sources_sha256","data_manifest_sha256"}
    aggregate = {"status": status, "source_status": report.get("status"), "analysis_type": "exploratory_development_monitoring",
        "population": {key: value for key, value in provenance.items() if key in whitelist}, "feature_sets": BRANCHES,
        "selected_pipelines": selected_summary, "candidate_models": candidate_summary, "control_models": control_summary,
        "prognosis": {"source_support_status": report.get("prognosis", {}), "models": prognosis_summary},
        "interpretation": {"target": "questionnaire-defined follow-up risk/score, not diagnosis or treatment benefit", "entry": "one-time first-observed PHQ entry; first visits excluded, later low-PHQ visits retained",
            "known_user": "past same-person observations may be in fit/calibration; genuine monitoring estimand",
            "new_user": "validation people absent from fit/calibration; participant transport comparator",
            "independent_test": False, "coverage": "all labelled evaluation rows including abstentions; paired and complete-case results use explicitly narrower populations",
            "thresholds": "saved calibration/fallback thresholds; evaluation never retunes them", "bootstrap": "participant clusters of fixed OOF predictions; excludes full refit variability",
            "failed_folds": "coverage denominators contain saved OOF rows; a missing failed fold is not reconstructed and blocks a complete-run claim"}}
    audit = {"status": "ok" if preservation and role_verified and selected_summary else "not_verified", "independent_metric_recalculation": {"passed": True, "checks": checks, "absolute_tolerance": 1e-12, "method": "sklearn_metrics_explicit_equal_participant_weights"},
        "preservation_verified": preservation, "source_role_checks_available": bool(role_checks), "source_role_checks_false": role_false,
        "source_role_checks_incomplete": any(value is None for item in role_checks for value in item.values()),
        "source_failure_count": len(failures), "source_failure_types": {str(key): int(value) for key,value in pd.Series([f.get("exception_type","documented_support_or_selection_failure") for f in failures], dtype="str").value_counts().items()},
        "bootstrap_replicates": n_boot, "seed": seed, "test_evaluated": False, "individual_records_in_public_summaries": False}
    return {"aggregate": aggregate, "comparisons": comparisons, "subgroups": subgroups, "audit": audit,
            "model_rows": selected_summary+candidate_summary+control_summary+prognosis_summary, "selector_rows": _selector_rows(report, selected), "private_group_keys": private_keys}


def _flat_rows(rows):
    result = []
    for row in rows:
        value = {key: row[key] for key in ["source", *CONTEXT, "folds"] if key in row}
        value.update({key: item for key,item in row["metrics"].items() if not isinstance(item,(list,dict))})
        for name, interval in row.get("confidence_intervals", {}).get("metrics", {}).items():
            value[name+"_ci_lower"], value[name+"_ci_upper"] = interval["lower"], interval["upper"]
            value[name+"_bootstrap_valid"] = interval["n_valid"]
        value["actual_D_neural_fraction_of_predictions"] = row.get("routing",{}).get("actual_D_neural_fraction_of_predictions")
        result.append(value)
    return result


def _csv(path, rows):
    from mindwatch.pipeline import private_output
    path = private_output(path)
    fd = os.open(path, os.O_CREAT|os.O_WRONLY|os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)


def _markdown(aggregate, audit):
    lines = ["# 정신건강 모니터링 v6 결과", "", f"상태: `{aggregate['status']}`. 독립 시험군을 평가하지 않은 탐색적 개발 검증입니다.", "",
             "진입 시점 PHQ 조건을 한 번 적용한 뒤 첫 관측을 제외한 후속 시점을 평가했습니다. known_user는 같은 사람의 과거 관측을 허용하며, new_user는 평가 사람을 학습·보정에서 제외합니다.", "",
             "| 모드 | 입력 | 참여자 | 평가행 | AUROC (95% CI) | AP | Brier | 예측 가능 비율 |", "|---|---|---:|---:|---|---:|---:|---:|"]
    fmt = lambda x: "추정 불가" if x is None else f"{x:.3f}"
    for row in aggregate["selected_pipelines"]:
        if row["endpoint"] != "y_phq9" or row["task"] != "classification": continue
        m = row["metrics"]
        ci = row["confidence_intervals"].get("metrics",{}).get("auroc",{})
        auc = f"{fmt(m['auroc'])} ({fmt(ci.get('lower'))}–{fmt(ci.get('upper'))})"
        lines.append(f"| {row['mode']} | {row['feature_set']} | {m['n_labelled_participants']} | {m['n_labelled_visits']} | {auc} | {fmt(m['auprc'])} | {fmt(m['brier'])} | {fmt(m['coverage'])} |")
    lines += ["", "A는 과거 설문·기본정보, B는 A+센서, C는 A+EMA, D는 A+센서+EMA입니다. 입력 부족 시 사전에 정한 하위 입력 경로를 사용하므로 경로별 사용량을 함께 확인해야 합니다.", "",
              "B−A가 주 비교이며 C−A와 D−C는 별도 Holm 보조 가족입니다. 센서 공통 가용 집합과 네 경로 공통 가용 집합은 전체 평가 코호트와 구분했습니다. 임계값은 저장된 보정/대체 경로 값을 그대로 사용했습니다.", "",
              f"독립 sklearn 재계산: {audit['independent_metric_recalculation']['checks']}개, 절대 오차 허용 1e-12. 원래 산출물 보존 확인: {audit['preservation_verified']}. 기록된 실행 실패: {audit['source_failure_count']}개.", "",
              "참여자 bootstrap은 저장된 OOF 예측의 불확실성을 평가합니다. 임상적 유효성, 진단 능력 또는 새로운 독립 시험 성능을 입증하지 않습니다.", ""]
    return "\n".join(lines)


def write_monitoring_report(output: Path, report_dir: Path) -> dict[str, Any]:
    """Read private derived OOFs and save a new restricted aggregate version."""
    output, report_dir = Path(output).resolve(), Path(report_dir).resolve()
    if report_dir == output or report_dir in output.parents or (report_dir.exists() and any(report_dir.iterdir())):
        raise FileExistsError("Use a fresh separate versioned report directory.")
    source_report = output/"development_report.json"
    report = json.loads(source_report.read_text())
    if report.get("status") not in ACCEPTED_STATUS:
        raise ValueError("Only completed or explicitly completed-with-failures runs may be reported.")
    preservation_path = output/"preservation_audit.json"
    preservation = json.loads(preservation_path.read_text())
    if preservation.get("all_prior_artifacts_unchanged") is not True:
        raise ValueError("A verified preservation audit is required.")
    names = {"selected": "selected_oof.parquet", "candidate": "candidate_oof.parquet", "control": "control_oof.parquet", "prognosis": "prognosis_oof.parquet"}
    paths = [source_report, preservation_path, *[output/name for name in names.values() if (output/name).is_file()]]
    before = {path.name: sha256(path) for path in paths}
    if not (output/names["selected"]).is_file():
        raise ValueError("Selected monitoring OOF predictions are required.")
    tables = {key: pd.read_parquet(output/name) if (output/name).is_file() else None for key,name in names.items()}
    config = report.get("config", {})
    result = aggregate_report_tables(report, **tables, preservation_audit=preservation,
        n_boot=int(config.get("bootstrap_replicates", 2000)), seed=int(config.get("seed",42)))
    if before != {path.name: sha256(path) for path in paths}:
        raise ValueError("Source reporting artifacts changed during aggregation.")
    secure_dir(report_dir)
    write_json(report_dir/"aggregate_results.json", result["aggregate"])
    write_json(report_dir/"comparisons.json", result["comparisons"])
    write_json(report_dir/"subgroups.json", result["subgroups"])
    _csv(report_dir/"model_summary.csv", _flat_rows(result["model_rows"]))
    _csv(report_dir/"nested_selector_summary.csv", result["selector_rows"])
    if not result["private_group_keys"].empty:
        write_parquet(report_dir/"private_subgroup_key.parquet", result["private_group_keys"])
    status_text = _markdown(result["aggregate"], result["audit"])
    for path in (report_dir/"MONITORING_STATUS_v6.md",):
        try:
            fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd,"w",encoding="utf-8") as handle: handle.write(status_text)
    audit = {**result["audit"], "source_artifact_sha256": before, "all_source_reporting_artifacts_unchanged": True}
    write_json(report_dir/"audit.json", audit)
    return {"status": result["aggregate"]["status"], "selected_pipeline_groups": len(result["aggregate"]["selected_pipelines"]),
            "independent_metric_checks": audit["independent_metric_recalculation"]["checks"], "report_directory": str(report_dir), "test_evaluated": False}
