"""Software checks use synthetic OOFs only; no cohort or model artifacts read."""
from copy import deepcopy
import json
from pathlib import Path
import stat

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score

from mindwatch.research import monitoring_report as mr
from mindwatch.research.privacy import clean_json, sha256, write_json, write_parquet


@pytest.fixture
def source_report():
    return {"status":"completed_development_only", "config":{"seed":42,"bootstrap_replicates":8},
        "provenance":{"entry_participants":16,"followup_participants":16,"followup_visits":31,
                      "old_test_loaded":False,"original_test_people_excluded":4},
        "folds":[{"mode":"known_user","metadata":{"checks":{"development_only":True,"unique_samples":True,"strict_index_gap_fit_validation":True}}}],
        "failures":[], "controls":{"prespecified_candidate":"D_tcn"},
        "prognosis":{"status":"no_available_pairs"}}


@pytest.fixture
def selected():
    rows = []
    for branch in "ABCD":
        for person in range(16):
            for visit in range(1 if person == 0 else 2):
                y = (person+visit) % 2
                sensor, ema = person%3 != 0, person%4 != 0
                path = "A"
                if branch in "BD" and sensor: path = "B"
                if branch in "CD" and ema: path = "D" if path == "B" else "C"
                p = .8 if y else .2
                if path == "A": p = .5
                if path == "C": p = .65 if y else .35
                if person == 0: p = np.nan
                rows.append({"sample_id":f"synthetic_{person}_{visit}","subject_id":f"synthetic_{person}",
                    "mode":"known_user","endpoint":"y_phq9","task":"classification", "feature_set":branch,"model":f"selected_{branch}",
                    "partition":"development","target":float(y),"probability":p,"threshold":.55 if np.isfinite(p) else np.nan,
                    "outer_fold":visit,"baseline_phq9":11 if person%2 else 6,
                    "sensor_available":sensor,"ema_available":ema,"quality__sensor_observed_days":14 if sensor else 0,
                    "used_modalities":path if np.isfinite(p) else "none","model_version":f"monitoring_v6:classification:{path}:elasticnet",
                    "calibration_status":"platt_calibrated","age_group":"synthetic_age_one" if person%2 else "synthetic_age_two",
                    "sex":"synthetic_sex_one" if person%2 else "synthetic_sex_two", "region_group":"private_synthetic_region_"+str(person%2)})
    return pd.DataFrame(rows)


@pytest.fixture
def preservation():
    return {"all_prior_artifacts_unchanged":True}


def aggregate(report, selected, preservation, **kwargs):
    return mr.aggregate_report_tables(report,selected,preservation_audit=preservation,n_boot=8,seed=42,**kwargs)


def test_primary_weighted_metrics_ci_and_abstention_denominator(source_report, selected, preservation):
    before = selected.copy(deep=True)
    result = aggregate(source_report,selected,preservation)
    pd.testing.assert_frame_equal(selected,before)
    assert result["aggregate"]["status"] == "completed_development_report"
    a = next(r for r in result["aggregate"]["selected_pipelines"] if r["feature_set"] == "A")
    rows = selected.loc[selected.feature_set.eq("A") & selected.probability.notna()]
    w = 1/rows.groupby("subject_id").subject_id.transform("count")
    assert a["metrics"]["auroc"] == roc_auc_score(rows.target,rows.probability,sample_weight=w)
    assert a["metrics"]["auprc"] == average_precision_score(rows.target,rows.probability,sample_weight=w)
    assert a["metrics"]["brier"] == brier_score_loss(rows.target,rows.probability,sample_weight=w)
    assert a["metrics"]["n_labelled_visits"] == 31
    assert a["metrics"]["n_predicted_visits"] == 30
    assert a["metrics"]["coverage"] == pytest.approx(15/16)
    assert a["metrics"]["sensitivity"] == 0
    assert a["metrics"]["specificity"] == 1
    assert a["confidence_intervals"]["metrics"]["auroc"]["n_bootstrap"] == 8
    assert a["confidence_intervals"]["metrics"]["sensitivity"]["n_valid"] == 8
    assert result["audit"]["independent_metric_recalculation"]["absolute_tolerance"] == 1e-12
    assert result["audit"]["test_evaluated"] is False


def test_only_one_primary_contrast_and_separate_cohorts(source_report,selected,preservation):
    other = selected.copy();other["mode"] = "new_user"
    result = aggregate(source_report,pd.concat([selected,other],ignore_index=True),preservation)
    comparisons = result["comparisons"]["comparisons"]
    primary = [r for r in comparisons if r["primary"]]
    assert len(primary) == 1
    assert (primary[0]["mode"],primary[0]["comparison"],primary[0]["cohort"]) == ("known_user","B-A","all_labelled")
    assert primary[0]["p_holm_secondary_family"] is None
    assert primary[0]["prespecified_sensor_added_value_supported"] is True
    ba = {r["cohort"]:r for r in comparisons if r["mode"] == "known_user" and r["comparison"] == "B-A"}
    assert ba["all_labelled"]["n_common_labelled_rows"] > ba["sensor_common"]["n_common_labelled_rows"] > ba["all_four_complete_case"]["n_common_labelled_rows"]
    for row in comparisons:
        if row["comparison"] in {"C-A","D-C"} and row["p_value"] is not None:
            assert row["p_holm_secondary_family"] >= row["p_value"]


def test_same_family_and_fallback_control_routing(source_report,selected,preservation):
    candidates = []
    for family in ("elasticnet","catboost"):
        table = selected.copy()
        table["model"] = table.feature_set+"_"+family
        candidates.append(table)
    original = selected.loc[selected.feature_set.eq("D")].copy()
    original["model"] = "D_tcn"
    # Prespecified TCN had insufficient support: every actual prediction uses A.
    original["used_modalities"] = "A"
    original["model_version"] = "monitoring_v6:classification:A:elasticnet"
    controls = original.copy();controls["model"] = "seed_42"
    result = aggregate(source_report,selected,preservation,candidate=pd.concat([*candidates,original],ignore_index=True),control=controls)
    comparisons = result["comparisons"]
    assert {r["family"] for r in comparisons["comparisons"] if r["source"] == "same_family_candidate"} == {"elasticnet","catboost"}
    ctrl = comparisons["controls"][0]
    assert ctrl["inference"]["delta_a_minus_b"] == 0
    assert ctrl["control_routing"]["actual_D_neural_fraction_of_predictions"] == 0
    assert ctrl["control_routing"]["model_version_counts"] == {"monitoring_v6:classification:A:elasticnet":31}
    assert "no_neural_stability_claim" in ctrl["interpretation"]
    assert result["aggregate"]["control_models"][0]["routing"]["used_modalities_counts"] == {"A":31}
    routed = original.copy();routed["used_modalities"] = "D";routed["model_version"] = "monitoring_v6:classification:D:tcn"
    routing = mr._routing_summary(mr._normalise(routed,source="candidate"),"classification")
    assert routing["actual_D_neural_fraction_of_predictions"] == 1
    current_version = "monitoring_v6:classification:y_phq9:D:tcn:0123456789abcdef:bundle-fedcba9876543210"
    routed["model_version"] = current_version
    routing = mr._routing_summary(routed,"classification")
    assert routing["actual_D_neural_fraction_of_predictions"] == 1
    assert routing["n_actual_D_neural_predicted_rows"] == 30
    assert routing["model_version_counts"] == {current_version:31}
    routed["model_version"] = "monitoring_v6:classification:y_phq9:D:grud:0123456789abcdef"
    assert mr._routing_summary(routed,"classification")["actual_D_neural_fraction_of_predictions"] == 1
    for version in (
        "monitoring_v6_1:classification:y_phq9:D:tcn:0123456789abcdef:bundle-fedcba9876543210",
        "monitoring_v6_1:classification:y_phq9:D:grud:0123456789abcdef",
        "monitoring_v6_1:classification:D:tcn",
    ):
        routed["model_version"] = version
        patched = mr._routing_summary(routed,"classification")
        assert patched["actual_D_neural_fraction_of_predictions"] == 1
        assert patched["model_version_counts"] == {version:31}
    routed["model_version"] = "potential_private_identifier"
    assert "potential_private_identifier" not in json.dumps(mr._routing_summary(routed,"classification"))


def test_sensor_added_value_also_requires_brier_nonworsening(source_report,selected,preservation):
    b = selected.feature_set.eq("B") & selected.probability.notna()
    # Perfect ranking with gross overprediction: AUROC improves, Brier worsens.
    selected.loc[b,"probability"] = np.where(selected.loc[b,"target"].eq(1),.99,.98)
    result = aggregate(source_report,selected,preservation)
    primary = next(row for row in result["comparisons"]["comparisons"] if row["primary"])
    assert primary["inference"]["ci_lower"] > 0
    assert primary["prespecified_sensor_added_value_supported"] is False


def test_regression_and_forecast_support_reporting(source_report,selected,preservation):
    rows = selected.loc[selected.feature_set.eq("A")].copy()
    rows["endpoint"],rows["task"] = "phq9","regression"
    rows["target"] = np.arange(len(rows))%20
    rows["score"] = rows.target+np.where(np.arange(len(rows))%2,1.,-2.)
    rows["probability"] = np.nan
    future = rows.drop(columns=["mode","feature_set"]).copy()
    future["endpoint"],future["model"] = "future_phq9","A_ridge"
    report = deepcopy(source_report)
    report["prognosis"] = {"status":"completed_with_documented_support_gates","records":[{"fold":0,"target":"future_ge10","branch":"B","status":"insufficient_fit_support","fit_pairs":10,"fit_events":2}]}
    result = aggregate(report,pd.concat([selected,rows],ignore_index=True),preservation,prognosis=future)
    reg = next(r for r in result["aggregate"]["selected_pipelines"] if r["task"] == "regression")
    weights = 1/rows.groupby("subject_id").subject_id.transform("count")
    assert reg["metrics"]["mae"] == mean_absolute_error(rows.target,rows.score,sample_weight=weights)
    assert reg["confidence_intervals"]["metrics"]["mae"]["n_valid"] == 8
    prognosis = result["aggregate"]["prognosis"]
    assert prognosis["source_support_status"]["records"][0]["status"] == "insufficient_fit_support"
    assert prognosis["models"][0]["mode"] == "known_user"
    assert prognosis["models"][0]["feature_set"] == "A"


def test_subgroup_categories_only_in_private_key(source_report,selected,preservation):
    result = aggregate(source_report,selected,preservation)
    public = {k:v for k,v in result.items() if k != "private_group_keys"}
    text = json.dumps(clean_json(public),ensure_ascii=False)
    assert "private_synthetic_region_" not in text
    assert "synthetic_age_one" not in text
    assert "synthetic_sex_one" not in text
    assert "private_synthetic_region_0" in set(result["private_group_keys"].category)
    assert any(r.get("grouping") == "baseline_phq9_ge10" for r in result["subgroups"]["groups"])
    assert any(r.get("grouping") == "coverage" for r in result["subgroups"]["groups"])


@pytest.mark.parametrize("defect",["test","duplicate","nonbinary","invalid_probability","outcome_mismatch","participant_mismatch","fold_mismatch"])
def test_invalid_oofs_refused_without_record_values(source_report,selected,preservation,defect):
    broken = selected.copy()
    if defect == "test": broken.loc[0,"partition"] = "temporal_test"
    if defect == "duplicate": broken = pd.concat([broken,broken.iloc[[0]]],ignore_index=True)
    if defect == "nonbinary": broken.loc[0,"target"] = 2
    if defect == "invalid_probability": broken.loc[1,"probability"] = 1.1
    second = broken.index[broken.feature_set.eq("B")][1]
    if defect == "outcome_mismatch": broken.loc[second,"target"] = 1-broken.loc[second,"target"]
    if defect == "participant_mismatch": broken.loc[second,"subject_id"] = "private_synthetic_wrong_person"
    if defect == "fold_mismatch": broken.loc[second,"outer_fold"] = 17
    with pytest.raises(ValueError) as error: aggregate(source_report,broken,preservation)
    assert "synthetic" not in str(error.value)


def test_missing_model_rows_are_descriptive_only(source_report,selected,preservation):
    # A failed model fold can never silently become a valid paired primary test.
    broken = selected.drop(selected.index[(selected.feature_set.eq("B")) & selected.outer_fold.eq(1)])
    report = deepcopy(source_report)
    report["status"] = "completed_with_failures";report["failures"] = [{"exception_type":"SyntheticCheckpointError"}]
    result = aggregate(report,broken,preservation)
    primary = next(r for r in result["comparisons"]["comparisons"] if r["primary"])
    assert primary["status"] == "partial_common_membership"
    assert primary["p_value"] is None
    assert primary["prespecified_sensor_added_value_supported"] is None
    assert result["aggregate"]["status"] == "completed_with_failures_development_report"


@pytest.mark.parametrize("defect",["preservation","role","role_missing","source_failure","source_running","test_evaluated"])
def test_status_never_overclaims_completion(source_report,selected,preservation,defect):
    report, audit = deepcopy(source_report),deepcopy(preservation)
    if defect == "preservation": audit["all_prior_artifacts_unchanged"] = False
    if defect == "role": report["folds"][0]["metadata"]["checks"]["development_only"] = False
    if defect == "role_missing": report["folds"] = []
    if defect == "source_failure": report["failures"] = [{"exception_type":"SyntheticFailure"}]
    if defect == "source_running": report["status"] = "running"
    if defect == "test_evaluated":
        report["test_evaluated"] = True
        with pytest.raises(ValueError): aggregate(report,selected,audit)
        return
    result = aggregate(report,selected,audit)
    assert result["aggregate"]["status"] != "completed_development_report"


def test_single_class_and_missing_saved_thresholds(source_report,selected,preservation):
    selected["target"] = 0.
    selected["threshold"] = np.nan
    result = aggregate(source_report,selected,preservation)
    for row in result["aggregate"]["selected_pipelines"]:
        assert row["metrics"]["status"] == "single_class"
        assert row["metrics"]["auroc"] is None
        assert row["metrics"]["sensitivity"] is None
        assert row["confidence_intervals"]["metrics"]["auroc"]["n_valid"] == 0


def test_writer_versioning_permissions_preservation_and_no_raw_keys(tmp_path,monkeypatch,source_report,selected,preservation):
    source, reports = tmp_path/"source",tmp_path/"reports_v1"
    source.mkdir()
    # Redirect the module-derived projectroot status path into synthetic storage.
    project = tmp_path/"fake_project"
    fake_module = project/"src"/"klosdom"/"monitoring_report.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(mr,"__file__",str(fake_module))
    write_json(source/"development_report.json",source_report)
    write_json(source/"preservation_audit.json",preservation)
    write_parquet(source/"selected_oof.parquet",selected)
    hashes = {p.name:sha256(p) for p in source.iterdir()}
    summary = mr.write_monitoring_report(source,reports)
    assert summary["status"] == "completed_development_report"
    assert summary["test_evaluated"] is False
    expected = {"aggregate_results.json","model_summary.csv","nested_selector_summary.csv","comparisons.json","subgroups.json","audit.json","private_subgroup_key.parquet","MONITORING_STATUS_v6.md"}
    assert {p.name for p in reports.iterdir()} == expected
    assert stat.S_IMODE(reports.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in reports.iterdir())
    assert hashes == {p.name:sha256(p) for p in source.iterdir()}
    for p in reports.iterdir():
        if p.suffix in {".json",".csv",".md"}:
            text = p.read_text()
            assert "private_synthetic_region_" not in text
            assert "synthetic_0" not in text
    status_path = project/"MONITORING_STATUS_v6.md"
    assert not status_path.exists()
    mr.write_monitoring_report(source,tmp_path/"reports_v2")
    assert not status_path.exists()
    with pytest.raises(FileExistsError): mr.write_monitoring_report(source,reports)


@pytest.mark.parametrize("defect",["running","preservation","selected_missing"])
def test_writer_requires_completed_source_and_verified_audit(tmp_path,source_report,selected,preservation,defect):
    source = tmp_path/"source";source.mkdir()
    if defect == "running": source_report["status"] = "running"
    if defect == "preservation": preservation["all_prior_artifacts_unchanged"] = False
    write_json(source/"development_report.json",source_report)
    write_json(source/"preservation_audit.json",preservation)
    if defect != "selected_missing": write_parquet(source/"selected_oof.parquet",selected)
    with pytest.raises(ValueError): mr.write_monitoring_report(source,tmp_path/"reports")
    assert not (tmp_path/"reports").exists()
