"""Synthetic verification of secondary analyses; no patient records are loaded."""
import importlib.util
import json
import stat

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.analyses import (
    longitudinal_associations, paired_modality_comparisons, regression_metrics,
    retention_report, run_geography_validation, run_prognosis_regression,
    subgroup_reports,
)


def cohort(n_subjects=60, repeats=3):
    rng = np.random.default_rng(17)
    records = []
    for subject in range(n_subjects):
        personal = rng.normal()
        enrolled = pd.Timestamp("2023-01-01")+pd.Timedelta(days=subject)
        for visit in range(repeats):
            date = enrolled+pd.Timedelta(days=124*visit)
            signal = personal+rng.normal()
            score = int(np.clip(np.round(8+4*personal+2*signal+rng.normal()), 0, 27))
            records.append({"sample_id": f"sample-{subject}-{visit}", "subject_id": f"subject-{subject}",
                            "index_date": date, "enrolled_date": enrolled, "wave": visit+1,
                            "partition": "development", "phq9": score,
                            "target": float(score >= 10), "probability": float(np.clip(.2+.6*(score >= 10), .01, .99)),
                            "region_group": f"secret-region-{subject%3}", "sex": f"secret-sex-{subject%2}",
                            "age_group": f"secret-age-{subject%4}", "coverage": float(rng.uniform()),
                            "sensor__mean": signal, "sensor__second": rng.normal(),
                            "sensor_window_end": date-pd.Timedelta(days=1),
                            "target_date": date+pd.Timedelta(days=124), "future_phq9": int(np.clip(score+rng.integers(-3, 4), 0, 27))})
    return pd.DataFrame(records)


def oof(frame, probability=None):
    result = frame[["sample_id", "subject_id", "target"]].copy()
    result["probability"] = frame.probability if probability is None else probability
    result["outer_fold"] = np.arange(len(frame)) % 3
    return result


def test_paired_comparison_intersection_and_holm_family(tmp_path):
    data = cohort(20, 2)
    baseline, augmented = oof(data), oof(data, np.full(len(data), .5))
    augmented = augmented.iloc[:-4].copy()
    report = paired_modality_comparisons({"sensor": baseline, "with_audio": augmented},
                                         [("with_audio", "sensor"), ("with_text", "sensor")],
                                         tmp_path, n_bootstrap=50)
    first, unavailable = report["comparisons"]
    assert first["n_common_samples"] == len(augmented)
    assert first["n_paired_predicted_samples"] == len(augmented)
    assert first["delta_a_minus_b"] <= 0
    assert first["p_holm"] >= first["p_value"]
    assert unavailable["p_holm"] is None
    assert report["holm_family_size"] == 2
    assert stat.S_IMODE((tmp_path / "modality_comparisons.json").stat().st_mode) == 0o600


def test_paired_comparison_rejects_subject_target_and_fold_mismatches():
    table = oof(cohort(10, 1))
    for column, value in (("subject_id", "other-person"), ("target", 1-table.iloc[0].target), ("outer_fold", 9)):
        other = table.copy()
        other.loc[0, column] = value
        with pytest.raises(ValueError):
            paired_modality_comparisons({"a": table, "b": other}, [("a", "b")], n_bootstrap=5)


def test_subgroups_keep_abstentions_and_hide_category_values(tmp_path, capsys):
    data = cohort(20, 1)
    data.loc[:4, "probability"] = np.nan
    data.loc[19, "partition"] = "temporal_test"
    data["target"] = data.target.astype(object)
    data.loc[19, "target"] = "SEALED_OUTCOME_DO_NOT_READ"
    report = subgroup_reports(data, coverage_column="coverage", output_dir=tmp_path)
    assert "secret-" not in json.dumps(report)
    assert capsys.readouterr().out == ""
    assert all(row["analysis_status"] == "small_n_descriptive_only" for row in report["groups"])
    assert any(row["coverage"] < 1 for row in report["groups"] if row["coverage"] is not None)
    assert all(row["calibration_slope"] is None for row in report["groups"])
    key = pd.read_parquet(tmp_path / "private_subgroup_key.parquet")
    assert key.category.astype(str).str.contains("secret-").any()


def test_geography_engine_fit_calibration_and_domain_are_disjoint(tmp_path):
    data = cohort(60, 2)
    report = run_geography_validation(data, ["sensor__mean", "sensor__second"], output_dir=tmp_path,
                                       min_test_subjects=10)
    assert report["status"] == "ok"
    assert report["validation_type"] == "development_internal_external_geography_not_institution"
    assert "secret-" not in json.dumps(report)
    roles = pd.read_parquet(tmp_path / "geography_roles.parquet")
    for _, domain in roles.groupby("geography"):
        sets = {name: set(table.subject_id) for name, table in domain.groupby("role")}
        assert not sets["fit"] & sets["calibration"]
        assert not sets["fit"] & sets["held_geography"]
        assert not sets["calibration"] & sets["held_geography"]
    assert roles.groupby(["geography", "subject_id"]).role.nunique().max() == 1


def test_geography_does_not_access_test_outcomes(tmp_path):
    data = cohort(60, 2)
    test = data.iloc[:2].copy()
    test["subject_id"], test["sample_id"], test["partition"] = "sealed-subject", ["sealed-0", "sealed-1"], "temporal_test"
    test["phq9"] = "SEALED_OUTCOME_DO_NOT_READ"
    report = run_geography_validation(pd.concat([data, test], ignore_index=True), ["sensor__mean"], output_dir=tmp_path)
    assert report["status"] == "ok"
    assert "sealed-subject" not in set(pd.read_parquet(tmp_path / "geography_roles.parquet").subject_id)


def test_longitudinal_optional_dependency_status(monkeypatch):
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "statsmodels" else original(name))
    result = longitudinal_associations(cohort(), ["sensor__mean"])
    assert result["status"] == "optional_dependency_unavailable"
    assert result["dependency"] == "statsmodels"


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="optional statsmodels unavailable")
def test_mixed_effect_between_and_within_terms_are_distinct():
    result = longitudinal_associations(cohort(50, 5), ["sensor__mean"])
    assert result["exploratory"] and not result["causal_interpretation"]
    assert {row["term"] for row in result["results"]} == {"between_subject_mean", "within_subject_deviation"}
    assert result["fdr_family_size"] == 2
    assert all("q_bh_exploratory" in row for row in result["results"])
    assert all(row["status"] == "ok" for row in result["results"])


def test_retention_distinguishes_maturity_and_unknown_missingness(tmp_path):
    visits = pd.DataFrame({"subject_id": ["early", "late"], "index_date": pd.to_datetime(["2023-01-01", "2023-05-20"]),
                           "enrolled_date": pd.to_datetime(["2023-01-01", "2023-05-20"]), "wave": [1, 1]})
    result = retention_report(visits, "2023-07-01", output_dir=tmp_path, max_waves=2)
    followup = result["waves"][1]
    assert followup["n_overdue_missing_status_unknown"] == 1
    assert followup["n_not_yet_due_proxy"] == 1
    assert not result["actual_dropout_estimated"]
    assert result["withdrawal_information"] == "unknown"
    states = pd.read_parquet(tmp_path / "private_retention_states.parquet")
    assert len(states) == 4


def test_longitudinal_does_not_report_success_without_repeated_visits():
    result = longitudinal_associations(cohort(20, 1), ["sensor__mean"])
    if result["status"] != "optional_dependency_unavailable":
        assert result["status"] == "no_estimable_associations"
        assert result["n_estimable_terms"] == 0
        assert all(row["status"] == "insufficient_repeated_subjects" for row in result["results"])


def test_regression_metrics_equal_participant_weight():
    result = regression_metrics([0, 0, 10], [0, 0, 0], ["repeated", "repeated", "single"])
    assert result["mae"] == 5
    assert result["rmse"] == pytest.approx(np.sqrt(50))
    assert result["calibration_status"] == "constant_predictions"


def provenance():
    return {"past_only": True, "extraction_version": "synthetic-v1", "window_end_column": "sensor_window_end"}


def test_prognosis_requires_past_only_provenance_and_rejects_current_date():
    data = cohort(10, 2)
    missing = run_prognosis_regression(data, ["sensor__mean"], "2025-01-01", {})
    assert missing["status"] == "past_only_provenance_unavailable"
    data.loc[0, "sensor_window_end"] = data.loc[0, "index_date"]
    with pytest.raises(ValueError, match="strictly before"):
        run_prognosis_regression(data, ["sensor__mean"], "2025-01-01", provenance())


def test_prognosis_grouped_oof_purges_late_targets_and_test_rows(tmp_path):
    data = cohort(20, 2)
    data.loc[0, "target_date"] = pd.Timestamp("2026-01-01")
    data["future_phq9"] = data.future_phq9.astype(object)
    data.loc[0, "future_phq9"] = "SEALED_AFTER_CUTOFF"
    data.loc[2, "sensor__mean"] = np.nan
    test = data.iloc[4:6].copy()
    test["subject_id"], test["sample_id"], test["partition"] = "sealed-person", ["sealed-0", "sealed-1"], "temporal_test"
    test["future_phq9"] = "SEALED_TEST"
    result = run_prognosis_regression(pd.concat([data, test], ignore_index=True), ["sensor__mean"], "2025-01-01",
                                      provenance(), tmp_path, outer_folds=4)
    assert result["status"] == "ok"
    assert result["n_excluded_development_pairs"] == 1
    assert len(result["models"]) == 3
    predictions = pd.read_parquet(tmp_path / "prognosis_oof_predictions.parquet")
    assert "sample-0-0" not in set(predictions.sample_id)
    assert "sealed-person" not in set(predictions.subject_id)
    assert predictions.groupby("subject_id").outer_fold.nunique().max() == 1
    augmented = predictions[predictions.model.eq("current_phq_plus_sensor")]
    assert augmented.prediction.isna().sum() == 1
    assert len(result["paired_common_sample_metrics"]) == 2


def test_no_sensor_values_abstains_without_dropping_population(tmp_path):
    data = cohort(10, 2)
    data["sensor__mean"] = np.nan
    result = run_prognosis_regression(data, ["sensor__mean"], "2025-01-01", provenance(), tmp_path, outer_folds=3)
    augmented = next(row for row in result["models"] if row["model"] == "current_phq_plus_sensor")
    assert augmented["status"] == "no_predictions"
    assert augmented["coverage"] == 0
    assert augmented["n_labelled_visits"] == len(data)
