"""Synthetic follow-up monitoring data tests; no real cohort records are read."""
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.monitoring_data import build_past_history, load_monitoring_data, personal_sensor_features


def test_past_history_excludes_own_and_future_scores_and_respects_release_proxy():
    visits = pd.DataFrame({"sample_id": ["a0", "a1", "a2"], "subject_id": ["a"]*3,
                           "index_date": pd.to_datetime(["2025-01-01 09:00", "2025-01-01 21:00", "2025-01-03 09:00"]),
                           "phq9": [5., 9., 22.], "gad7": [2., 4., 19.], "isi": [8., 10., 27.]})
    result = build_past_history(visits)
    assert result.history_available.tolist() == [False, False, True]
    assert result.loc[:1, "history__baseline_phq9"].isna().all()
    assert result.loc[2, "history__baseline_phq9"] == 5
    assert result.loc[2, "history__previous_phq9"] == 9
    assert result.loc[2, "history__past_mean_phq9"] == 7
    assert result.loc[2, "history__past_change_phq9"] == 4
    modified = visits.copy()
    modified.loc[2, ["phq9", "gad7", "isi"]] = [1., 1., 1.]
    pd.testing.assert_frame_equal(result, build_past_history(modified))
    shuffled = visits.iloc[[2, 0, 1]].copy()
    shuffled.index = [0, 0, 0]
    keyed = build_past_history(shuffled).set_index("sample_id").sort_index()
    pd.testing.assert_frame_equal(result.set_index("sample_id").sort_index(), keyed)


def test_personal_baseline_exact_intervals_and_future_invariance():
    t = pd.Timestamp("2025-03-01")
    dates = pd.date_range(t-pd.Timedelta(days=43), t+pd.Timedelta(days=2))
    daily = pd.DataFrame({"subject_id": "a", "date": dates, "steps": np.arange(len(dates), dtype=float)})
    visits = pd.DataFrame({"sample_id": ["a1"], "subject_id": ["a"], "index_date": [t]})
    result = personal_sensor_features(visits, daily, ["steps"])
    baseline = daily.loc[daily.date.ge(t-pd.Timedelta(days=42)) & daily.date.lt(t-pd.Timedelta(days=14)), "steps"]
    recent = daily.loc[daily.date.ge(t-pd.Timedelta(days=14)) & daily.date.lt(t), "steps"]
    assert result.loc[0, "quality__personal_steps__baseline_observed_days"] == 28
    assert result.loc[0, "sensor__personal_steps__baseline_median_28d"] == baseline.median()
    assert result.loc[0, "sensor__personal_steps__recent_minus_baseline"] == recent.median()-baseline.median()
    iqr = baseline.quantile(.75)-baseline.quantile(.25)
    assert result.loc[0, "sensor__personal_steps__robust_z"] == pytest.approx((recent.median()-baseline.median())/(iqr/1.349))
    modified = daily.copy()
    modified["steps"] = modified.steps.astype(object)
    modified.loc[modified.date.ge(t), "steps"] = "FUTURE_NOT_NUMERIC"
    modified.loc[modified.date.lt(t-pd.Timedelta(days=42)), "steps"] = 99999999
    pd.testing.assert_frame_equal(result, personal_sensor_features(visits, modified, ["steps"]))


def test_personal_insufficient_history_zero_iqr_and_channel_specific_days():
    t = pd.Timestamp("2025-03-01")
    dates = pd.date_range(t-pd.Timedelta(days=42), t-pd.Timedelta(days=1))
    daily = pd.DataFrame({"subject_id": "a", "date": dates, "steps": 3., "sleep": np.nan})
    daily.loc[:12, "sleep"] = 2.
    visits = pd.DataFrame({"sample_id": ["a1", "absent"], "subject_id": ["a", "unknown"], "index_date": [t, t]})
    result = personal_sensor_features(visits, daily, ["steps", "sleep"])
    assert result.loc[0, "quality__personal_steps__baseline_zero_iqr"] == 1
    assert np.isnan(result.loc[0, "sensor__personal_steps__robust_z"])
    assert result.loc[0, "quality__personal_sleep__baseline_observed_days"] == 13
    assert result.loc[0, "quality__personal_sleep__baseline_insufficient"] == 1
    assert np.isnan(result.loc[0, "sensor__personal_sleep__baseline_median_28d"])
    assert result.loc[1, "quality__personal_steps__baseline_insufficient"] == 1
    assert len(result) == 2


def synthetic_dataset(path):
    records = []
    specs = {"a": [6, 2, 12], "b": [3, 15], "c": [7], "d": [9, 4]}
    for subject, scores in specs.items():
        for wave, score in enumerate(scores):
            records.append({"sample_id": f"{subject}{wave}", "subject_id": subject,
                            "index_date": pd.Timestamp("2025-01-01")+pd.Timedelta(days=124*wave),
                            "phq9": score, "gad7": min(score, 21), "isi": score,
                            "age_group": "synthetic", "sex": "0"})
    visits = pd.DataFrame(records)
    split = visits[["sample_id", "subject_id", "index_date"]].copy()
    split["partition"] = "development"
    visits.to_parquet(path/"visits.parquet", index=False)
    split.to_parquet(path/"split_manifest.parquet", index=False)
    (path/"split.json").write_text(json.dumps({"original_test_subjects_reserved": 3, "cutoff": "2026-01-01"}))
    (path/"data_manifest.json").write_text(json.dumps({"synthetic": True}))
    channels = [f"signal_{i}" for i in range(11)]
    values = np.ones((len(visits), 14, len(channels)), dtype=np.float32)
    mask = np.ones_like(values)
    # Sparse follow-up is retained, and exact four-day availability is sufficient.
    a1 = visits.index[visits.sample_id.eq("a1")][0]
    d1 = visits.index[visits.sample_id.eq("d1")][0]
    mask[a1, :10] = 0
    mask[d1] = 0
    values[~mask.astype(bool)] = np.nan
    features = pd.DataFrame({"sample_id": visits.sample_id,
                             "quality__sensor_observed_days": mask.any(2).sum(1),
                             "quality__sensor_eligible": mask.any(2).sum(1) >= 4,
                             "sensor__signal_0__mean": 1., "sensor__signal_0__coverage": mask[:, :, 0].mean(1),
                             "ema__ema_mood__coverage": np.where(visits.sample_id.isin(["a2", "d1"]), 1/14, 0),
                             "ema__ema_mood__mean": np.where(visits.sample_id.isin(["a2", "d1"]), 4., np.nan),
                             "demo__sex_code_0": 1., "history__previous_phq9": 999999.})
    features.to_parquet(path/"features_14d.parquet", index=False)
    reverse = np.arange(len(visits))[::-1]
    np.savez(path/"sequences_14d.npz", sample_ids=visits.sample_id.to_numpy(dtype=str)[reverse], values=values[reverse],
             mask=mask[reverse], feature_names=np.asarray(channels, dtype=str))
    daily = []
    for subject in specs:
        for day in pd.date_range("2024-11-01", "2025-10-01"):
            daily.append({"subject_id": subject, "date": day, **{name: float(day.day+i) for i, name in enumerate(channels)}})
    pd.DataFrame(daily).to_parquet(path/"daily.parquet", index=False)
    return visits, features


def test_monitoring_entry_once_followups_all_retained_and_designs_separated(tmp_path, capsys):
    synthetic_dataset(tmp_path)
    data = load_monitoring_data(tmp_path)
    assert capsys.readouterr().out == ""
    assert set(data.frame.sample_id) == {"a1", "a2", "d1"}
    assert set(data.frame.subject_id) == {"a", "d"}
    assert len(data.source_frame) == 6
    assert data.provenance["entry_participants"] == 3
    assert data.provenance["followup_visits"] == 3
    assert data.frame.phq9.lt(5).sum() == 2
    keyed = data.frame.set_index("sample_id")
    assert keyed.loc["a1", "sensor_available"]
    assert not keyed.loc["d1", "sensor_available"]
    assert keyed.loc["d1", "ema_available"]
    assert not keyed.loc["a1", "ema_available"]
    assert keyed.loc["a1", "baseline_phq9"] == 6
    assert keyed.loc["a1", "history__previous_phq9"] == 6
    assert keyed.loc["a2", "history__previous_phq9"] == 2
    assert data.frame.history_available.all()
    assert (data.frame.label_available_at == data.frame.index_date+pd.Timedelta(days=1)).all()
    assert {"y_phq9", "y_gad7", "y_isi", "phq9", "gad7", "isi"}.issubset(data.frame)
    for name, design in data.designs.items():
        assert design["tabular"].shape[0] == len(data.frame)
        assert design["tabular"].shape[1] == len(design["tabular_names"])
        assert all(col.startswith(("history__", "demo__")) for col in data.designs["A"]["tabular_names"])
        assert not any("forecast_current" in col for col in design["tabular_names"])
        assert not any(col in design["tabular_names"] for col in ["phq9", "gad7", "isi", "y_phq9", "subject_id", "sample_id"])
        if name in {"B", "D"}:
            assert design["sequence_columns"] == 14*11*3
            assert design["sequence"].shape[1] == 14*11*3+len(design["static_names"])
            assert all(col.startswith(("history__", "demo__", "ema__")) for col in design["static_names"])
    assert set(data.designs) == {"A", "B", "C", "D"}
    assert data.provenance["as_of_upload_time"] == "unknown"
    assert not data.provenance["old_test_loaded"]


def test_prognosis_includes_initial_index_and_has_future_release(tmp_path):
    synthetic_dataset(tmp_path)
    data = load_monitoring_data(tmp_path)
    assert set(data.prognosis_frame.sample_id) == {"a0", "a1", "d0"}
    assert data.prognosis_frame.history_available.all()
    assert not data.prognosis_frame.set_index("sample_id").loc["a0", "past_history_available"]
    assert data.source_frame.set_index("sample_id").loc["a0", "history_available"] == False
    assert data.prognosis_frame.horizon_days.between(90, 180).all()
    assert data.prognosis_frame.partition.eq("development").all()
    assert (data.prognosis_frame.label_available_at == data.prognosis_frame.target_date+pd.Timedelta(days=1)).all()
    assert {"future_phq9", "worsening_ge5", "future_ge10"}.issubset(data.prognosis_frame)
    for name, design in data.prognosis_designs.items():
        assert design["tabular"].shape[0] == len(data.prognosis_frame)
        assert "history__forecast_current_phq9" in design["tabular_names"]
        assert "history__forecast_current_phq9" not in data.designs[name]["tabular_names"]


def test_loader_filters_sealed_membership_before_outcome_processing(tmp_path):
    visits, _ = synthetic_dataset(tmp_path)
    sealed = visits.iloc[[0]].copy()
    sealed["sample_id"], sealed["subject_id"] = "sealed", "sealed-person"
    sealed[["phq9", "gad7", "isi"]] = "DO_NOT_USE_SEALED_LABEL"
    combined = pd.concat([visits, sealed], ignore_index=True)
    combined[["phq9", "gad7", "isi"]] = combined[["phq9", "gad7", "isi"]].astype(str)
    combined.to_parquet(tmp_path/"visits.parquet", index=False)
    split = pd.read_parquet(tmp_path/"split_manifest.parquet")
    sealed_split = sealed[["sample_id", "subject_id", "index_date"]].copy()
    sealed_split["partition"] = "temporal_test"
    pd.concat([split, sealed_split], ignore_index=True).to_parquet(tmp_path/"split_manifest.parquet", index=False)
    data = load_monitoring_data(tmp_path)
    assert "sealed-person" not in set(data.source_frame.subject_id)
    assert data.provenance["followup_visits"] == 3


def test_loader_detects_sensor_eligibility_drift(tmp_path):
    _, features = synthetic_dataset(tmp_path)
    features.loc[features.sample_id.eq("a1"), "quality__sensor_observed_days"] = 9
    features.to_parquet(tmp_path/"features_14d.parquet", index=False)
    with pytest.raises(ValueError, match="observed-day counts differ"):
        load_monitoring_data(tmp_path)


def test_current_and_future_labels_cannot_change_past_monitoring_design(tmp_path):
    visits, _ = synthetic_dataset(tmp_path)
    before = load_monitoring_data(tmp_path)
    visits.loc[visits.sample_id.eq("a1"), ["phq9", "gad7", "isi"]] = [20, 20, 20]
    visits.loc[visits.sample_id.eq("a2"), ["phq9", "gad7", "isi"]] = [0, 0, 0]
    visits.to_parquet(tmp_path/"visits.parquet", index=False)
    after = load_monitoring_data(tmp_path)
    row_before = before.frame.index[before.frame.sample_id.eq("a1")][0]
    row_after = after.frame.index[after.frame.sample_id.eq("a1")][0]
    for name in before.designs:
        np.testing.assert_allclose(before.designs[name]["tabular"][row_before], after.designs[name]["tabular"][row_after], equal_nan=True)
