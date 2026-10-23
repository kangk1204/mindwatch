"""Monitoring split software tests with synthetic records and no model fitting."""
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.monitoring_splits import assert_monitoring_safety, make_monitoring_fold, monitoring_forward_splits


def synthetic(rows):
    result = pd.DataFrame(rows, columns=["subject_id", "index_date", "label_available_at"])
    result["index_date"] = pd.to_datetime(result.index_date, format="mixed")
    result["label_available_at"] = pd.to_datetime(result.label_available_at, format="mixed")
    result["prediction_time"] = result.index_date
    result["sample_id"] = [f"synthetic-{i}" for i in range(len(result))]
    result["partition"] = "development"
    result["outcome_unused"] = np.arange(len(result)) % 2
    result["sensor__unused"] = np.arange(len(result), dtype=float)
    result.index = np.arange(900, 900+len(result))
    return result


def example():
    return synthetic([
        ("returning", "2025-07-01", "2025-07-02"),
        ("returning", "2025-08-25", "2025-09-16"),
        ("returning", "2025-10-10", "2025-11-02"),
        ("fit_only", "2025-08-05", "2025-08-05"),
        ("fit_late", "2025-08-05", "2025-08-06"),
        ("cal_only", "2025-07-10", "2025-07-11"),
        ("cal_only", "2025-08-20", "2025-08-21"),
        ("cal_late", "2025-09-16", "2025-09-17"),
        ("cal_end", "2025-09-17", "2025-09-17"),
        ("valid_late", "2025-10-31", "2025-11-05"),
        ("outside", "2025-11-01", "2025-11-01"),
    ])


def test_known_user_retains_genuine_past_and_strictly_purges_unavailable_training_labels():
    data = example()
    before = data.copy(deep=True)
    fold = make_monitoring_fold(data, "2025-10-01", "2025-11-01")
    assert fold["fit"].tolist() == [0, 3, 5]
    assert fold["calibration"].tolist() == [1, 6]
    assert fold["validation"].tolist() == [2, 9]
    meta = fold["metadata"]
    assert meta["mode"] == "known_user" and meta["status"] == "ok"
    assert meta["calendar_boundaries"]["fit_end_exclusive"] == "2025-08-06T00:00:00"
    assert meta["calendar_boundaries"]["calibration_start_inclusive"] == "2025-08-20T00:00:00"
    assert meta["calendar_boundaries"]["calibration_end_exclusive"] == "2025-09-17T00:00:00"
    assert meta["purge_counts"]["fit_rows_removed_for_label_availability"] == 1
    assert meta["purge_counts"]["calibration_rows_removed_for_label_availability"] == 1
    assert meta["participant_overlap_counts"]["fit_validation_participants"] == 1
    assert meta["participant_overlap_counts"]["fit_calibration_participants"] == 2
    assert meta["validation_participants_seen_in_fit_or_calibration"] == 1
    assert meta["validation_participants_not_seen_in_fit_or_calibration"] == 1
    assert all(value is True for value in meta["checks"].values())
    pd.testing.assert_frame_equal(data, before)


def test_validation_late_labels_remain_denominator_and_no_within_month_updates():
    data = example()
    fold = make_monitoring_fold(data, "2025-10-01", "2025-11-01")
    assert fold["metadata"]["validation_labels_available_at_or_after_window_end"] == 2
    assert fold["metadata"]["partitions"]["validation"]["n_rows"] == 2
    assert fold["metadata"]["within_month_updates"] is False
    assert set(data.iloc[fold["validation"]].sample_id).isdisjoint(data.iloc[fold["fit"]].sample_id)
    assert data.iloc[fold["fit"]].label_available_at.max() < pd.Timestamp("2025-08-06")
    # A future validation outcome's timing must not change frozen fit/cal roles.
    late = data.copy()
    late.iloc[2, late.columns.get_loc("label_available_at")] = pd.Timestamp("2026-02-01")
    second = make_monitoring_fold(late, "2025-10-01", "2025-11-01")
    for role in ("fit", "calibration", "validation"):
        assert np.array_equal(fold[role], second[role])


def test_new_user_purges_all_validation_history_and_calibration_people():
    data = example()
    fold = make_monitoring_fold(data, "2025-10-01", "2025-11-01", mode="new_user")
    assert fold["fit"].tolist() == [3]
    assert fold["calibration"].tolist() == [6]
    assert fold["validation"].tolist() == [2, 9]
    assert all(value == 0 for value in fold["metadata"]["participant_overlap_counts"].values())
    assert fold["metadata"]["purge_counts"]["fit_rows_removed_for_participant_policy"] == 2
    assert fold["metadata"]["purge_counts"]["calibration_rows_removed_for_participant_policy"] == 1
    known = make_monitoring_fold(data, "2025-10-01", "2025-11-01")
    with pytest.raises(ValueError, match="participant"):
        assert_monitoring_safety(data, known["fit"], known["validation"], calibration=known["calibration"], mode="new_user")


def test_safety_rejects_row_overlap_strict_index_gap_and_strict_label_gap():
    data = example()
    with pytest.raises(ValueError, match="row"):
        assert_monitoring_safety(data, [0], [0], mode="known_user")
    # Sep16 -> Oct1 is 15 days, but its label Sep17 is only 14 days before Oct1.
    dates = synthetic([("a", "2025-09-16", "2025-09-17"), ("a", "2025-10-01", "2025-10-02")])
    with pytest.raises(ValueError, match="gap"):
        assert_monitoring_safety(dates, [0], [1], gap_days=14)
    dates.iloc[0, dates.columns.get_loc("label_available_at")] = pd.Timestamp("2025-09-16")
    assert assert_monitoring_safety(dates, [0], [1])["status"] == "ok"
    dates.iloc[0, dates.columns.get_loc("index_date")] = pd.Timestamp("2025-09-17")
    dates.iloc[0, dates.columns.get_loc("prediction_time")] = pd.Timestamp("2025-09-17")
    dates.iloc[0, dates.columns.get_loc("label_available_at")] = pd.Timestamp("2025-09-17")
    with pytest.raises(ValueError, match="gap"):
        assert_monitoring_safety(dates, [0], [1])


def test_forecast_target_requires_explicit_valid_availability_and_provenance():
    data = example()
    data["target_date"] = data.label_available_at.copy()
    assert make_monitoring_fold(data, "2025-10-01", "2025-11-01")["metadata"]["status"] == "ok"
    with pytest.raises(ValueError, match="explicit"):
        make_monitoring_fold(data, "2025-10-01", "2025-11-01", label_time_column=None)
    before_target = data.copy()
    before_target.iloc[2, before_target.columns.get_loc("label_available_at")] = pd.Timestamp("2025-10-20")
    with pytest.raises(ValueError, match="availability"):
        make_monitoring_fold(before_target, "2025-10-01", "2025-11-01")
    future_provenance = data.copy()
    future_provenance["prediction_time"] = future_provenance.index_date + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="provenance"):
        make_monitoring_fold(future_provenance, "2025-10-01", "2025-11-01")
    fallback = make_monitoring_fold(data.drop(columns="prediction_time"), "2025-10-01", "2025-11-01")
    assert fallback["metadata"]["prediction_time_source"] == "index_date_anchor_prediction_time_not_provided"


def test_inner_known_user_overlap_and_new_user_purge_are_deterministic_and_label_agnostic():
    day = pd.date_range("2025-01-01", periods=120)
    data = synthetic([(f"p-{i % 30}", date, date+pd.Timedelta(days=2)) for i, date in enumerate(day)])
    original = data.copy(deep=True)
    known = monitoring_forward_splits(data, n_splits=2)
    new = monitoring_forward_splits(data, n_splits=2, mode="new_user")
    pd.testing.assert_frame_equal(data, original)
    assert len(known) == len(new) == 2
    assert any(set(data.iloc[a].subject_id) & set(data.iloc[b].subject_id) for a, b in known)
    for (fit, valid), (new_fit, new_valid) in zip(known, new):
        assert np.array_equal(valid, new_valid)
        assert not set(data.iloc[new_fit].subject_id) & set(data.iloc[new_valid].subject_id)
        assert data.iloc[fit].label_available_at.max()+pd.Timedelta(days=14) < data.iloc[valid].index_date.min()
    data["outcome_unused"] = np.random.default_rng(2).normal(size=len(data))
    data["sensor__unused"] = np.nan
    altered = monitoring_forward_splits(data, n_splits=2)
    for first, second in zip(known, altered):
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_inner_label_availability_purge_and_date_boundaries_do_not_adapt_to_labels():
    days = pd.date_range("2025-01-01", periods=100)
    data = synthetic([(f"p-{i}", date, date+pd.Timedelta(days=30)) for i, date in enumerate(days)])
    splits = monitoring_forward_splits(data, gap_days=14)
    for fit, valid in splits:
        if len(fit):
            assert data.iloc[fit].label_available_at.max()+pd.Timedelta(days=14) < data.iloc[valid].index_date.min()
        assert not data.iloc[valid].empty
    # Changing the availability of one early label changes only permissible fit
    # membership, never validation date boundaries.
    changed = data.copy()
    changed.iloc[0, changed.columns.get_loc("label_available_at")] = pd.Timestamp("2026-01-01")
    updated = monitoring_forward_splits(changed, gap_days=14)
    for (_, valid), (_, new_valid) in zip(splits, updated):
        assert np.array_equal(valid, new_valid)


def test_empty_partitions_and_aggregate_only_metadata_are_explicit():
    data = synthetic([("same", "2025-10-01", "2025-11-02")])
    fold = make_monitoring_fold(data, "2025-10-01", "2025-11-01")
    assert fold["metadata"]["status"] == "empty_partition"
    assert fold["fit"].dtype == np.int64 and not len(fold["fit"])
    assert fold["metadata"]["checks"]["strict_index_gap_fit_validation"] is None
    encoded = json.dumps(fold["metadata"])
    assert "synthetic-" not in encoded and "subject_id" not in encoded
    assert fold["metadata"]["class_support"] == "not_inspected_runner_must_check"


@pytest.mark.parametrize("change", ["test", "mixed", "duplicate", "missing_date", "missing_label", "bad_label", "missing_id", "aware", "missing_partition"])
def test_invalid_inputs_fail_closed(change):
    data = example()
    if change == "test": data["partition"] = "temporal_test"
    elif change == "mixed": data.iloc[0, data.columns.get_loc("partition")] = "temporal_test"
    elif change == "duplicate": data["sample_id"] = "duplicate"
    elif change == "missing_date": data["index_date"] = pd.NaT
    elif change == "missing_label": data = data.drop(columns="label_available_at")
    elif change == "bad_label": data["label_available_at"] = data.index_date-pd.Timedelta(days=1)
    elif change == "missing_id": data["subject_id"] = ""
    elif change == "aware": data["prediction_time"] = data.prediction_time.dt.tz_localize("Asia/Seoul")
    elif change == "missing_partition": data = data.drop(columns="partition")
    with pytest.raises(ValueError):
        make_monitoring_fold(data, "2025-10-01", "2025-11-01")


@pytest.mark.parametrize("kwargs", [{"mode": "unknown"}, {"gap_days": -1}, {"gap_days": True}, {"calibration_days": 0}])
def test_invalid_outer_settings_fail_closed(kwargs):
    with pytest.raises(ValueError):
        make_monitoring_fold(example(), "2025-10-01", "2025-11-01", **kwargs)


@pytest.mark.parametrize("kwargs", [{"mode": "unknown"}, {"n_splits": 0}, {"n_splits": True}, {"min_train_fraction": np.nan}, {"min_train_fraction": 1}, {"gap_days": -2}])
def test_invalid_inner_settings_fail_closed(kwargs):
    with pytest.raises(ValueError):
        monitoring_forward_splits(example(), **kwargs)
