"""Calendar/person/label leakage tests using synthetic records only."""
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.temporal_splits import assert_forward_safety, forward_splits, make_forward_fold


def frame(rows):
    result = pd.DataFrame(rows, columns=["subject_id", "index_date"])
    result["index_date"] = pd.to_datetime(result.index_date)
    result["sample_id"] = [f"synthetic-{i}" for i in range(len(result))]
    result["partition"] = "development"
    result["y_unused"] = np.arange(len(result)) % 2
    # Deliberately unrelated DataFrame index labels: outputs must be iloc.
    result.index = np.arange(1000, 1000 + len(result))
    return result


def calendar_example():
    return frame([
        ("fit", "2025-07-01"), ("fit", "2025-08-05"),
        ("cal", "2025-07-02"), ("cal", "2025-08-20"), ("cal", "2025-09-16"),
        ("valid", "2025-07-03"), ("valid", "2025-08-25"), ("valid", "2025-10-01"), ("valid", "2025-10-31"),
        ("gap", "2025-08-06"), ("gap", "2025-09-17"),
        ("outside", "2025-11-01"),
    ])


def test_outer_calendar_boundaries_all_person_rows_purged_and_positional_indices():
    data = calendar_example()
    fold = make_forward_fold(data, "2025-10-01", "2025-11-01")
    assert fold["fit"].tolist() == [0, 1]
    assert fold["calibration"].tolist() == [3, 4]
    assert fold["validation"].tolist() == [7, 8]
    assert set(data.iloc[fold["fit"]].subject_id) == {"fit"}
    meta = fold["metadata"]
    assert meta["status"] == "ok"
    assert meta["partitions"]["fit"]["n_participants"] == 1
    assert meta["calibration_interval"] == {"start_inclusive": "2025-08-20T00:00:00", "end_exclusive": "2025-09-17T00:00:00"}
    assert meta["fit_end_exclusive"] == "2025-08-06T00:00:00"
    assert meta["purge_counts"]["fit_rows_removed_for_participant_overlap"] == 2
    assert meta["purge_counts"]["calibration_rows_removed_for_validation_participants"] == 1
    assert all(value is True for value in meta["checks"].values())
    # Metadata contains no identifiers or record tables.
    encoded = json.dumps(meta)
    assert "synthetic-" not in encoded and "subject_id" not in encoded


def test_assertion_rejects_same_person_across_roles_even_without_row_overlap():
    data = calendar_example()
    with pytest.raises(ValueError, match="participant"):
        assert_forward_safety(data, [0, 2], [7], calibration=[3])
    with pytest.raises(ValueError, match="participant"):
        assert_forward_safety(data, [5], [7])
    with pytest.raises(ValueError, match="participant"):
        assert_forward_safety(data, [0], [7], calibration=[6])


def test_strict_gap_rejects_exact_boundary_and_unknown_or_duplicate_positions():
    data = frame([("a", "2025-09-01"), ("b", "2025-09-15"), ("c", "2025-09-16")])
    with pytest.raises(ValueError, match="gap"):
        assert_forward_safety(data, [0], [1], gap_days=14)
    assert assert_forward_safety(data, [0], [2], gap_days=14)["status"] == "ok"
    for invalid in ([0, 0], [-1], [9], [0.5], [True], [[0]]):
        with pytest.raises(ValueError):
            assert_forward_safety(data, invalid, [2])


def test_inner_calendar_folds_expand_and_purge_validation_people_label_independently():
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    data = frame([(f"p-{i}", date) for i, date in enumerate(dates)])
    # Repeated future participant: remove the old row from that fold's fit.
    data.iloc[0, data.columns.get_loc("subject_id")] = data.iloc[60].subject_id
    original = forward_splits(data, n_splits=3, gap_days=14, min_train_fraction=.4)
    assert len(original) == 3
    last_fit_date = None
    for fit, valid in original:
        assert not set(data.iloc[fit].subject_id) & set(data.iloc[valid].subject_id)
        assert data.iloc[fit].index_date.max() + pd.Timedelta(days=14) < data.iloc[valid].index_date.min()
        assert_forward_safety(data, fit, valid)
        if last_fit_date is not None:
            assert data.iloc[fit].index_date.max() > last_fit_date
        last_fit_date = data.iloc[fit].index_date.max()
    assert 0 not in original[0][0]
    changed = data.copy()
    changed["y_unused"] = np.random.default_rng(8).normal(size=len(data))
    repeated = forward_splits(changed, n_splits=3, gap_days=14, min_train_fraction=.4)
    for first, second in zip(original, repeated):
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_inner_boundaries_use_whole_calendar_days_and_ignore_row_order():
    data = frame([("a", "2025-01-01 01:00"), ("b", "2025-01-01 23:00"),
                  ("c", "2025-01-02 02:00"), ("d", "2025-01-03 03:00"),
                  ("e", "2025-01-03 22:00"), ("f", "2025-01-04 06:00")])
    splits = forward_splits(data, n_splits=2, gap_days=0, min_train_fraction=.5)
    assert len(splits) == 2
    assert splits[0][1].tolist() == [3, 4]
    shuffled = data.sample(frac=1, random_state=3)
    shuffled_splits = forward_splits(shuffled, n_splits=2, gap_days=0, min_train_fraction=.5)
    for (fit, valid), (other_fit, other_valid) in zip(splits, shuffled_splits):
        assert set(data.iloc[fit].sample_id) == set(shuffled.iloc[other_fit].sample_id)
        assert set(data.iloc[valid].sample_id) == set(shuffled.iloc[other_valid].sample_id)


def test_explicit_future_label_availability_is_purged_at_fit_and_calibration_boundaries():
    data = calendar_example()
    data["target_date"] = data.index_date.copy()
    data.iloc[1, data.columns.get_loc("target_date")] = pd.Timestamp("2025-08-06")
    data.iloc[4, data.columns.get_loc("target_date")] = pd.Timestamp("2025-09-17")
    with pytest.raises(ValueError, match="explicit"):
        make_forward_fold(data, "2025-10-01", "2025-11-01")
    explicit = make_forward_fold(data, "2025-10-01", "2025-11-01", label_time_column="target_date")
    assert explicit["fit"].tolist() == [0]
    assert explicit["calibration"].tolist() == [3]
    assert explicit["metadata"]["purge_counts"]["fit_rows_removed_for_label_availability"] == 1
    assert explicit["metadata"]["purge_counts"]["calibration_rows_removed_for_label_availability"] == 1
    with pytest.raises(ValueError, match="availability"):
        assert_forward_safety(data, [1], [7], calibration=[3], label_time_column="target_date")


def test_inner_forward_purges_future_labels_and_cannot_override_target_date_with_early_times():
    data = frame([(f"p-{i}", date) for i, date in enumerate(pd.date_range("2025-01-01", periods=100))])
    data["target_date"] = data.index_date + pd.Timedelta(days=30)
    with pytest.raises(ValueError):
        forward_splits(data)
    explicit = forward_splits(data, label_time_column="target_date")
    for fit, valid in explicit:
        if len(fit):
            assert data.iloc[fit].target_date.max() + pd.Timedelta(days=14) < data.iloc[valid].index_date.min()
    with pytest.raises(ValueError):
        forward_splits(data, label_time_column="index_date")


def test_explicit_late_label_availability_cannot_rescue_a_target_before_the_index():
    data = calendar_example()
    data["target_date"] = data.index_date - pd.Timedelta(days=1)
    data["label_available_date"] = data.index_date + pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="precede"):
        make_forward_fold(data, "2025-10-01", "2025-11-01", label_time_column="label_available_date")


def test_mixed_test_partitions_invalid_dates_duplicate_samples_and_unknown_label_dates_fail_closed():
    data = calendar_example()
    for changed in (data.assign(partition="temporal_test"), data.drop(columns="partition"),
                    data.assign(index_date=pd.NaT), data.assign(sample_id="duplicate"),
                    data.assign(subject_id=""), data.assign(index_date=np.arange(len(data))),
                    data.assign(target_date=pd.NaT)):
        with pytest.raises(ValueError):
            make_forward_fold(changed, "2025-10-01", "2025-11-01")
    mixed = data.copy()
    mixed.iloc[0, mixed.columns.get_loc("partition")] = "temporal_test"
    with pytest.raises(ValueError):
        forward_splits(mixed)
    with pytest.raises(ValueError):
        make_forward_fold(data, "2025-11-01", "2025-10-01")
    with pytest.raises(ValueError):
        make_forward_fold(data, 20251001, "2025-11-01")
    aware = data.copy()
    aware["index_date"] = aware.index_date.dt.tz_localize("Asia/Seoul")
    with pytest.raises(ValueError):
        forward_splits(aware)


def test_empty_partitions_are_reported_instead_of_hidden_or_randomly_repaired():
    data = frame([("same", "2025-01-01"), ("same", "2025-10-05")])
    fold = make_forward_fold(data, "2025-10-01", "2025-11-01")
    assert not len(fold["fit"]) and not len(fold["calibration"])
    assert fold["metadata"]["status"] == "empty_partition"
    assert fold["metadata"]["checks"]["strict_index_gap_fit_validation"] is None
    inner = forward_splits(data, gap_days=14)
    assert len(inner) == 1 and len(inner[0][0]) == 0 and len(inner[0][1]) == 1


@pytest.mark.parametrize("kwargs", [{"gap_days": -1}, {"gap_days": 1.5}, {"calibration_days": 0}, {"gap_days": True}])
def test_invalid_outer_durations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_forward_fold(calendar_example(), "2025-10-01", "2025-11-01", **kwargs)


@pytest.mark.parametrize("kwargs", [{"n_splits": 0}, {"n_splits": True}, {"min_train_fraction": 0}, {"min_train_fraction": 1}, {"min_train_fraction": np.nan}])
def test_invalid_inner_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        forward_splits(calendar_example(), **kwargs)
