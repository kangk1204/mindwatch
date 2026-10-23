"""Statistical contract regressions; all records and predictions are synthetic."""
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mindwatch.models import calibrate
from mindwatch.validation import classification_metrics
from mindwatch.research.benchmarks import BenchmarkConfig, _estimator
from mindwatch.research.monitoring_models import fit_monitoring_bundle
from mindwatch.research.monitoring_splits import make_monitoring_fold
from mindwatch.research.oof_sensor import fit_offset, time_oof
from mindwatch.research.monitoring_study import _persist


@pytest.mark.parametrize("labels", [[.8, 1.8], [-.1, 1.], [np.nan, 1.], [0., np.inf]])
def test_fractional_or_nonfinite_binary_outcomes_are_not_truncated(labels):
    for function in (classification_metrics, calibrate):
        with pytest.raises(ValueError, match="binary"):
            function(labels, [.1, .9], ["synthetic_a", "synthetic_b"])


@pytest.mark.parametrize("probabilities", [[-.1, .8], [.2, 1.1], [.2, np.nan]])
def test_calibration_validates_probabilities_before_support_fallback(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        calibrate([0, 1], probabilities, ["synthetic_a", "synthetic_b"])


def test_metrics_report_separated_calibration_as_unavailable():
    report = classification_metrics([0, 0, 1, 1], [.1, .2, .8, .9], list("abcd"))
    assert report["AUROC"] == 1.
    assert report["calibration_slope"] is None
    assert report["calibration_intercept"] is None
    assert report["calibration_status"] == "nonidentifiable_or_not_converged"


def test_elasticnet_differs_from_l2_on_every_supported_sklearn_api():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 8)), np.arange(60) % 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        elastic = _estimator("elasticnet", {"C": .1, "l1_ratio": .5}, 42, BenchmarkConfig()).fit(x, y)
        ridge = _estimator("elasticnet", {"C": .1, "l1_ratio": 0.}, 42, BenchmarkConfig()).fit(x, y)
    assert not any("l1_ratio parameter is only used" in str(item.message) for item in caught)
    assert np.max(abs(elastic.predict_proba(x) - ridge.predict_proba(x))) > .01


def oof_inputs():
    dates = pd.to_datetime(["2025-01-01"] * 4 + ["2025-04-01"] * 4)
    frame = pd.DataFrame({"sample_id": [f"synthetic_sample_{i}" for i in range(8)],
                          "subject_id": [f"synthetic_person_{i}" for i in range(8)],
                          "index_date": dates, "label_available_at": dates + pd.Timedelta(days=1),
                          "target": np.arange(8) % 2, "partition": "development"})
    x = pd.DataFrame({"h_previous_phq9": np.arange(8, dtype=float)})
    folds = [(np.arange(4), np.arange(4, 8))]
    return frame, x, folds


@pytest.mark.parametrize("name", ["h_target", "history__current_phq9", "h_future_phq9",
                                  "sensor__label", "ema__outcome", "demo__patient_id"])
def test_oof_rejects_disguised_current_targets_and_identifiers(name):
    frame, x, folds = oof_inputs()
    with pytest.raises(ValueError):
        time_oof(frame, x.rename(columns={x.columns[0]: name}), folds, LogisticRegression)


@pytest.mark.parametrize("corruption", ["future_index", "bad_gap", "float_positions", "missing_id", "reserved_test"])
def test_oof_rejects_invalid_time_membership_and_indices(corruption):
    frame, x, folds = oof_inputs()
    kwargs = {}
    if corruption == "future_index":
        frame.loc[:3, "index_date"] = pd.Timestamp("2026-01-01")
    elif corruption == "bad_gap":
        kwargs["gap_days"] = -100
    elif corruption == "float_positions":
        folds = [(np.arange(4) + .4, np.arange(4, 8))]
    elif corruption == "missing_id":
        frame.loc[0, "subject_id"] = None
    else:
        frame.loc[0, "partition"] = "temporal_test"
    with pytest.raises(ValueError):
        time_oof(frame, x, folds, LogisticRegression, **kwargs)


def test_oof_uses_class_identity_and_validation_labels_never_enter_fit():
    class ReverseClasses:
        classes_ = np.array([1, 0])
        def fit(self, x, y, sample_weight):
            self.probability = np.average(y, weights=sample_weight) / 2
            return self
        def predict_proba(self, x):
            return np.tile([self.probability, 1-self.probability], (len(x), 1))
    frame, x, folds = oof_inputs()
    first, _ = time_oof(frame, x, folds, ReverseClasses, policy="new")
    frame.loc[4:, "target"] = 1-frame.loc[4:, "target"]
    second, _ = time_oof(frame, x, folds, ReverseClasses, policy="new")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[4:], np.full(4, .25))


def test_monitoring_calibration_counts_people_not_repeated_visits():
    y = np.arange(60) % 2
    dates = pd.to_datetime(["2025-01-01"] * 40 + ["2025-03-01"] * 20)
    frame = pd.DataFrame({"sample_id": [f"synthetic_row_{i}" for i in range(60)],
        "subject_id": [f"synthetic_fit_{i}" for i in range(40)] + [f"synthetic_cal_{i%2}" for i in range(20)],
        "index_date": dates, "label_available_at": dates + pd.Timedelta(days=1),
        "history_available": True, "sensor_available": False, "ema_available": False,
        "y_phq9": y})
    data = SimpleNamespace(frame=frame, designs={"A": {
        "tabular": (y + np.random.default_rng(1).normal(size=60))[:, None],
        "tabular_names": ["history__previous_phq9"]}})
    bundle = fit_monitoring_bundle(data, np.arange(40), np.arange(40, 60),
        {"feature_set": "A", "model": "elasticnet", "parameters": {"C": .1, "l1_ratio": 0.}},
        {"fit_end_exclusive": "2025-02-01", "calibration_end_exclusive": "2025-04-01"})
    status = bundle.metadata["branches"]["A"]
    assert status["calibration_class_counts"] == {"0": 10, "1": 10}
    assert status["calibration_class_participants"] == {"0": 1, "1": 1}
    assert status["calibration_status"] == "insufficient_calibration"
    assert bundle.branches["A"].calibrator is None


def test_monitoring_rejects_earlier_unanchored_prediction_provenance():
    dates = pd.to_datetime(["2025-07-01", "2025-08-25", "2025-10-01"])
    frame = pd.DataFrame({"sample_id": ["synthetic_a", "synthetic_b", "synthetic_c"],
        "subject_id": ["synthetic_a", "synthetic_b", "synthetic_c"], "partition": "development",
        "index_date": dates, "label_available_at": dates + pd.Timedelta(days=1),
        "prediction_time": pd.to_datetime(["2025-07-01", "2025-08-25", "2025-01-01"])})
    with pytest.raises(ValueError, match="anchor"):
        make_monitoring_fold(frame, "2025-10-01", "2025-11-01")


def persistence_inputs():
    target = np.array([0, 1] * 10 + [None, None, None], dtype=object)
    previous = np.array([3., 12.] * 10 + [4., np.nan, 11.])
    frame = pd.DataFrame({"subject_id": [f"synthetic_person_{i}" for i in range(23)],
                          "y_gad7": target})
    data = SimpleNamespace(frame=frame, designs={"A": {
        "tabular": previous[:, None], "tabular_names": ["history__previous_gad7"]}})
    return data, np.arange(20), np.arange(20, 23)


def test_persistence_ignores_evaluation_targets_and_keeps_missing_history_rows():
    data, cal, valid = persistence_inputs()
    first = _persist(data, cal, valid, "y_gad7", "classification")
    data.frame.loc[valid, "y_gad7"] = "heldout_labels_must_not_be_read"
    second = _persist(data, cal, valid, "y_gad7", "classification")
    pd.testing.assert_frame_equal(first, second)
    assert len(second) == len(valid)
    assert np.isfinite(second.probability.iloc[[0, 2]]).all()
    assert np.isnan(second.probability.iloc[1])


def test_persistence_repeated_rows_do_not_satisfy_calibration_people_gate():
    data, cal, valid = persistence_inputs()
    data.frame.loc[cal, "subject_id"] = [f"synthetic_cal_{i%2}" for i in cal]
    result = _persist(data, cal, valid, "y_gad7", "classification")
    assert result.probability.isna().all()
    assert result.calibration_status.eq("insufficient_calibration").all()


def offset_inputs():
    x = pd.DataFrame({"s_activity": [0., 1., 4., 6., 8., np.nan]})
    y = np.array([0, 0, 0, 1, 1, 0], dtype=float)
    ids = np.array([f"synthetic_offset_{i}" for i in range(len(x))])
    return x, y, ids, np.full(len(x), .4), np.ones(len(x))


@pytest.mark.parametrize("name", ["s_target", "sensor__phq9", "s_subject_id", "s_future_outcome"])
def test_offset_rejects_target_and_identifier_sensor_names(name):
    x, y, ids, base, quality = offset_inputs()
    with pytest.raises(ValueError):
        fit_offset(x.rename(columns={"s_activity": name}), y, ids, base, quality)


@pytest.mark.parametrize("key", ["gamma", "penalty"])
@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_offset_rejects_invalid_fixed_correction_parameters(key, value):
    with pytest.raises(ValueError, match=key):
        fit_offset(*offset_inputs(), **{key: value})


@pytest.mark.parametrize("field", ["targets", "participants", "baseline", "quality"])
def test_offset_requires_aligned_training_vectors(field):
    args = list(offset_inputs())
    index = {"targets": 1, "participants": 2, "baseline": 3, "quality": 4}[field]
    args[index] = args[index][:-1]
    with pytest.raises(ValueError, match=field):
        fit_offset(*args)


def test_offset_rejects_misordered_series_and_infinite_sensor_values():
    x, y, ids, base, quality = offset_inputs()
    with pytest.raises(ValueError, match="misaligned"):
        fit_offset(x, y, pd.Series(ids, index=np.arange(len(x))[::-1]), base, quality)
    x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite sensor"):
        fit_offset(x, y, ids, base, quality)


def test_offset_missing_sensor_rows_cannot_fit_nonzero_corrections():
    args = list(offset_inputs())
    first = fit_offset(*args)
    args[1][-1] = 1
    second = fit_offset(*args)
    np.testing.assert_allclose(first.coefficients, second.coefficients, atol=1e-12, rtol=0)
    x, _, _, base, quality = args
    assert first.predict(base, x, quality)[-1] == base[-1]
    baseline = base.copy()
    baseline[0] = np.nan
    assert np.isnan(first.predict(baseline, x, quality)[0])


@pytest.mark.parametrize("corruption", ["probability_inf", "probability_range", "quality_nan", "sensor_inf"])
def test_offset_prediction_fails_closed_on_invalid_inputs(corruption):
    x, y, ids, base, quality = offset_inputs()
    model = fit_offset(x, y, ids, base, quality)
    if corruption == "probability_inf": base[0] = np.inf
    elif corruption == "probability_range": base[0] = 1.2
    elif corruption == "quality_nan": quality[0] = np.nan
    else: x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        model.predict(base, x, quality)
