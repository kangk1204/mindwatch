import pytest
pytest.importorskip("torch")
"""Synthetic known-user monitoring tests; no patient rows or prior results."""
from copy import deepcopy
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.monitoring_models import (branch_available, fit_base_estimator,
                                        fit_monitoring_bundle, monitoring_regression_metrics,
                                        predict_base)


def fixture():
    rng = np.random.default_rng(42)
    n = 166
    labels = np.arange(n) % 2
    history = np.ones(n, bool)
    sensor = np.arange(n) % 4 < 2
    ema = np.arange(n) % 8 < 4
    sensor[160:] = [1, 1, 0, 0, 1, 1]
    ema[160:] = [1, 0, 1, 0, 1, 1]
    history[164] = False
    dates = pd.to_datetime([pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(80)] +
                           [pd.Timestamp("2025-04-01") + pd.Timedelta(days=i) for i in range(80)] +
                           [pd.Timestamp("2025-07-01") + pd.Timedelta(days=i) for i in range(6)])
    frame = pd.DataFrame({"sample_id": [f"synthetic_sample_{i}" for i in range(n)],
                           "subject_id": [f"synthetic_subject_{i%20}" for i in range(n)],
                           "index_date": dates, "label_available_at": dates + pd.Timedelta(days=1),
                           "history_available": history, "sensor_available": sensor, "ema_available": ema,
                           "partition": "development", "y_phq9": labels.astype(object),
                           "future_phq9": 10 + 5*labels + rng.normal(size=n)})
    frame.loc[160:, "y_phq9"] = "HELD_OUT_LABEL_NOT_FOR_MODEL"
    h = np.column_stack((5 + labels + rng.normal(size=n), rng.normal(size=n)))
    s = rng.normal(size=n) + labels
    e = rng.normal(size=n) + labels
    s[~sensor], e[~ema] = np.nan, np.nan
    values = rng.normal(size=(n, 14, 2)).astype(np.float32)
    mask = np.ones_like(values)
    mask[~sensor] = 0
    values[mask == 0] = np.nan
    from mindwatch.research.neural import flatten_sequence, past_only_delta
    packed = flatten_sequence(values, mask, past_only_delta(mask))
    hnames = ["history__previous_phq9", "demo__age_decade"]
    designs = {
        "A": {"tabular": h.copy(), "tabular_names": hnames.copy()},
        "B": {"tabular": np.column_stack((h, s)), "tabular_names": hnames + ["sensor__mean"],
              "sequence": np.column_stack((packed, h)), "static_names": hnames.copy(), "sequence_columns": packed.shape[1]},
        "C": {"tabular": np.column_stack((h, e)), "tabular_names": hnames + ["ema__mood_mean"]},
        "D": {"tabular": np.column_stack((h, s, e)), "tabular_names": hnames + ["sensor__mean", "ema__mood_mean"],
              "sequence": np.column_stack((packed, h, e)), "static_names": hnames + ["ema__mood_mean"],
              "sequence_columns": packed.shape[1]},
    }
    data = SimpleNamespace(frame=frame, designs=designs)
    config = {"version": "synthetic_monitoring_v6", "window": 14, "seed": 42,
              "neural_seeds": (42, 2026, 2027), "neural_batch_size": 32, "neural_device": "cpu",
              "target_sensitivity": .8, "fit_end_exclusive": "2025-04-01",
              "calibration_end_exclusive": "2025-07-01"}
    return data, np.arange(80), np.arange(80, 160), config


def selection(branch="D", model="elasticnet", parameters=None):
    return {"feature_set": branch, "model": model, "parameters": parameters or {"C": .1, "l1_ratio": 0.0},
            "selection_status": "fixed_synthetic_candidate"}


def test_routing_preserves_every_row_order_and_known_user_calibration():
    data, fit, calibration, config = fixture()
    bundle = fit_monitoring_bundle(data, fit, calibration, selection(), config)
    rows = np.asarray([163, 160, 164, 161, 162, 165])
    predicted = bundle.predict(data, rows)
    assert predicted.used_modalities.tolist() == ["A", "D", "none", "B", "C", "D"]
    assert len(predicted) == len(rows)
    assert np.isnan(predicted.probability.iloc[2])
    assert predicted.fallback_reason.iloc[2] == "insufficient_history"
    assert predicted.calibration_status.drop(index=2).eq("platt_calibrated").all()
    assert predicted.threshold.drop(index=2).notna().all()
    assert set(bundle.branches) == {"A", "B", "C", "D"}
    assert bundle.metadata["fit_calibration_subject_overlap_allowed"]
    assert bundle.metadata["branches"]["D"]["calibration_class_counts"] == {"0": 10, "1": 10}
    assert bundle.branches["D"].base.model_name == "elasticnet"
    assert ":classification:y_phq9:D:elasticnet:" in predicted.model_version.iloc[1]


@pytest.mark.parametrize("branch,expected", [("A", ["A", "A", "A", "A"]),
                                            ("B", ["B", "B", "A", "A"]),
                                            ("C", ["C", "A", "C", "A"])])
def test_selected_lower_feature_set_does_not_gain_unrequested_modalities(branch, expected):
    data, fit, calibration, config = fixture()
    bundle = fit_monitoring_bundle(data, fit, calibration, selection(branch), config)
    assert bundle.predict(data, np.arange(160, 164)).used_modalities.tolist() == expected


def test_insufficient_calibration_is_identity_without_an_operating_threshold():
    data, fit, calibration, config = fixture()
    bundle = fit_monitoring_bundle(data, fit, calibration[:8], selection(), config)
    predicted = bundle.predict(data, np.asarray([160]))
    raw = predict_base(bundle.branches["D"].base, data, "D", np.asarray([160]))[0]
    assert predicted.probability.iloc[0] == raw
    assert np.isnan(predicted.threshold.iloc[0])
    assert predicted.calibration_status.iloc[0] == "insufficient_calibration"
    assert bundle.branches["D"].calibrator is None


def test_upper_fit_failure_is_recorded_and_falls_to_true_lower_family(monkeypatch):
    import mindwatch.research.monitoring_models as module
    data, fit, calibration, config = fixture()
    original = module.fit_base_estimator
    def unavailable_upper(data, branch, model, parameters, fit_rows, config, **kwargs):
        if branch == "D":
            raise RuntimeError("synthetic upper training unavailable")
        return original(data, branch, model, parameters, fit_rows, config, **kwargs)
    monkeypatch.setattr(module, "fit_base_estimator", unavailable_upper)
    bundle = fit_monitoring_bundle(data, fit, calibration,
                                    selection("D", "grud", {"hidden_dim": 8, "epochs": 1}), config)
    predicted = bundle.predict(data, np.asarray([160]))
    assert predicted.used_modalities.iloc[0] == "B"
    assert "branch_unfitted_D" in predicted.fallback_reason.iloc[0]
    assert ":y_phq9:B:elasticnet:" in predicted.model_version.iloc[0]
    assert bundle.metadata["branches"]["D"]["status"] == "fit_failed"
    assert bundle.metadata["branches"]["D"]["requested_model"] == "grud"


def test_insufficient_full_branch_class_support_falls_lower():
    data, fit, calibration, config = fixture()
    affected = fit[branch_available(data.frame, "D")[fit]]
    data.frame.loc[affected, "y_phq9"] = 0
    bundle = fit_monitoring_bundle(data, fit, calibration, selection(), config)
    assert bundle.metadata["branches"]["D"]["status"] == "insufficient_fit_support"
    assert bundle.predict(data, np.asarray([160])).used_modalities.iloc[0] == "B"


def test_declared_neural_support_gate_prevents_small_fit_and_falls_lower(monkeypatch):
    import mindwatch.research.monitoring_models as module
    data, fit, calibration, config = fixture()
    config["min_neural_people"] = 100
    config["min_neural_per_class"] = 20
    original = module.fit_base_estimator
    def no_small_neural(data, branch, model, parameters, fit_rows, config, **kwargs):
        if model in {"grud", "tcn"}:
            raise AssertionError("Support-gated neural fit must never be called")
        return original(data, branch, model, parameters, fit_rows, config, **kwargs)
    monkeypatch.setattr(module, "fit_base_estimator", no_small_neural)
    bundle = fit_monitoring_bundle(data, fit, calibration,
                                    selection("D", "tcn", {"epochs": 1}), config)
    assert bundle.metadata["branches"]["D"]["status"] == "insufficient_fit_support"
    assert bundle.metadata["branches"]["D"]["minimum_fit_participants"] == 100
    predicted = bundle.predict(data, np.array([160]))
    assert predicted.used_modalities.iloc[0] == "B"
    assert ":y_phq9:B:elasticnet:" in predicted.model_version.iloc[0]


def test_held_out_outcomes_never_affect_raw_or_routed_predictions_and_pickle():
    data, fit, calibration, config = fixture()
    bundle = fit_monitoring_bundle(data, fit, calibration, selection(), config)
    rows = np.arange(160, 166)
    before = bundle.predict(data, rows)
    data.frame["future_phq9"] = data.frame["future_phq9"].astype(object)
    data.frame.loc[160:, ["y_phq9", "future_phq9"]] = "CHANGED_UNUSED_LABEL"
    after = bundle.predict(data, rows)
    pd.testing.assert_frame_equal(before, after)
    restored = pickle.loads(pickle.dumps(bundle))
    pd.testing.assert_frame_equal(before, restored.predict(data, rows))


@pytest.mark.parametrize("bad", ["subject_id", "demo__patient_id", "sensor__phq9", "history__phq9", "ema__current_gad7"])
def test_wrong_schema_fails_instead_of_adding_targets_or_identifiers(bad):
    data, fit, calibration, config = fixture()
    data.designs["D"]["tabular_names"][-1] = bad
    with pytest.raises(ValueError):
        fit_monitoring_bundle(data, fit, calibration, selection(), config)


def test_prediction_schema_reordering_is_rejected_by_base_estimator():
    data, fit, _, config = fixture()
    rows = fit[branch_available(data.frame, "B")[fit]]
    estimator = fit_base_estimator(data, "B", "elasticnet", {"C": .1}, rows, config)
    data.designs["B"]["tabular_names"][:2] = data.designs["B"]["tabular_names"][1::-1]
    with pytest.raises(ValueError, match="order changed"):
        predict_base(estimator, data, "B", np.asarray([160]))


def test_fit_base_never_filters_unavailable_or_missing_label_rows():
    data, fit, _, config = fixture()
    with pytest.raises(ValueError, match="unavailable"):
        fit_base_estimator(data, "B", "elasticnet", {}, fit, config)
    data.frame.loc[0, "y_phq9"] = np.nan
    with pytest.raises(ValueError, match="discard"):
        fit_base_estimator(data, "A", "elasticnet", {}, fit, config)


def test_row_overlap_and_label_availability_crossing_boundaries_are_rejected():
    data, fit, calibration, config = fixture()
    with pytest.raises(ValueError, match="row overlap"):
        fit_monitoring_bundle(data, fit, np.r_[calibration, 0], selection(), config)
    data.frame.loc[0, "label_available_at"] = pd.Timestamp("2025-04-02")
    with pytest.raises(ValueError, match="availability boundary"):
        fit_monitoring_bundle(data, fit, calibration, selection(), config)


def test_regression_bundle_routes_and_keeps_distinct_version_from_classification():
    data, fit, calibration, config = fixture()
    regression = fit_monitoring_bundle(data, fit, calibration,
                                        selection("D", "ridge", {"alpha": 10}), config,
                                        target_column="future_phq9", task="regression")
    classification = fit_monitoring_bundle(data, fit, calibration, selection(), config)
    rows = np.arange(160, 166)
    scores, risks = regression.predict(data, rows), classification.predict(data, rows)
    assert "score" in scores and "probability" not in scores
    assert scores.used_modalities.tolist() == risks.used_modalities.tolist()
    assert scores.calibration_status.drop(index=4).eq("not_applicable_regression").all()
    assert scores.threshold.isna().all()
    assert scores.model_version.iloc[0] != risks.model_version.iloc[0]
    assert regression.metadata["branches"]["A"]["actual_model"] == "ridge"
    measured = monitoring_regression_metrics(data.frame.iloc[rows].future_phq9, scores.score,
                                              data.frame.iloc[rows].subject_id)
    assert measured["n_labelled_visits"] == 6 and measured["n_predicted_visits"] == 5
    assert measured["mae"] >= 0 and measured["rmse"] >= measured["mae"]


def test_forecast_current_scores_require_explicit_separate_forecast_task():
    data, fit, _, config = fixture()
    data.designs["A"]["tabular_names"][0] = "history__forecast_current_phq9"
    with pytest.raises(ValueError, match="Current/raw"):
        fit_base_estimator(data, "A", "ridge", {}, fit, config, target_column="future_phq9", task="regression")
    config["allow_current_state_for_forecast"] = True
    estimator = fit_base_estimator(data, "A", "ridge", {}, fit, config, target_column="future_phq9", task="regression")
    assert np.isfinite(predict_base(estimator, data, "A", np.asarray([160]), task="regression")).all()
    with pytest.raises(ValueError, match="Current-state"):
        fit_base_estimator(data, "A", "elasticnet", {}, fit, config)


@pytest.mark.parametrize("family", ["grud", "tcn"])
def test_three_seed_neural_average_and_unseen_channel_are_stable(family):
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data, fit, _, config = fixture()
        rows = fit[branch_available(data.frame, "B")[fit]]
        matrix = data.designs["B"]["sequence"]
        # Channel1 is absent in the fit. Make new held-out observations there.
        values = matrix[:, :28].reshape(-1, 14, 2)
        mask = matrix[:, 28:56].reshape(-1, 14, 2)
        values[rows, :, 1], mask[rows, :, 1] = np.nan, 0
        from mindwatch.research.neural import past_only_delta
        matrix[:, 56:84] = past_only_delta(mask).reshape(-1, 28)
        fitted = fit_base_estimator(data, "B", family, {"hidden_dim": 8, "epochs": 1, "dropout": 0}, rows, config)
        assert fitted.metadata["seeds"] == [42, 2026, 2027]
        assert len(fitted.estimators) == 3
        held = np.asarray([160, 161, 165])
        expected = np.mean([member.predict_proba(matrix[held])[:, 1] for member in fitted.estimators], axis=0)
        before = predict_base(fitted, data, "B", held)
        np.testing.assert_allclose(before, expected)
        values[held, :, 1] = 1e6
        mask[held, :, 1] = 1
        matrix[:, 56:84] = past_only_delta(mask).reshape(-1, 28)
        np.testing.assert_array_equal(before, predict_base(fitted, data, "B", held))
        restored = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_array_equal(before, predict_base(restored, data, "B", held))
    finally:
        torch.set_num_threads(previous)


def test_regression_catboost_exact_family_and_budget():
    pytest.importorskip("catboost")
    data, fit, _, config = fixture()
    fitted = fit_base_estimator(data, "A", "catboost", {"depth": 3, "l2_leaf_reg": 10}, fit, config,
                                 target_column="future_phq9", task="regression")
    model = fitted.estimators[0].named_steps["model"]
    assert model.get_params()["iterations"] == 300
    assert model.get_params()["loss_function"] == "MAE"
    assert fitted.metadata["actual_model"] == "catboost"
    assert np.isfinite(predict_base(fitted, data, "A", np.asarray([160]), task="regression")).all()


def test_canonical_seed_control_override_and_explicit_call_override_are_honored(monkeypatch):
    import mindwatch.research.temporal_features as temporal
    class SeedClassifier:
        def __init__(self, seed, **kwargs):
            self.seed = seed
        def fit(self, X, y, sample_weight=None, groups=None):
            self.classes_ = np.array([0, 1])
            return self
        def predict_proba(self, X):
            p = np.full(len(X), self.seed / 100.0)
            return np.column_stack((1-p, p))
    monkeypatch.setattr(temporal, "TemporalFusionClassifier", SeedClassifier)
    data, fit, _, config = fixture()
    rows = fit[branch_available(data.frame, "B")[fit]]
    config["seeds"] = [9, 10]  # Overrides the older neural_seeds alias in fixture.
    fitted = fit_base_estimator(data, "B", "grud", {}, rows, config)
    assert fitted.metadata["seeds"] == [9, 10]
    np.testing.assert_allclose(predict_base(fitted, data, "B", np.array([160])), [.095])
    explicit = fit_base_estimator(data, "B", "grud", {}, rows, config, seeds=(77,))
    assert explicit.metadata["seeds"] == [77]
    np.testing.assert_allclose(predict_base(explicit, data, "B", np.array([160])), [.77])


def test_versions_bind_endpoint_fit_order_and_calibration_and_survive_reload():
    data, fit, calibration, config = fixture()
    data.provenance = {"data_manifest_sha256": "synthetic_dataset_digest"}
    data.frame["y_gad7"] = data.frame.y_phq9.copy()
    first = fit_base_estimator(data, "A", "elasticnet", {"C": .1}, fit, config)
    same = fit_base_estimator(data, "A", "elasticnet", {"C": .1}, fit, config)
    different_target = fit_base_estimator(data, "A", "elasticnet", {"C": .1}, fit, config, target_column="y_gad7")
    different_rows = fit_base_estimator(data, "A", "elasticnet", {"C": .1}, fit[:-2], config)
    different_order = fit_base_estimator(data, "A", "elasticnet", {"C": .1}, fit[::-1], config)
    assert first.metadata["model_version"] == same.metadata["model_version"]
    assert len({f.metadata["model_version"] for f in [first, different_target, different_rows, different_order]}) == 4
    assert pickle.loads(pickle.dumps(first)).metadata["model_version"] == first.metadata["model_version"]
    bundle = fit_monitoring_bundle(data, fit, calibration, selection(), config)
    other_calibration = fit_monitoring_bundle(data, fit, calibration[:-2], selection(), config)
    assert bundle.metadata["model_version"] != other_calibration.metadata["model_version"]
    prediction = bundle.predict(data, np.array([160, 161]))
    other = other_calibration.predict(data, np.array([160, 161]))
    assert (prediction.model_version != other.model_version).all()
    assert ":D:elasticnet:" in prediction.model_version.iloc[0]
    assert ":B:elasticnet:" in prediction.model_version.iloc[1]
    assert ":bundle-" in prediction.model_version.iloc[0]
    restored = pickle.loads(pickle.dumps(bundle))
    pd.testing.assert_frame_equal(prediction, restored.predict(data, np.array([160, 161])))
    assert not any("synthetic_sample_" in version for version in prediction.model_version)
