import os
"""Synthetic tests for past-only temporal summaries and static-tail fitting."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mindwatch.research.temporal_features import TemporalFusionClassifier, temporal_features


def test_exact_daily_means_change_slope_and_observed_pair_correlation():
    values = np.arange(14, dtype=float)[None, :, None]
    mask = np.ones_like(values)
    signals, quality = temporal_features(values, mask, ["sensor__Steps"])
    prefix = "sensor__temporal_steps__"
    np.testing.assert_allclose(
        signals[[prefix + "mean_1d", prefix + "mean_3d", prefix + "mean_7d", prefix + "mean_14d"]],
        [[13, 12, 10, 6.5]],
    )
    assert signals.loc[0, prefix + "change_7d_vs_previous_7d"] == 7
    assert signals.loc[0, prefix + "slope_14d"] == 1
    assert np.isclose(signals.loc[0, prefix + "lag1_autocorrelation_14d"], 1)
    assert quality.loc[0, "quality__temporal_steps__lag1_observed_pairs_14d"] == 13
    assert all(c.startswith("sensor__temporal_") for c in signals)
    assert all(c.startswith("quality__temporal_") for c in quality)
    assert not set(signals) & set(quality)


def test_calendar_gaps_are_not_interpolated_or_treated_as_adjacent():
    values = np.full((2, 14, 1), np.nan)
    mask = np.zeros_like(values)
    values[0, [0, 13], 0] = [0, 26]
    mask[0, [0, 13], 0] = 1
    values[1, [0, 2, 4, 6], 0] = [1, 2, 3, 4]
    mask[1, [0, 2, 4, 6], 0] = 1
    signals, quality = temporal_features(values, mask, ["steps"])
    prefix = "sensor__temporal_steps__"
    assert signals.loc[0, prefix + "slope_14d"] == 2
    assert signals.loc[0, prefix + "mean_3d"] == 26
    assert signals.loc[0, prefix + "change_7d_vs_previous_7d"] == 26
    assert np.isnan(signals.loc[1, prefix + "mean_1d"])
    assert np.isnan(signals.loc[1, prefix + "mean_7d"])
    assert np.isnan(signals.loc[1, prefix + "change_7d_vs_previous_7d"])
    assert quality.loc[1, "quality__temporal_steps__lag1_observed_pairs_14d"] == 0
    assert np.isnan(signals[prefix + "lag1_autocorrelation_14d"]).all()


def test_future_perturbation_and_older_than_window_values_are_irrelevant():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 40, 2))
    mask = rng.integers(0, 2, size=values.shape).astype(float)
    values[mask == 0] = np.nan
    expected, expected_quality = temporal_features(values, mask, ["steps", "sleep"], end_index=30)
    # These values are outside the [16,30) feature window. Future validation must
    # not inspect them either; the intentionally invalid mask would otherwise fail.
    values[:, :16], values[:, 30:] = -1e8, np.inf
    mask[:, :16], mask[:, 30:] = 55, np.nan
    actual, actual_quality = temporal_features(values, mask, ["steps", "sleep"], end_index=30)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(expected_quality, actual_quality)


def test_all_missing_zero_constant_and_masked_placeholders():
    values = np.array([np.full((14, 1), np.nan), np.zeros((14, 1)), np.full((14, 1), 5.0)])
    mask = np.ones_like(values)
    mask[0] = 0
    signals, quality = temporal_features(values, mask, ["sleep"])
    assert signals.iloc[0].isna().all()
    assert quality.iloc[0].eq(0).all()
    assert signals.loc[1, "sensor__temporal_sleep__mean_1d"] == 0
    assert signals.loc[2, "sensor__temporal_sleep__slope_14d"] == 0
    assert np.isnan(signals["sensor__temporal_sleep__lag1_autocorrelation_14d"]).all()
    values[0] = np.inf
    again, again_quality = temporal_features(values, mask, ["sleep"])
    pd.testing.assert_frame_equal(signals, again)
    pd.testing.assert_frame_equal(quality, again_quality)
    empty, empty_quality = temporal_features(values[:0], mask[:0], ["sleep"])
    assert empty.shape == (0, 7) and empty_quality.shape == (0, 10)


@pytest.mark.parametrize("channels", [["phq9"], ["subject_id"], ["target"], ["patient_id"]])
def test_target_identifier_channels_are_rejected(channels):
    with pytest.raises(ValueError, match="targets and identifiers"):
        temporal_features(np.ones((1, 14, 1)), np.ones((1, 14, 1)), channels)


def test_invalid_observed_data_or_mask_or_window_and_duplicate_names_fail():
    values = np.ones((1, 14, 1))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        temporal_features(values, np.ones_like(values), ["steps"])
    with pytest.raises(ValueError, match="binary"):
        temporal_features(np.ones_like(values), np.full_like(values, 0.5), ["steps"])
    with pytest.raises(ValueError, match="14 past"):
        temporal_features(values[:, :7], np.ones_like(values[:, :7]), ["steps"])
    with pytest.raises(ValueError, match="collide"):
        temporal_features(np.ones((1, 14, 2)), np.ones((1, 14, 2)), ["sleep time", "sleep-time"])


class RecordingClassifier:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def fit(self, X, y, sample_weight=None, groups=None):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_weight = np.asarray(sample_weight).copy()
        self.fit_groups = groups
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.last_X = np.asarray(X).copy()
        return np.tile([0.4, 0.6], (len(X), 1))


def packed_fixture():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(8, 14, 2))
    mask = np.ones_like(values)
    delta = np.ones_like(values)
    seq = np.concatenate([a.reshape(8, -1) for a in [values, mask, delta]], axis=1)
    tail = np.column_stack((np.arange(8, dtype=float), np.full(8, np.nan), np.arange(8) % 2))
    tail[1, 0] = np.nan
    X = np.concatenate((seq, tail), axis=1)
    y = np.arange(8) % 2
    return X, y, seq.shape[1]


def test_fold_only_imputation_scaling_and_groups_weights_forwarding(monkeypatch):
    import mindwatch.research.temporal_features as module
    monkeypatch.setattr(module, "_classifier", lambda *args, **kwargs: RecordingClassifier(**kwargs))
    X, y, sequence_width = packed_fixture()
    names = ("ema__mood_mean", "history__previous_phq9", "demo__sex_code_1")
    groups = np.repeat(["a", "b", "c", "d"], 2)
    weights = np.array([0.5] * 8)
    estimator = TemporalFusionClassifier(n_static_features=3, static_feature_names=names,
                                          hidden_dim=32, epochs=20)
    assert clone(estimator).get_params() == estimator.get_params()
    estimator.fit(X, y, groups=groups, sample_weight=weights)
    preprocessor = estimator.static_preprocessor_
    imputer = preprocessor.named_steps["impute"]
    scaler = preprocessor.named_steps["scale"]
    medians, means, scales = imputer.statistics_.copy(), scaler.mean_.copy(), scaler.scale_.copy()
    assert medians[1] == 0  # Completely missing training column retains its width.
    assert estimator.classifier_.parameters["n_demographics"] == 3
    assert estimator.classifier_.parameters["hidden_dim"] == 32
    assert estimator.classifier_.parameters["epochs"] == 20
    assert estimator.classifier_.fit_groups is groups
    np.testing.assert_array_equal(estimator.classifier_.fit_weight, weights)
    np.testing.assert_array_equal(estimator.classifier_.fit_X[:, :sequence_width], X[:, :sequence_width])
    np.testing.assert_allclose(estimator.classifier_.fit_X[:, sequence_width:].mean(axis=0), 0, atol=1e-10)
    held_out = X.copy()
    held_out[:, -3:] = [[1e6, -1e6, 1e5]]
    estimator.predict_proba(held_out)
    assert np.all(estimator.classifier_.last_X[:, sequence_width + 1] == 0)
    estimator.predict_proba(X)
    np.testing.assert_array_equal(imputer.statistics_, medians)
    np.testing.assert_array_equal(scaler.mean_, means)
    np.testing.assert_array_equal(scaler.scale_, scales)
    np.testing.assert_array_equal(estimator.classifier_.last_X[:, :sequence_width], X[:, :sequence_width])


def test_participant_weights_drive_scaler_and_delegate_when_not_supplied(monkeypatch):
    import mindwatch.research.temporal_features as module
    monkeypatch.setattr(module, "_classifier", lambda *args, **kwargs: RecordingClassifier(**kwargs))
    X, y, _ = packed_fixture()
    X = X[:, :-2]  # Exactly one static feature remains.
    groups = np.array(["a"] * 6 + ["b"] * 2)
    model = TemporalFusionClassifier(n_static_features=1, static_feature_names=("ema__mood",))
    model.fit(X, y, groups=groups)
    weight = model.classifier_.fit_weight
    np.testing.assert_allclose(weight, [1/6] * 6 + [1/2] * 2)
    assert np.isclose(weight[:6].sum(), weight[6:].sum())
    imputed = model.static_preprocessor_.named_steps["impute"].transform(X[:, -1:])
    expected_mean = np.average(imputed, axis=0, weights=weight)
    np.testing.assert_allclose(model.static_preprocessor_.named_steps["scale"].mean_, expected_mean)


def test_zero_weight_rows_cannot_fit_tail_statistics(monkeypatch):
    import mindwatch.research.temporal_features as module
    monkeypatch.setattr(module, "_classifier", lambda *args, **kwargs: RecordingClassifier(**kwargs))
    X, y, _ = packed_fixture()
    X = X[:, :-2]
    X[-1, -1] = 1e9
    weights = np.r_[np.ones(7), 0.0]
    model = TemporalFusionClassifier(n_static_features=1, static_feature_names=("ema__mood",))
    model.fit(X, y, sample_weight=weights)
    statistics = model.static_preprocessor_.named_steps["impute"].statistics_
    assert statistics[0] == np.nanmedian(X[:-1, -1])
    assert model.static_preprocessor_.named_steps["scale"].mean_[0] < 10


@pytest.mark.parametrize("names", [None, ("sample_id",), ("demo__patient_id",),
                                   ("demo__phq9",), ("ema__target",), ("history__phq9",),
                                   ("audio__embedding_0",)])
def test_static_schema_rejects_raw_targets_identifiers_and_speech(names):
    X, y, _ = packed_fixture()
    X = X[:, :-2]
    with pytest.raises(ValueError):
        TemporalFusionClassifier(n_static_features=1, static_feature_names=names).fit(X, y)


@pytest.mark.parametrize("architecture", ["grud", "tcn"])
def test_actual_neural_training_preserves_sensor_abstention_with_static_tail(architecture):
    torch = pytest.importorskip("torch")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        X, y, sequence_width = packed_fixture()
        model = TemporalFusionClassifier(
            n_static_features=3,
            static_feature_names=("ema__mood_mean", "history__previous_phq9", "demo__sex_code_1"),
            architecture=architecture, hidden_dim=8, epochs=2, batch_size=8, dropout=0,
        ).fit(X, y, groups=np.arange(len(X)))
        probability = model.predict_proba(X)
        assert np.isfinite(probability).all()
        width = sequence_width // 3
        changed = X.copy()
        changed[0, :width] = np.nan
        changed[0, width:2*width] = 0
        assert np.isnan(model.predict_proba(changed)[0]).all()
        assert np.isnan(model.predict(changed)[0])
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(model.predict_proba(changed), restored.predict_proba(changed), equal_nan=True)
    finally:
        torch.set_num_threads(old_threads)


def test_mtm_delegates_to_official_adapter_with_static_preprocessing(monkeypatch):
    pytest.importorskip('torch')
    import mindwatch.research.mtm_adapter as mtm
    monkeypatch.setattr(mtm, "OfficialMTMClassifier", RecordingClassifier)
    X, y, _ = packed_fixture()
    model = TemporalFusionClassifier(n_static_features=3,
                                      static_feature_names=("ema__mood", "history__previous_phq9", "demo__sex"),
                                      architecture="mtm", mtm_source_dir="synthetic_public_source_path",
                                      hidden_dim=32, epochs=20, device="cpu")
    model.fit(X, y, groups=np.arange(8))
    assert model.classifier_.parameters["source_dir"] == "synthetic_public_source_path"
    assert model.classifier_.parameters["n_demographics"] == 3
    assert model.classifier_.parameters["hidden_dim"] == 32
    assert model.classifier_.parameters["epochs"] == 20


def test_dataframe_schema_and_feature_order_are_preserved(monkeypatch):
    import mindwatch.research.temporal_features as module
    monkeypatch.setattr(module, "_classifier", lambda *args, **kwargs: RecordingClassifier(**kwargs))
    X, y, sequence_width = packed_fixture()
    names = ("ema__mood", "history__previous_phq9", "demo__sex")
    columns = [f"sensor__seq_{i:04d}" for i in range(sequence_width)] + list(names)
    frame = pd.DataFrame(X, columns=columns)
    model = TemporalFusionClassifier(n_static_features=3, static_feature_names=names).fit(frame, y)
    model.predict_proba(frame)
    changed = frame.copy()
    changed.columns = [columns[1], columns[0], *columns[2:]]
    with pytest.raises(ValueError, match="order changed"):
        model.predict_proba(changed)
    with pytest.raises(ValueError):
        TemporalFusionClassifier(n_static_features=3, static_feature_names=names).fit(
            frame.assign(subject_id=np.arange(8)), y)


def test_actual_official_mtm_sensor_only_adapter_abstains_without_sensor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    pytest.importorskip("timm")
    public_source = Path(os.environ.get("MINDWATCH_MTM_SOURCE", str(Path(__file__).resolve().parents[1] / ".cache" / "mtm")))
    if not (public_source / "mtm" / "mtm.py").is_file():
        pytest.skip("Pinned public MTM code has not been explicitly prepared")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        X, y, sequence_width = packed_fixture()
        X = X[:, :sequence_width]
        model = TemporalFusionClassifier(architecture="mtm", mtm_source_dir=str(public_source),
                                          n_static_features=0, hidden_dim=8, epochs=1,
                                          batch_size=8, dropout=0, device="cpu").fit(X, y, groups=np.arange(8))
        assert model.static_preprocessor_ is None
        assert np.isfinite(model.predict_proba(X)).all()
        assert model.classifier_.provenance_["upstream_source_modified"] is False
        width = sequence_width // 3
        X[0, :width] = np.nan
        X[0, width:2*width] = 0
        assert np.isnan(model.predict_proba(X)[0]).all()
    finally:
        torch.set_num_threads(previous_threads)
