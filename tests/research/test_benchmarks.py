import pytest
pytest.importorskip("torch")
"""Synthetic grouped-validation and frozen-test tests, not biomedical evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mindwatch.research.benchmarks import (
    BenchmarkConfig, OOFStackingClassifier, adaptive_group_folds, finalize,
    model_status, one_se_selection, participant_calibration_split, run_development,
    _fit,
)


def synthetic_frame():
    rng = np.random.default_rng(5)
    n = 100
    subjects = np.repeat(np.arange(n), 2)
    target = np.repeat(np.arange(n) % 2, 2)
    frame = pd.DataFrame({
        "sample_id": [f"sample-{i}" for i in range(len(subjects))],
        "subject_id": [f"subject-{i}" for i in subjects],
        "index_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(len(subjects)), unit="D"),
        "partition": np.where(subjects < 80, "development", "temporal_test"),
        "y_phq9": target,
        "sensor__activity": target + rng.normal(0, 1, len(subjects)),
        "sensor__sleep": rng.normal(0, 1, len(subjects)),
        "demo__age_group_1": (subjects % 3 == 0).astype(float),
    })
    frame.loc[frame.index[::9], "sensor__sleep"] = np.nan
    return frame


def small_config():
    return BenchmarkConfig(models=("prevalence", "elasticnet"), outer_folds=3, inner_folds=2, calibration_fraction=0.25, bootstrap_replicates=30, parameter_grids={"elasticnet": [{"C": 0.5, "l1_ratio": 0.0}]})


def test_adaptive_folds_and_calibration_never_share_participants():
    subjects = np.repeat(np.arange(12), 3)
    y = np.repeat(np.arange(12) % 2, 3)
    splits = adaptive_group_folds(y, subjects, n_splits=8, seed=42)
    assert 2 <= len(splits) <= 6
    for train, valid in splits:
        assert not set(subjects[train]) & set(subjects[valid])
        assert set(y[train]) == set(y[valid]) == {0, 1}
    fit, calibration = participant_calibration_split(y, subjects, fraction=0.25)
    assert not set(subjects[fit]) & set(subjects[calibration])
    assert adaptive_group_folds(np.zeros(8), np.arange(8)) == []


def test_one_se_prefers_a_simple_competitive_model():
    result = one_se_selection([
        {"model": "elasticnet", "status": "ok", "fold_aurocs": [0.79, 0.80, 0.81]},
        {"model": "catboost", "status": "ok", "fold_aurocs": [0.90, 0.75, 0.81]},
        {"model": "ebm", "status": "incomplete_cv", "fold_aurocs": [0.99]},
    ])
    assert result["selected_model"] == "elasticnet"


def test_development_does_not_depend_on_temporal_values_and_freezes(tmp_path):
    frame = synthetic_frame()
    features = ["sensor__activity", "sensor__sleep", "demo__age_group_1"]
    report = run_development(frame, features, "y_phq9", tmp_path / "first", small_config())
    assert report["status"] == "selection_frozen"
    assert report["analysis_type"] == "exploratory"
    assert report["temporal_test_evaluated"] is False
    changed = frame.copy()
    mask = changed["partition"].eq("temporal_test")
    changed.loc[mask, "y_phq9"] = 98765  # Invalid if development reads these values.
    changed.loc[mask, features] = np.inf
    second = run_development(changed, features, "y_phq9", tmp_path / "second", small_config())
    assert report == second
    selection = json.loads((tmp_path / "first/selection.json").read_text())
    assert selection["sealed_selection"]
    assert selection["operating_point"]["threshold"] is not None
    oof = pd.read_parquet(tmp_path / "first/oof_predictions.parquet")
    assert set(oof.subject_id).isdisjoint(set(frame.loc[mask, "subject_id"]))
    manifest = pd.read_parquet(tmp_path / "first/development_cv_manifest.parquet")
    for _, fold in manifest.groupby("outer_fold"):
        assert fold.groupby("subject_id")["role"].nunique().max() == 1
    assert (tmp_path / "first").stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (tmp_path / "first").iterdir())
    with pytest.raises(FileExistsError):
        run_development(frame, features, "y_phq9", tmp_path / "first", small_config())


def test_finalize_requires_selection_and_uses_frozen_threshold(tmp_path):
    frame = synthetic_frame()
    with pytest.raises(ValueError):
        finalize(frame, tmp_path)
    features = ["sensor__activity", "sensor__sleep", "demo__age_group_1"]
    run_development(frame, features, "y_phq9", tmp_path, small_config())
    selection_before = (tmp_path / "selection.json").read_bytes()
    checkpoint_before = (tmp_path / "frozen_development_model.joblib").read_bytes()
    # Two test visits have no measured modality; demographic availability cannot
    # turn those into predictions in the main multimodal protocol.
    test_index = frame.index[frame.partition.eq("temporal_test")][:2]
    frame.loc[test_index, ["sensor__activity", "sensor__sleep"]] = np.nan
    result = finalize(frame, tmp_path)
    predictions = pd.read_parquet(tmp_path / "temporal_predictions.parquet")
    assert predictions.probability.isna().sum() == 2
    assert result["metrics"]["coverage"] < 1
    assert result["threshold"] == json.loads(selection_before)["operating_point"]["threshold"]
    assert (tmp_path / "selection.json").read_bytes() == selection_before
    assert (tmp_path / "frozen_development_model.joblib").read_bytes() == checkpoint_before
    with pytest.raises(FileExistsError):
        finalize(frame, tmp_path)


def test_finalize_rejects_changed_development_and_overlap(tmp_path):
    frame = synthetic_frame()
    features = ["sensor__activity", "sensor__sleep"]
    run_development(frame, features, "y_phq9", tmp_path, small_config())
    changed = frame.copy()
    changed.loc[0, "sensor__activity"] += 10
    with pytest.raises(ValueError):
        finalize(changed, tmp_path)
    changed = frame.copy()
    changed.loc[0, "partition"] = "temporal_test"
    with pytest.raises(ValueError):
        run_development(changed, features, "y_phq9", tmp_path / "overlap", small_config())


def test_optional_models_never_silently_fallback_and_empty_input_abstains(tmp_path):
    status = model_status("tabpfn", BenchmarkConfig())
    assert status["status"] in {"dependency_unavailable", "local_weights_unavailable", "participant_weighting_unsupported"}
    frame = synthetic_frame()
    frame["sensor__activity"] = np.nan
    result = run_development(frame, ["sensor__activity"], "y_phq9", tmp_path, small_config())
    assert result["status"] == "insufficient_outer_class_support"
    assert not (tmp_path / "selection.json").exists()


def test_zero_coverage_is_not_an_observed_modality(tmp_path):
    frame = synthetic_frame()
    frame["sensor__activity__mean"] = np.nan
    frame["sensor__activity__coverage"] = 0.0
    result = run_development(frame, ["sensor__activity__mean", "sensor__activity__coverage", "demo__age_group_1"], "y_phq9", tmp_path, small_config())
    assert result["n_eligible_visits"] == 0
    assert result["status"] == "insufficient_outer_class_support"


def test_nested_training_only_oof_stacking_utility():
    frame = synthetic_frame().query("partition == 'development'")
    X, y, subjects = frame[["sensor__activity"]].to_numpy(), frame.y_phq9.to_numpy(), frame.subject_id.to_numpy()
    stacker = OOFStackingClassifier({"regularized": LogisticRegression(C=0.1), "weak": LogisticRegression(C=0.01)}, inner_folds=3)
    stacker.fit(X, y, subjects)
    assert stacker.oof_predictions_.shape == (len(y), 2)
    assert np.isfinite(stacker.oof_predictions_).all()
    assert stacker.predict_proba(X[:4]).shape == (4, 2)


def test_fit_forwards_group_identifiers_and_equal_participant_weights():
    class GroupAwareModel:
        def fit(self, X, y, sample_weight=None, groups=None):
            self.groups, self.weights = groups, sample_weight
            return self
    fitted = _fit(GroupAwareModel(), np.ones((3, 1)), [0, 0, 1], np.array(["a", "a", "b"]))
    assert list(fitted.groups) == ["a", "a", "b"]
    assert fitted.weights.tolist() == [0.5, 0.5, 1.0]


def test_sequence_registry_requires_actual_official_source():
    assert model_status("mtm", BenchmarkConfig())["status"] in {"official_source_unavailable", "official_adapter_unavailable", "dependency_unavailable"}


@pytest.mark.parametrize("architecture", ["grud", "tcn"])
def test_neural_through_nested_engine_and_packed_empty_window(tmp_path, architecture):
    pytest.importorskip("torch")
    from mindwatch.research.neural import flatten_sequence
    frame = synthetic_frame()
    rng = np.random.default_rng(20)
    n, t, f = len(frame), 3, 1
    values = rng.normal(size=(n, t, f)).astype(np.float32)
    values += frame.y_phq9.to_numpy()[:, None, None]
    mask, delta = np.ones_like(values), np.ones_like(values)
    values[0], mask[0] = np.nan, 0
    demographic = frame[["demo__age_group_1"]].to_numpy()
    packed = flatten_sequence(values, mask, delta, demographic)
    columns = [f"sensor__seq_{i}" for i in range(packed.shape[1] - 1)] + ["demo__age_group_1"]
    seq = pd.concat([frame[["sample_id", "subject_id", "index_date", "partition", "y_phq9"]], pd.DataFrame(packed, columns=columns)], axis=1)
    config = BenchmarkConfig(models=(architecture,), outer_folds=2, inner_folds=2, calibration_fraction=0.25, sequence_length=t, neural_epochs=1, n_demographics=1, parameter_grids={architecture: [{"hidden_dim": 4, "dropout": 0.0}]})
    result = run_development(seq, columns, "y_phq9", tmp_path, config)
    assert result["status"] == "selection_frozen"
    assert result["n_eligible_visits"] == 159
    assert result["selection"]["selected_model"] == architecture
    assert result["models"][0]["status"] == "ok"


def test_official_mtm_through_nested_engine_when_public_source_is_cached(tmp_path):
    pytest.importorskip("torch")
    from mindwatch.research.neural import flatten_sequence
    source = Path(__file__).resolve().parents[1] / ".cache" / "mtm"
    config = BenchmarkConfig(models=("mtm",), outer_folds=2, inner_folds=2, calibration_fraction=0.25, sequence_length=4, neural_epochs=1, n_demographics=1, mtm_source_dir=str(source), parameter_grids={"mtm": [{"hidden_dim": 8, "dropout": 0.0}]})
    if model_status("mtm", config)["status"] != "available":
        pytest.skip("Pinned official public MTM source/dependencies are not cached.")
    frame = synthetic_frame()
    rng = np.random.default_rng(12)
    values = rng.normal(size=(len(frame), 4, 2)).astype(np.float32)
    mask, delta = np.ones_like(values), np.ones_like(values)
    packed = flatten_sequence(values, mask, delta, frame[["demo__age_group_1"]].to_numpy())
    columns = [f"sensor__seq_{i}" for i in range(packed.shape[1] - 1)] + ["demo__age_group_1"]
    seq = pd.concat([frame[["sample_id", "subject_id", "index_date", "partition", "y_phq9"]], pd.DataFrame(packed, columns=columns)], axis=1)
    report = run_development(seq, columns, "y_phq9", tmp_path, config)
    assert report["status"] == "selection_frozen"
    assert report["models"][0]["provenance"]["revision"]
