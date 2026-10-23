"""Synthetic stacking execution tests; no participant research records are read."""
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.stacking import StackingConfig, finalize_stacking, run_stacking_development


def synthetic():
    rng = np.random.default_rng(99)
    subject = np.repeat(np.arange(100), 2)
    y = np.repeat(np.arange(100) % 2, 2)
    frame = pd.DataFrame({
        "sample_id": [f"synthetic-v-{i}" for i in range(len(y))],
        "subject_id": [f"synthetic-p-{i}" for i in subject],
        "index_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(len(y)), unit="D"),
        "partition": np.where(subject < 80, "development", "temporal_test"),
        "y_phq9": y.astype(float),
        "sensor__signal": y + rng.normal(size=len(y)),
        "audio__signal": y + rng.normal(size=len(y)),
        "sensor__coverage": 1.0,
        "quality__sensor_eligible": (subject % 11 != 0).astype(int),
        "demo__group": (subject % 3 == 0).astype(float),
        # Synthetic canaries only, used to verify each base prediction's subject
        # was absent from its own training fit. Never use real identity features.
        "sensor__canary": subject.astype(float),
        "audio__canary": subject.astype(float),
    })
    frame.loc[subject % 5 == 0, ["sensor__signal", "sensor__canary"]] = np.nan
    frame.loc[subject % 7 == 0, ["audio__signal", "audio__canary"]] = np.nan
    columns = {"sensor": ["sensor__canary", "sensor__signal", "sensor__coverage", "demo__group"],
               "audio": ["audio__canary", "audio__signal"]}
    return frame, columns


def cfg():
    return StackingConfig(outer_folds=3, inner_folds=2, calibration_fraction=.25, bootstrap_replicates=25)


def test_executes_nested_oof_and_never_reads_temporal_values(tmp_path):
    frame, columns = synthetic()
    report = run_stacking_development(frame, columns, "y_phq9", tmp_path / "a", cfg())
    assert report["status"] == "stacking_selection_frozen"
    assert report["temporal_test_evaluated"] is False
    assert report["analysis_type"] == "exploratory"
    altered = frame.copy()
    temporal = altered.partition.eq("temporal_test")
    altered.loc[temporal, "y_phq9"] = 98765.0
    altered.loc[temporal, ["sensor__signal", "audio__signal"]] = np.inf
    second = run_stacking_development(altered, columns, "y_phq9", tmp_path / "b", cfg())
    assert report == second
    outer = pd.read_parquet(tmp_path / "a/stacking_outer_manifest.parquet")
    inner = pd.read_parquet(tmp_path / "a/stacking_inner_manifest.parquet")
    proof = pd.read_parquet(tmp_path / "a/stacking_inner_oof_predictions.parquet")
    for fold, rows in outer.groupby("outer_fold"):
        assert rows.groupby("subject_id").role.nunique().max() == 1
        fit_subjects = set(rows.loc[rows.role.eq("fit"), "subject_id"])
        for _, part in inner.loc[inner.outer_fold.eq(fold)].groupby("inner_fold"):
            assert set(part.subject_id).issubset(fit_subjects)
            assert part.groupby("subject_id").role.nunique().max() == 1
        assert set(proof.loc[proof.outer_fold.eq(fold), "subject_id"]).issubset(fit_subjects)
    oof = pd.read_parquet(tmp_path / "a/stacking_oof_predictions.parquet")
    assert set(oof.subject_id).isdisjoint(set(frame.loc[temporal, "subject_id"]))
    assert (tmp_path / "a").stat().st_mode & 0o777 == 0o700
    assert all(file.stat().st_mode & 0o777 == 0o600 for file in (tmp_path / "a").iterdir())
    with pytest.raises(FileExistsError):
        run_stacking_development(frame, columns, "y_phq9", tmp_path / "a", cfg())


class ParticipantGuard:
    def fit(self, X, y, sample_weight=None, groups=None):
        self.seen = set(X[:, 0])
        self.classes_ = np.array([0, 1])
        # Each observed participant contributes one total unit of weight.
        for subject in np.unique(groups):
            assert sample_weight[groups == subject].sum() == pytest.approx(1)
        return self
    def predict_proba(self, X):
        assert not self.seen.intersection(X[:, 0])
        p = 0.2 + 0.6 / (1 + np.exp(-X[:, 1]))
        return np.column_stack([1-p, p])


def test_base_fit_never_contains_prediction_participant(monkeypatch, tmp_path):
    import mindwatch.research.stacking as stacking
    monkeypatch.setattr(stacking, "_estimator", lambda *args, **kwargs: ParticipantGuard())
    frame, columns = synthetic()
    result = run_stacking_development(frame, columns, "y_phq9", tmp_path, cfg())
    assert result["status"] == "stacking_selection_frozen"


def test_missing_modalities_full_coverage_and_paired_comparison_are_distinct(tmp_path):
    frame, columns = synthetic()
    report = run_stacking_development(frame, columns, "y_phq9", tmp_path, cfg())
    assert report["status"] == "stacking_selection_frozen"
    metrics = {record["model"]: record for record in report["full_cohort_models"]}
    assert metrics["stacked"]["n_labelled_visits"] == metrics["base_sensor"]["n_labelled_visits"] == 160
    assert metrics["stacked"]["coverage"] > metrics["base_sensor"]["coverage"]
    assert metrics["stacked"]["coverage"] < 1  # No signal from either branch.
    common = report["paired_common_models"]
    assert common[0]["n_predicted_visits"] == common[1]["n_predicted_visits"] == report["n_common_predicted_visits"]
    assert all(record["coverage"] == 1 for record in common)
    assert report["paired_comparisons"]["auroc"]["n_bootstrap"] == 25


def test_finalize_requires_frozen_selection_and_never_refits(tmp_path):
    frame, columns = synthetic()
    with pytest.raises(ValueError):
        finalize_stacking(frame, tmp_path / "missing")
    out = tmp_path / "stack"
    report = run_stacking_development(frame, columns, "y_phq9", out, cfg())
    assert report["status"] == "stacking_selection_frozen"
    selection_before = (out / "stacking_selection.json").read_bytes()
    model_before = (out / "frozen_stacking_model.joblib").read_bytes()
    result = finalize_stacking(frame, out)
    assert result["status"] == "stacking_finalized"
    assert result["temporal_test_evaluated"] is True
    metrics = {row["model"]: row for row in result["full_cohort_models"]}
    assert metrics["stacked"]["n_labelled_visits"] == 40
    assert metrics["stacked"]["coverage"] > metrics["base_sensor"]["coverage"]
    assert selection_before == (out / "stacking_selection.json").read_bytes()
    assert model_before == (out / "frozen_stacking_model.joblib").read_bytes()
    selection = json.loads(selection_before)
    predictions = pd.read_parquet(out / "stacking_temporal_predictions.parquet")
    assert predictions.loc[predictions.model.eq("stacked"), "threshold"].eq(selection["operating_points"]["stacked"]["threshold"]).all()
    with pytest.raises(FileExistsError):
        finalize_stacking(frame, out)


def test_changed_development_and_checkpoint_are_rejected(tmp_path):
    frame, columns = synthetic()
    out = tmp_path / "stack"
    assert run_stacking_development(frame, columns, "y_phq9", out, cfg())["status"] == "stacking_selection_frozen"
    changed = frame.copy()
    changed.loc[2, "audio__signal"] += 9
    with pytest.raises(ValueError):
        finalize_stacking(changed, out)
    changed = frame.copy()
    changed.loc[2, "quality__sensor_eligible"] = 0
    with pytest.raises(ValueError):
        finalize_stacking(changed, out)
    path = out / "frozen_stacking_model.joblib"
    with path.open("ab") as handle:
        handle.write(b"synthetic-corruption")
    with pytest.raises(ValueError):
        finalize_stacking(frame, out)


def test_insufficient_meta_support_is_explicit_and_does_not_freeze(tmp_path):
    frame, columns = synthetic()
    for names in columns.values():
        for name in names:
            if name.startswith(("sensor__", "audio__")) and not name.endswith("coverage"):
                frame[name] = np.nan
    result = run_stacking_development(frame, columns, "y_phq9", tmp_path, cfg())
    assert result["status"] == "stacking_unavailable"
    assert result["reason"] == "insufficient_meta_oof_class_support"
    assert not (tmp_path / "stacking_selection.json").exists()


def test_complete_case_mode_uses_identical_training_and_evaluation_rows(tmp_path):
    frame, columns = synthetic()
    config = cfg()
    config.cohort_mode = "complete_case"
    result = run_stacking_development(frame, columns, "y_phq9", tmp_path, config)
    assert result["status"] == "stacking_selection_frozen"
    assert result["training_cohort_policy"] == "identical_complete_case_fit_and_calibration_rows"
    reports = result["full_cohort_models"]
    assert reports[0]["n_predicted_visits"] == reports[1]["n_predicted_visits"]
    assert reports[0]["coverage"] == reports[1]["coverage"] < 1
    proof = pd.read_parquet(tmp_path / "stacking_inner_oof_predictions.parquet")
    assert proof[["base_probability__sensor", "base_probability__audio"]].notna().all().all()


def test_one_unavailable_modality_does_not_silently_become_single_model(tmp_path):
    frame, columns = synthetic()
    frame[["audio__canary", "audio__signal"]] = np.nan
    result = run_stacking_development(frame, columns, "y_phq9", tmp_path, cfg())
    assert result["status"] == "stacking_unavailable"
    assert result["reason"] == "fewer_than_two_trainable_modalities"
    assert not (tmp_path / "stacking_selection.json").exists()
