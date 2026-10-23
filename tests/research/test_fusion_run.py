import pytest
pytest.importorskip("torch")
"""Synthetic-only integration tests; no patient/metadata files are opened."""
from dataclasses import replace
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from mindwatch.research.fusion_run import (FusionRunConfig, finalize_fusion,
                                predict_fusion_frozen, run_fusion_development)
from mindwatch.research.neural import NeuralBatch, NeuralConfig, past_only_delta


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    rng = np.random.default_rng(42)
    people, repeats, time, features = 48, 2, 5, 2
    subject = np.repeat(np.arange(people), repeats)
    y = (subject % 2).astype(float)
    frame = pd.DataFrame({
        "sample_id": [f"synthetic_sample_{i}" for i in range(len(subject))],
        "subject_id": [f"synthetic_subject_{i}" for i in subject],
        "index_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.tile([0, 120], people), unit="D"),
        "partition": np.where(subject < 40, "development", "temporal_test"),
        "phq9": (2 + 12 * y).astype(object), "gad7": 2 + 11 * y, "isi": 3 + 14 * y,
    })
    frame.loc[frame.partition.eq("temporal_test"), "phq9"] = "SEALED_LABEL_MUST_NOT_BE_USED"
    frame.loc[np.arange(len(frame)) % 9 == 0, ["gad7", "isi"]] = np.nan
    values = rng.normal(0, 0.1, size=(len(subject), time, features)).astype(np.float32)
    values += (2 * y - 1)[:, None, None]
    mask = np.ones_like(values)
    audio = rng.normal(0, 0.1, size=(len(subject), 6)).astype(np.float32)
    text = rng.normal(0, 0.1, size=(len(subject), 7)).astype(np.float32)
    audio += y[:, None]
    text += y[:, None]
    recording = np.ones(len(subject), bool)
    # One labelled development visit and one temporal visit have no modality.
    for index in [3, 80]:
        mask[index] = 0
        values[index] = np.nan
        recording[index] = False
        audio[index] = text[index] = np.nan
    batch = NeuralBatch(values, mask, past_only_delta(mask), audio, text,
                         demographics=np.column_stack((np.arange(len(frame)) % 3, np.ones(len(frame)))).astype(np.float32),
                         recording_available=recording,
                         quality=np.ones((len(frame), 3), np.float32))
    config = FusionRunConfig(
        neural=NeuralConfig(features, audio_dim=6, text_dim=7, demo_dim=2,
                            hidden_dim=8, epochs=1, batch_size=32, dropout=0,
                            modality_dropout=0.5, seed=42),
        outer_folds=2, inner_folds=2, bootstrap_replicates=20,
        calibration_fraction=0.25, speech_gate_verified=True,
        speech_gate_reference="synthetic_fixture_only_not_metadata_evidence",
        protocol_unresolved=("Synthetic software test only",),
    )
    return frame, batch, config


def test_real_nested_training_shared_splits_calibration_and_frozen_inference(tmp_path):
    frame, batch, config = fixture()
    report = run_fusion_development(frame, batch, tmp_path / "fusion", config)
    assert report["status"] == "selection_frozen", report
    assert not report["temporal_test_evaluated"]
    assert {r["model"] for r in report["models"]} == {"phq_single", "multitask"}
    assert all(r["status"] == "ok" for r in report["models"])
    assert report["paired_multitask_minus_single"]["n_bootstrap"] == 20
    assert report["n_eligible_visits"] == 79
    assert all(r["population_oof_metrics"]["coverage"] < 1 for r in report["models"])
    splits = pd.read_parquet(tmp_path / "fusion" / "development_cv_manifest.parquet")
    for _, fold in splits.groupby("outer_fold"):
        roles = {role: set(group.subject_id) for role, group in fold.groupby("role")}
        assert not (roles["fit"] & roles["calibration"] or
                    roles["fit"] & roles["validation"] or
                    roles["calibration"] & roles["validation"])
    predictions = pd.read_parquet(tmp_path / "fusion" / "oof_predictions.parquet")
    a = predictions.loc[predictions.model.eq("phq_single"), ["sample_id", "outer_fold"]]
    b = predictions.loc[predictions.model.eq("multitask"), ["sample_id", "outer_fold"]]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    selection = json.loads((tmp_path / "fusion" / "selection.json").read_text())
    assert selection["sealed_selection"]
    assert selection["selected_model"] in {"phq_single", "multitask"}
    assert selection["config"]["neural"]["epochs"] == 1
    pred = predict_fusion_frozen(batch, tmp_path / "fusion")
    assert np.isnan(pred["phq9_probability"][[3, 80]]).all()
    assert pred["abstain"][[3, 80]].all()
    assert np.isfinite(np.delete(pred["phq9_probability"], [3, 80])).all()
    for path in (tmp_path / "fusion").iterdir():
        assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(tmp_path / "fusion").st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        run_fusion_development(frame, batch, tmp_path / "fusion", config)


def test_temporal_values_and_labels_cannot_change_development_selection(tmp_path):
    frame, batch, config = fixture()
    first = run_fusion_development(frame, batch, tmp_path / "first", config)
    modified = frame.copy()
    modified["gad7"] = modified["gad7"].astype(object)
    modified["isi"] = modified["isi"].astype(object)
    test = modified.partition.eq("temporal_test").to_numpy()
    modified.loc[test, ["phq9", "gad7", "isi"]] = "SEALED_CHANGED"
    changed_batch = batch.subset(np.arange(len(batch)))
    changed_batch.values[test] = 1e7
    changed_batch.audio[test] = -1e7
    changed_batch.text[test] = 1e7
    second = run_fusion_development(modified, changed_batch, tmp_path / "second", config)
    assert first["status"] == second["status"] == "selection_frozen"
    assert first["selection"] == second["selection"]
    assert first["models"] == second["models"]
    one = json.loads((tmp_path / "first" / "selection.json").read_text())
    two = json.loads((tmp_path / "second" / "selection.json").read_text())
    assert one["development_fingerprint"] == two["development_fingerprint"]


def test_finalize_requires_gate_uses_locked_choice_and_is_once_only(tmp_path):
    frame, batch, config = fixture()
    with pytest.raises(ValueError, match="persisted first"):
        finalize_fusion(frame, batch, tmp_path / "fusion")
    run_fusion_development(frame, batch, tmp_path / "fusion", config)
    selection_before = (tmp_path / "fusion" / "selection.json").read_bytes()
    test = frame.partition.eq("temporal_test").to_numpy()
    indices = np.flatnonzero(test)
    subject_ids = np.repeat(np.arange(40, 48), 2)
    frame.loc[test, "phq9"] = 2 + 12 * (subject_ids % 2)
    result = finalize_fusion(frame, batch, tmp_path / "fusion")
    assert result["status"] == "finalized"
    assert result["temporal_test_evaluated"]
    assert result["metrics"]["phq9"]["n_labelled_visits"] == len(indices)
    assert result["metrics"]["phq9"]["n_predicted_visits"] == len(indices) - 1
    assert (tmp_path / "fusion" / "selection.json").read_bytes() == selection_before
    with pytest.raises(FileExistsError, match="already exists"):
        finalize_fusion(frame, batch, tmp_path / "fusion")


def test_review_gate_zero_and_unknown_metadata_never_train(tmp_path, monkeypatch):
    import mindwatch.research.fusion_run as module
    frame, batch, config = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Training should not be invoked when speech metadata is gated")
    monkeypatch.setattr(module, "fit_neural", forbidden)
    report = run_fusion_development(frame, batch, tmp_path / "unverified",
                                     replace(config, speech_gate_verified=False))
    assert report["status"] == "speech_metadata_gate_unverified"
    batch.recording_available[:] = False
    report = run_fusion_development(frame, batch, tmp_path / "zero", config)
    assert report["status"] == "no_reviewed_speech_inputs"
    assert not (tmp_path / "zero" / "selection.json").exists()


def test_frozen_development_change_or_checkpoint_tampering_rejected(tmp_path):
    frame, batch, config = fixture()
    run_fusion_development(frame, batch, tmp_path / "fusion", config)
    changed = frame.copy()
    changed.loc[0, "phq9"] = 27
    with pytest.raises(ValueError, match="development data changed"):
        finalize_fusion(changed, batch, tmp_path / "fusion")
    with (tmp_path / "fusion" / "frozen_development_model.joblib").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        predict_fusion_frozen(batch, tmp_path / "fusion")


def test_participant_partition_overlap_and_missing_recording_flags_are_rejected(tmp_path):
    frame, batch, config = fixture()
    changed = frame.copy()
    changed.loc[80, "subject_id"] = changed.loc[0, "subject_id"]
    with pytest.raises(ValueError, match="Participant overlap"):
        run_fusion_development(changed, batch, tmp_path / "overlap", config)
    batch.recording_available = None
    with pytest.raises(ValueError, match="reviewed recording availability"):
        run_fusion_development(frame, batch, tmp_path / "missing", config)
