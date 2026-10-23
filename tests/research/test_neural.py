import os
import pytest
pytest.importorskip("torch")
"""Synthetic software checks. They establish no clinical performance."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from mindwatch.research.neural import (
    GRUDEncoder, GatedMultimodalModel, MaskAwareTCN, NeuralBatch, NeuralConfig,
    NeuralTargets, NeuralValidation, SensorNeuralClassifier, evaluate_neural_loss,
    external_model_status, fit_neural, flatten_sequence, masked_multitask_loss,
    participant_weights, past_only_delta, predict_neural, save_private_checkpoint,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sensor_batch(n=12, t=5, f=2, seed=12):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, t, f)).astype(np.float32)
    mask = (rng.uniform(size=values.shape) > 0.2).astype(np.float32)
    mask[:, 0] = 1
    values[mask == 0] = np.nan
    return NeuralBatch(values, mask, past_only_delta(mask))


def tensor_batch(batch):
    return batch.tensors(torch.device("cpu"))


@pytest.mark.parametrize("encoder_class", [GRUDEncoder, MaskAwareTCN])
def test_sensor_gradients_masking_and_causality(encoder_class):
    torch.manual_seed(4)
    batch = sensor_batch()
    encoder = encoder_class(2, 8)
    encoder.set_training_statistics(batch.values, batch.mask)
    encoder.eval()
    tb = tensor_batch(batch)
    values = tb["values"].clone().requires_grad_(True)
    sequence = encoder(values, tb["mask"], tb["delta"], return_sequence=True)
    changed = values.detach().clone()
    changed[:, 3:] = torch.nan_to_num(changed[:, 3:]) + 100
    changed_sequence = encoder(changed, tb["mask"], tb["delta"], return_sequence=True)
    torch.testing.assert_close(sequence[:, :3], changed_sequence[:, :3])
    missing_changed = values.detach().clone()
    missing_changed[tb["mask"] == 0] = 1e8
    torch.testing.assert_close(sequence, encoder(missing_changed, tb["mask"], tb["delta"],
                                                 return_sequence=True))
    sequence.square().mean().backward()
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad[tb["mask"] == 0]) == 0
    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_delta_uses_only_previous_masks_and_no_future_interpolation():
    m = np.array([[[1], [0], [0], [1], [0]]], dtype=np.float32)
    d = past_only_delta(m)
    np.testing.assert_array_equal(d.ravel(), [0, 1, 2, 3, 1])
    changed = m.copy()
    changed[:, 3:] = 0
    np.testing.assert_array_equal(d[:, :4], past_only_delta(changed)[:, :4])


def test_train_statistics_are_masked_and_never_fit_validation():
    batch = NeuralBatch(np.array([[[1.0, np.nan]], [[3.0, np.nan]]], np.float32),
                        np.array([[[1.0, 0.0]], [[1.0, 0.0]]], np.float32),
                        np.zeros((2, 1, 2), np.float32))
    validation = NeuralBatch(np.array([[[10000.0, 2.0]], [[20000.0, 3.0]]], np.float32),
                             np.ones((2, 1, 2), np.float32), np.zeros((2, 1, 2), np.float32))
    fit = fit_neural(batch, NeuralTargets(phq_binary=np.array([0, 1])),
                     NeuralConfig(2, hidden_dim=4, epochs=2, modality_dropout=0,
                                  dropout=0, device="cpu"), groups=np.array(["a", "b"]),
                     validation=NeuralValidation(validation, NeuralTargets(phq_binary=np.array([0, 1])),
                                                 np.array(["c", "d"])))
    np.testing.assert_allclose(fit.model.sensor.feature_mean.numpy(), [2, 0])
    np.testing.assert_array_equal(fit.model.sensor.feature_observed.numpy(), [True, False])
    np.testing.assert_allclose(fit.model.sensor.feature_scale.numpy(), [1, 1])
    assert fit.used_validation


@pytest.mark.parametrize("encoder_class", [GRUDEncoder, MaskAwareTCN])
def test_unseen_training_channel_cannot_change_observed_channel_hidden_states(encoder_class):
    torch.manual_seed(17)
    # Channel 1 has never been observed in the fit partition. Channel 0 remains
    # observed and unchanged, so its predictions must ignore all held-out changes
    # to the unseen channel, including its elapsed-time/collection pattern.
    values = np.stack((np.arange(14), np.full(14, np.nan)), axis=-1)[None].astype(np.float32)
    mask = np.isfinite(values).astype(np.float32)
    encoder = encoder_class(2, 8)
    encoder.set_training_statistics(values, mask)
    encoder.eval()
    baseline = tensor_batch(NeuralBatch(values, mask, past_only_delta(mask)))
    changed_values = values.copy()
    changed_values[:, :, 1] = np.arange(14) * 1000.0
    changed_mask = mask.copy()
    changed_mask[:, :, 1] = 1
    changed = tensor_batch(NeuralBatch(changed_values, changed_mask, past_only_delta(changed_mask)))
    expected = encoder(baseline["values"], baseline["mask"], baseline["delta"], return_sequence=True)
    actual = encoder(changed["values"], changed["mask"], changed["delta"], return_sequence=True)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    # Normal elapsed times for an observed training channel must remain intact.
    _, _, prepared_delta = encoder.prepare(changed["values"], changed["mask"], changed["delta"])
    torch.testing.assert_close(prepared_delta[:, :, 0], changed["delta"][:, :, 0], rtol=0, atol=0)


def test_official_mtm_unseen_channel_changes_are_ignored_after_shared_prepare():
    from pathlib import Path
    pytest.importorskip("einops")
    pytest.importorskip("timm")
    from mindwatch.research.mtm_adapter import _OfficialMTMNetwork
    source = Path(os.environ.get("MINDWATCH_MTM_SOURCE", str(Path(__file__).resolve().parents[1] / ".cache" / "mtm")))
    if not (source / "mtm" / "mtm.py").is_file():
        pytest.skip("Pinned public MTM source has not been explicitly prepared")
    torch.manual_seed(17)
    values = np.stack((np.arange(14), np.full(14, np.nan)), axis=-1)[None].astype(np.float32)
    mask = np.isfinite(values).astype(np.float32)
    model = _OfficialMTMNetwork(2, 8, source, ratios=(2, 2), dropout=0)
    model.set_training_statistics(values, mask)
    model.eval()
    baseline = tensor_batch(NeuralBatch(values, mask, past_only_delta(mask)))
    changed_values = values.copy()
    changed_values[:, :, 1] = np.arange(14) * 1000.0
    changed_mask = mask.copy()
    changed_mask[:, :, 1] = 1
    changed = tensor_batch(NeuralBatch(changed_values, changed_mask, past_only_delta(changed_mask)))
    with torch.no_grad():
        expected = model(baseline["values"], baseline["mask"], baseline["delta"])
        actual = model(changed["values"], changed["mask"], changed["delta"])
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_grud_input_decay_changes_missing_history_and_receives_gradient():
    batch = NeuralBatch(np.array([[[4.0], [np.nan], [np.nan]]], np.float32),
                        np.array([[[1.0], [0], [0]]], np.float32),
                        np.array([[[0.0], [1], [2]]], np.float32))
    encoder = GRUDEncoder(1, 4)
    encoder.set_training_statistics(np.array([[[0.0]], [[4.0]]]), np.ones((2, 1, 1)))
    tb = tensor_batch(batch)
    result = encoder(tb["values"], tb["mask"], tb["delta"])
    result.square().sum().backward()
    assert encoder.decay_x_weight.grad.abs().sum() > 0
    assert encoder.decay_h.weight.grad.abs().sum() > 0
    assert encoder.cell.weight_ih.grad.abs().sum() > 0


def test_fusion_joint_absence_unavailable_embeddings_and_abstention():
    torch.manual_seed(1)
    batch = sensor_batch(n=4)
    batch.mask[1:] = 0
    batch.values[1:] = np.nan
    batch.audio = np.random.default_rng(1).normal(size=(4, 6)).astype(np.float32)
    batch.text = np.random.default_rng(2).normal(size=(4, 7)).astype(np.float32)
    batch.audio_available = np.array([0, 1, 1, 0])
    batch.text_available = np.array([0, 1, 1, 0])
    batch.recording_available = np.array([1, 1, 0, 0])
    batch.demographics = np.zeros((4, 2), np.float32)
    original_availability = batch.audio_available.copy()
    model = GatedMultimodalModel(NeuralConfig(2, audio_dim=6, text_dim=7, demo_dim=2,
                                              hidden_dim=8, multitask=True, dropout=0))
    model.sensor.set_training_statistics(batch.values, batch.mask)
    # Low-level masking test with explicitly activated heads; this is not a fit.
    model.supervised_heads.fill_(True)
    model.eval()
    pred = predict_neural(model, batch)
    np.testing.assert_array_equal(pred["abstain"], [False, False, True, True])
    np.testing.assert_array_equal(pred["availability"], [[1, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert np.isfinite(pred["phq9_probability"][:2]).all()
    assert np.isnan(pred["phq9_probability"][2:]).all()
    assert np.isnan(pred["gad7_score"][2:]).all()
    np.testing.assert_allclose(pred["gates"].sum(axis=1), [1, 1, 0, 0], atol=1e-6)
    batch.audio[[0, 2, 3]] = 1e6
    batch.text[[0, 2, 3]] = -1e6
    # Recording absence overrides a stale available flag and NaN cached vectors.
    batch.audio[2] = np.nan
    batch.text[2] = np.nan
    pred_changed = predict_neural(model, batch)
    np.testing.assert_allclose(pred["phq9_probability"], pred_changed["phq9_probability"], equal_nan=True)
    np.testing.assert_array_equal(batch.audio_available, original_availability)
    for name, limit in [("phq9", 27), ("gad7", 21), ("isi", 28)]:
        assert ((pred[name + "_score"][:2] >= 0) & (pred[name + "_score"][:2] <= limit)).all()


def test_fusion_dropout_never_loses_all_available_sources_or_splits_recording():
    batch = sensor_batch(n=24)
    batch.audio = np.ones((24, 4), np.float32)
    batch.text = np.ones((24, 5), np.float32)
    model = GatedMultimodalModel(NeuralConfig(2, audio_dim=4, text_dim=5,
                                              hidden_dim=8, modality_dropout=0.8))
    model.sensor.set_training_statistics(batch.values, batch.mask)
    model.train()
    output = model(tensor_batch(batch))
    active = output["gates"] > 0
    assert active.any(dim=1).all()
    torch.testing.assert_close(active[:, 1], active[:, 2])


def test_missing_auxiliary_labels_have_no_gradient_or_loss_and_single_task_has_no_aux_head():
    torch.manual_seed(1)
    batch = sensor_batch(n=4)
    model = GatedMultimodalModel(NeuralConfig(2, hidden_dim=8, multitask=True,
                                              dropout=0, modality_dropout=0))
    model.sensor.set_training_statistics(batch.values, batch.mask)
    model.eval()
    outputs = model(tensor_batch(batch))
    targets = {"phq9": torch.tensor([0.0, 15.0, float("nan"), 20.0]),
               "gad7": torch.full((4,), float("nan")),
               "isi": torch.tensor([float("nan"), 16.0, float("nan"), 0.0])}
    outputs["gad7_logit"].retain_grad()
    outputs["gad7_score"].retain_grad()
    outputs["phq9_logit"].retain_grad()
    outputs["isi_score"].retain_grad()
    loss = masked_multitask_loss(outputs, targets)
    no_gad_loss = masked_multitask_loss(outputs, {k: v for k, v in targets.items() if k != "gad7"})
    torch.testing.assert_close(loss, no_gad_loss)
    loss.backward()
    assert outputs["gad7_logit"].grad.abs().sum() == 0
    assert outputs["gad7_score"].grad.abs().sum() == 0
    assert outputs["phq9_logit"].grad[2] == 0
    assert outputs["isi_score"].grad[0] == 0
    assert outputs["phq9_logit"].grad[[0, 1, 3]].abs().sum() > 0
    single = GatedMultimodalModel(NeuralConfig(2, hidden_dim=8, multitask=False))
    assert not any("gad7" in key or "isi" in key for key in single.heads)


def test_evaluation_loss_is_invariant_to_batch_size_and_uses_participant_weights():
    batch = sensor_batch()
    targets = NeuralTargets(phq9=np.array([1, 15, 8, 4, np.nan, 20, 6, 8, 10, 18, 0, 3]))
    model = GatedMultimodalModel(NeuralConfig(2, hidden_dim=8, multitask=True, dropout=0))
    model.sensor.set_training_statistics(batch.values, batch.mask)
    groups = np.array(["a"] * 6 + ["b"] * 3 + ["c"] * 3)
    weights = participant_weights(groups)
    assert np.isclose(weights[groups == "a"].sum(), 1)
    assert np.isclose(weights[groups == "b"].sum(), 1)
    values = [evaluate_neural_loss(model, batch, targets, weights, batch_size=b) for b in [1, 5, 12]]
    np.testing.assert_allclose(values, [values[0]] * 3, rtol=2e-6)


@pytest.mark.parametrize("architecture", ["grud", "tcn"])
def test_tiny_clean_data_can_be_learned_and_seed_is_reproducible(architecture):
    n = 16
    y = (np.arange(n) % 2).astype(np.float32)
    rng = np.random.default_rng(12)
    x = np.repeat((2 * y - 1)[:, None, None], 5, axis=1)
    x = np.repeat(x, 2, axis=2) + rng.normal(0, 0.05, size=(n, 5, 2))
    mask = np.ones_like(x, dtype=np.float32)
    batch = NeuralBatch(x.astype(np.float32), mask, past_only_delta(mask))
    config = NeuralConfig(2, sensor_architecture=architecture, hidden_dim=8, epochs=50,
                           learning_rate=0.015, dropout=0, modality_dropout=0,
                           weight_decay=0, batch_size=16, seed=3)
    result = fit_neural(batch, NeuralTargets(phq_binary=y), config, groups=np.arange(n))
    assert result.history[-1]["train_loss"] < 0.5 * result.history[0]["train_loss"]
    pred = predict_neural(result.model, batch)["phq9_probability"]
    assert roc_auc_score(y, pred) > 0.95
    config.epochs = 2
    first = fit_neural(batch, NeuralTargets(phq_binary=y), config)
    second = fit_neural(batch, NeuralTargets(phq_binary=y), config)
    np.testing.assert_array_equal(predict_neural(first.model, batch)["phq9_probability"],
                                  predict_neural(second.model, batch)["phq9_probability"])
    assert not first.used_validation and first.best_epoch == 2


def test_validation_requires_disjoint_participants_and_no_implicit_split():
    batch = sensor_batch(n=4)
    targets = NeuralTargets(phq_binary=np.array([0, 1, 0, 1]))
    val = NeuralValidation(batch, targets, np.array(["a", "b", "c", "d"]))
    with pytest.raises(ValueError, match="mandatory"):
        fit_neural(batch, targets, NeuralConfig(2), validation=val)
    with pytest.raises(ValueError, match="overlap"):
        fit_neural(batch, targets, NeuralConfig(2), groups=val.groups, validation=val)


def test_unlabelled_heads_never_emit_initialized_scores():
    batch = sensor_batch(n=4)
    result = fit_neural(batch, NeuralTargets(phq_binary=np.array([0, 1, 0, 1])),
                        NeuralConfig(2, hidden_dim=4, epochs=1, multitask=True))
    pred = predict_neural(result.model, batch)
    assert np.isfinite(pred["phq9_probability"]).all()
    assert np.isnan(pred["phq9_score"]).all()
    assert np.isnan(pred["gad7_probability"]).all()
    assert np.isnan(pred["isi_score"]).all()


def test_sklearn_clone_flatten_wrapper_and_no_data_abstention():
    batch = sensor_batch(n=6)
    X = flatten_sequence(batch.values, batch.mask, batch.delta)
    model = SensorNeuralClassifier(sequence_length=5, hidden_dim=8, epochs=2, seed=7)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    model.fit(X, np.array([0, 1, 0, 1, 0, 1]), groups=np.arange(6))
    assert model.predict_proba(X).shape == (6, 2)
    batch.mask[0] = 0
    batch.values[0] = np.nan
    X = flatten_sequence(batch.values, batch.mask, batch.delta)
    assert np.isnan(model.predict(X)[0])
    assert np.isnan(model.predict_proba(X)[0]).all()


def test_sensor_adapter_demographic_tail_is_explicit_and_preserved():
    batch = sensor_batch(n=6)
    demo = np.column_stack((np.arange(6) % 2, np.ones(6))).astype(np.float32)
    X = flatten_sequence(batch.values, batch.mask, batch.delta, demo)
    model = SensorNeuralClassifier(sequence_length=5, n_demographics=2,
                                   hidden_dim=8, epochs=1)
    unpacked = model._unpack(X)
    np.testing.assert_array_equal(unpacked.demographics, demo)
    model.fit(X, np.array([0, 1, 0, 1, 0, 1]), groups=np.arange(6))
    assert model.model_.config.demo_dim == 2
    assert np.isfinite(model.predict_proba(X)).all()
    unpacked.mask[0] = 0
    unpacked.values[0] = np.nan
    absent = flatten_sequence(unpacked.values, unpacked.mask, unpacked.delta, demo)
    assert np.isnan(model.predict_proba(absent)[0]).all()


def test_validation_errors_and_no_fake_external_model():
    batch = sensor_batch(n=4)
    batch.mask[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        batch.validate(2)
    with pytest.raises(ValueError, match="contradicts"):
        NeuralTargets(phq9=np.array([12]), phq_binary=np.array([0])).validate(1)
    with pytest.raises(ValueError, match="scale"):
        NeuralTargets(isi=np.array([29])).validate(1)
    with pytest.raises(ValueError, match="MTM"):
        NeuralConfig(2, sensor_architecture="mtm").validate()
    pat = external_model_status("pat")
    assert not pat["implemented"] and not pat["input_and_source_prerequisites_met"]
    assert "minute_resolution_actigraphy_not_confirmed" in pat["reasons"]
    assert external_model_status("mtm")["implemented"]
    assert not external_model_status("mtm")["input_and_source_prerequisites_met"]


def test_checkpoint_is_private_and_reloads_without_patient_arrays(tmp_path):
    batch = sensor_batch(n=4)
    result = fit_neural(batch, NeuralTargets(phq_binary=np.array([0, 1, 0, 1])),
                        NeuralConfig(2, hidden_dim=4, epochs=1))
    path = tmp_path / "private" / "model.pt"
    save_private_checkpoint(result, path)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    saved = torch.load(path, weights_only=True)
    assert set(saved) == {"config", "state_dict", "history", "best_epoch", "seed", "used_validation"}
    clone_model = GatedMultimodalModel(NeuralConfig(**saved["config"]))
    clone_model.load_state_dict(saved["state_dict"])
    np.testing.assert_allclose(predict_neural(result.model, batch)["phq9_probability"],
                               predict_neural(clone_model, batch)["phq9_probability"])
