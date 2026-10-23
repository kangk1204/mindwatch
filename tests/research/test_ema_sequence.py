import pytest
pytest.importorskip("torch")
"""Synthetic-only correctness checks; no cohort files or results are opened."""
from copy import deepcopy

import joblib
import numpy as np
import pytest
import torch

from mindwatch.research import ema_sequence as sm


@pytest.fixture(autouse=True)
def small_cpu_test_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(n=12, sensor=True):
    rng = np.random.default_rng(7)
    em = (rng.random((n, 8, 4)) > .25).astype(float)
    ev = rng.uniform(1, 9, em.shape)
    ev[em == 0] = np.nan
    ed = rng.uniform(0, 30, em.shape)
    sv = rng.normal(size=(n, 14, 7))
    mask = (rng.random(sv.shape) > .2).astype(float)
    sv[mask == 0] = np.nan
    history = np.column_stack((rng.uniform(0, 27, n), rng.uniform(1, 120, n)))
    batch = sm.SeqBatch(ev, em, ed, history,
        ("h_initial_phq9", "h_days"), np.full((n, 4), 5.), sv, mask,
        np.broadcast_to(np.arange(14, dtype=float)[None, :, None], sv.shape).copy(),
        np.full(n, sensor, bool))
    y = np.full((n, 4), 5.25)
    groups = np.array([f"synthetic_person_{j // 2}" for j in range(n)])
    return batch, y, groups


def test_zero_head_and_gradients_after_first_update_and_tiny_overfit():
    batch, y, groups = fixture(sensor=False)
    net = sm._CorrectionGRU(4, 2, hidden=16)
    with torch.no_grad():
        output = net(torch.ones(3, 8, 12), torch.ones(3, 8, dtype=torch.bool),
                     torch.ones(3, 4))
    assert torch.count_nonzero(output) == 0
    fitted = sm.fit_sequence(batch, y, groups, epochs=20, learning_rate=.03)
    trace = fitted.traces["he"]
    assert trace["first_updates"][0]["head_gradient_norm"] > 0
    assert trace["first_updates"][0]["recurrent_gradient_norm"] == 0
    assert trace["first_updates"][1]["recurrent_gradient_norm"] > 0
    assert trace["loss"][-1] < .05
    assert np.mean(abs(fitted.predict(batch) - y)) < .05


def test_per_domain_participant_total_weights_and_weighted_median():
    y = np.array([[0., np.nan, 2., 5.], [0., 7., np.nan, 5.], [10., 3., 4., np.nan]])
    groups = np.array(["synthetic_repeat", "synthetic_repeat", "synthetic_single"])
    w = sm._label_weights(y, groups)
    np.testing.assert_allclose(w[:2].sum(0), [1, 1, 1, 1])
    np.testing.assert_allclose(w[2], [1, 1, 1, 0])
    assert sm._weighted_median(y[:, 0], w[:, 0]) == 0
    assert np.isnan(sm._weighted_median(np.array([np.nan]), np.array([0.])))


def test_subset_preserves_event_order_and_history_schema():
    batch, _, _ = fixture()
    got = batch.subset([3, 1])
    assert got.history_names == batch.history_names
    np.testing.assert_array_equal(got.ema_values, batch.ema_values[[3, 1]])
    assert got.persistence.shape == (2, 4)


def test_sensor_missing_returns_exact_he_and_sensor_training_freezes_he():
    batch, y, groups = fixture()
    fitted = sm.fit_sequence(batch, y, groups, epochs=2)
    assert fitted.traces["sensor"]["status"] == "fitted"
    assert all(not p.requires_grad for p in fitted.he_model.parameters())
    changed = deepcopy(batch)
    changed.sensor_available[:] = False
    changed.sensor_values[:] = np.nan
    changed.sensor_mask[:] = 1
    changed.sensor_delta[:] = np.nan
    np.testing.assert_array_equal(fitted.predict(changed, use_sensor=True),
                                  fitted.predict(changed, use_sensor=False))
    other = deepcopy(batch)
    other.sensor_values = np.where(other.sensor_mask, other.sensor_values * 100 + 50, np.nan)
    refit = sm.fit_sequence(other, y, groups, epochs=2)
    np.testing.assert_array_equal(fitted.predict(batch), refit.predict(batch))


def test_absent_fit_channels_block_values_mask_gap_and_new_static_values():
    batch, y, groups = fixture()
    batch.ema_mask[:, :, 3] = 0
    batch.ema_values[:, :, 3] = np.nan
    batch.sensor_mask[:, :, 6] = 0
    batch.sensor_values[:, :, 6] = np.nan
    # A separate optional static field can be unseen in training; required
    # history fields must remain available and released.
    batch.history = np.column_stack((batch.history, np.full(len(batch), np.nan)))
    batch.history_names += ("h_isi",)
    fitted = sm.fit_sequence(batch, y, groups, epochs=2)
    changed = deepcopy(batch)
    changed.ema_values[changed.ema_mask == 0] = -1e6
    changed.sensor_values[changed.sensor_mask == 0] = 1e6
    changed.ema_values[:, :, 3], changed.ema_mask[:, :, 3], changed.ema_delta[:, :, 3] = 999, 1, 1e7
    changed.sensor_values[:, :, 6], changed.sensor_mask[:, :, 6], changed.sensor_delta[:, :, 6] = 999, 1, 1e7
    changed.history[:, 2] = 999
    np.testing.assert_array_equal(fitted.predict(batch, use_sensor=True),
                                  fitted.predict(changed, use_sensor=True))


def test_padding_skips_state_and_predictions_do_not_depend_on_row_order():
    batch, y, groups = fixture()
    fitted = sm.fit_sequence(batch, y, groups, epochs=1)
    baseline = fitted.predict(batch, use_sensor=True)
    idx = np.arange(len(batch))[::-1]
    np.testing.assert_array_equal(baseline[idx], fitted.predict(batch.subset(idx), use_sensor=True))
    net = sm._CorrectionGRU(4, 2, hidden=4)
    with torch.no_grad():
        net.head.weight.fill_(.1)
    x = torch.rand(3, 8, 12)
    token = torch.zeros(3, 8, dtype=torch.bool)
    history = torch.zeros(3, 4)
    a = net(x, token, history)
    b = net(x * 1e5, token, history)
    assert torch.equal(a, b)


def test_train_only_moments_do_not_refit_and_readonly_inputs_are_unchanged():
    batch, y, groups = fixture()
    for key, value in vars(batch).items():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    y.setflags(write=False)
    before = {k: v.tobytes() for k, v in vars(batch).items() if isinstance(v, np.ndarray)}
    fitted = sm.fit_sequence(batch, y, groups, epochs=1)
    stats = deepcopy(fitted.traces["normalization"])
    held = deepcopy(batch)
    held.history = np.full_like(held.history, 1e5)
    held.ema_values = np.where(held.ema_mask, 1e5, np.nan)
    fitted.predict(held, use_sensor=True)
    assert fitted.traces["normalization"] == stats
    assert before == {k: v.tobytes() for k, v in vars(batch).items() if isinstance(v, np.ndarray)}


def test_missing_targets_receive_no_loss_and_untrained_domain_stays_nan():
    batch, y, groups = fixture(sensor=False)
    y[:, 2] = np.nan
    y[::2, 1] = np.nan
    batch.persistence[:, 0] = np.nan
    fitted = sm.fit_sequence(batch, y, groups, epochs=2)
    p = fitted.predict(batch)
    assert np.isfinite(p[:, [0, 1, 3]]).all()
    assert np.isnan(p[:, 2]).all()
    assert fitted.fallback[0] == 5.25
    assert fitted.traces["he"]["supervised_domains"] == [True, True, False, True]
    np.testing.assert_array_equal(fitted.he_model.head.weight.detach().numpy()[2], 0)
    assert fitted.he_model.head.bias.detach().numpy()[2] == 0


def test_exact_allowlist_and_invalid_inputs_rejected():
    batch, y, groups = fixture()
    batch.history_names = ("h_phq9", "h_unreviewed")
    with pytest.raises(ValueError, match="exact approved"):
        sm.fit_sequence(batch, y, groups, epochs=1)
    model = sm.fit_sequence(batch, y, groups, epochs=1,
                            allowed_history_names=batch.history_names)
    assert model.history_names == batch.history_names
    batch.history_names = ("h_phq9", "target_ema_depression")
    with pytest.raises(ValueError, match="Current targets"):
        sm.fit_sequence(batch, y, groups, epochs=1, allowed_history_names=batch.history_names)
    batch, y, groups = fixture()
    with pytest.raises(ValueError, match="budget"):
        sm.fit_sequence(batch, y, groups, epochs=21)
    y[:] = np.nan
    with pytest.raises(ValueError, match="No observed"):
        sm.fit_sequence(batch, y, groups, epochs=1)


def test_target_free_joblib_roundtrip_determinism_and_training_rng_restore(tmp_path):
    batch, y, groups = fixture()
    torch.manual_seed(99)
    state = torch.get_rng_state().clone()
    fitted = sm.fit_sequence(batch, y, groups, epochs=2)
    assert torch.equal(torch.get_rng_state(), state)
    second = sm.fit_sequence(batch, y, groups, epochs=2)
    np.testing.assert_array_equal(fitted.predict(batch, use_sensor=True), second.predict(batch, use_sensor=True))
    path = tmp_path / "synthetic_only.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    del y, groups
    for sensor in (False, True):
        np.testing.assert_array_equal(fitted.predict(batch, use_sensor=sensor),
                                      restored.predict(batch, use_sensor=sensor))


def test_prespecified_ensemble_is_pointwise_three_seed_median():
    batch, y, groups = fixture(sensor=False)
    ensemble = sm.fit_sequence_ensemble(batch, y, groups, epochs=1)
    assert [b.traces["seed"] for b in ensemble.bundles] == [42, 2026, 2027]
    expected = np.median(np.stack([b.predict(batch) for b in ensemble.bundles]), axis=0)
    np.testing.assert_array_equal(ensemble.predict(batch), expected)
    with pytest.raises(ValueError, match="prespecified seeds"):
        sm.fit_sequence_ensemble(batch, y, groups, seeds=(1, 2, 3), epochs=1)


def test_clipping_only_at_prediction_and_sensor_quality_is_bounded():
    batch, y, groups = fixture()
    fitted = sm.fit_sequence(batch, y, groups, epochs=1)
    quality = sm._sensor_quality(batch, fitted.sensor_scale)
    assert ((quality >= 0) & (quality <= 1)).all()
    with torch.no_grad():
        fitted.he_model.head.weight.zero_()
        fitted.he_model.head.bias.fill_(20)
    assert (fitted.predict(batch) == 10).all()
    assert (fitted.predict(batch, return_raw=True) > 10).all()


@pytest.mark.parametrize("field,value", [(0, np.nan), (0, np.inf), (1, np.nan),
                                         (1, np.inf), (1, 0.), (1, -1.)])
def test_required_history_gate_abstains_and_training_fails_closed(field, value):
    batch, y, groups = fixture()
    fitted = sm.fit_sequence(batch, y, groups, epochs=1)
    expected = fitted.predict(batch, use_sensor=True)
    invalid = deepcopy(batch)
    invalid.history[0, field] = value
    for sensor in (False, True):
        p = fitted.predict(invalid, use_sensor=sensor)
        assert np.isnan(p[0]).all()
        # Removing invalid rows changes matrix multiplication batch shape; valid
        # predictions must agree up to float32 arithmetic, without any refit.
        np.testing.assert_allclose(p[1:], fitted.predict(batch, use_sensor=sensor)[1:],
                                    atol=1e-7, rtol=0)
    with pytest.raises(ValueError, match="Invalid required history"):
        sm.fit_sequence(invalid, y, groups, epochs=1)
    assert np.isfinite(expected).all()


def test_all_invalid_history_abstains_and_no_statistics_fit_is_attempted(monkeypatch):
    batch, y, groups = fixture()
    fitted = sm.fit_sequence(batch, y, groups, epochs=1)
    batch.history[:, 1] = 0
    assert np.isnan(fitted.predict(batch)).all()
    assert np.isnan(fitted.predict(batch, use_sensor=True)).all()
    def must_not_run(*args, **kwargs):
        raise AssertionError("No normalizer may fit before history eligibility")
    monkeypatch.setattr(sm._SequenceScale, "fit", must_not_run)
    with pytest.raises(ValueError, match="Invalid required history"):
        sm.fit_sequence(batch, y, groups, epochs=1)
