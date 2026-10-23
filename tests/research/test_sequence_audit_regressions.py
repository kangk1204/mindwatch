"""Synthetic counterexamples for sparse weighting and neural prediction contracts."""
import pickle
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from mindwatch.research import ema_joint, neural_comparators, representations, minirocket_adapter


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sample(n=6):
    values = np.random.default_rng(7).normal(size=(n, 14, 2)).astype(np.float32)
    groups = np.array([f'synthetic_{i}' for i in range(n)])
    frame = pd.DataFrame({'subject_id': groups})
    x = pd.DataFrame({'e_last': np.arange(n, dtype=float)})
    return groups, frame, x, values


@pytest.mark.parametrize('kind', ['representation', 'neural', 'joint', 'rocket'])
def test_fractional_observed_weight_preserves_known_constant(kind, monkeypatch):
    groups = np.array(['synthetic_repeat']*100+['synthetic_single'])
    values = np.full((101, 14, 2), np.nan, np.float32)
    values[:, 0, 1] = 2
    values[0, 0, 0] = 10
    if kind == 'representation':
        model = representations.fit_representation(values, groups, kind='random', epochs=1, seeds=(42,))
    elif kind == 'neural':
        model = neural_comparators.fit(pd.DataFrame({'subject_id': groups}), pd.DataFrame({'e_last': np.arange(101)%2}),
                                      values, np.arange(101)%2, 'phq', None, 'grud', epochs=1, seeds=(42,))
    elif kind == 'joint':
        model = ema_joint.fit(groups, values, np.full((101, 4), 5.), np.full((101, 4), 5.),
                              'ema_depression', epochs=1, seeds=(42,))
    else:
        pytest.importorskip('numba')
        class UpstreamStub:
            def fit(self, x, num_features):
                assert x.dtype == np.float32 and x.flags.c_contiguous
                return num_features
            def transform(self, x, state):
                assert x.dtype == np.float32 and x.flags.c_contiguous
                return np.zeros((len(x), state), np.float32)
        monkeypatch.setattr(minirocket_adapter, 'load_source', lambda _: UpstreamStub())
        model = minirocket_adapter.RocketTransform('synthetic_source', 84).fit(values, groups)
        assert model.transform(values[:0]).shape == (0, 84)
    np.testing.assert_allclose(model.mean, [10., 2.], atol=1e-6)
    np.testing.assert_allclose(model.scale, [1., 1.], atol=1e-6)


def test_global_loss_normalizer_preserves_weighted_gradient_across_batches():
    prediction = torch.tensor([0., 0.], requires_grad=True)
    target, weight = torch.tensor([0., 1.]), torch.tensor([.1, .9])
    whole = neural_comparators._loss(prediction, target, weight, 'phq', weight.mean())
    expected = torch.autograd.grad(whole, prediction)[0]
    parts = []
    for i in range(2):
        loss = neural_comparators._loss(prediction[i:i+1], target[i:i+1], weight[i:i+1], 'phq', weight.mean())
        parts.append(torch.autograd.grad(loss, prediction)[0])
    torch.testing.assert_close(expected, torch.tensor([.05, -.45]))
    torch.testing.assert_close(torch.stack(parts).mean(0), expected)


def test_joint_missing_label_objective_is_global_and_zero_for_unsupervised_heads():
    prediction = torch.zeros((2, 4), requires_grad=True)
    target = torch.tensor([[1., 3., 0., 0.], [-1., -2., 0., 0.]])
    weight = torch.tensor([[.1, 1., 0., 0.], [.9, 0., 0., 0.]])
    normalizer, active = weight.sum(0)/2, weight.sum(0) > 0
    whole = ema_joint._loss(prediction, target, weight, normalizer, active)
    expected = torch.autograd.grad(whole, prediction)[0]
    parts = []
    for i in range(2):
        loss = ema_joint._loss(prediction[i:i+1], target[i:i+1], weight[i:i+1], normalizer, active)
        parts.append(torch.autograd.grad(loss, prediction)[0])
    torch.testing.assert_close(torch.stack(parts).mean(0), expected)
    assert torch.count_nonzero(expected[:, 2:]) == 0


def test_unlabelled_joint_domain_has_exact_prior_fallback_and_roundtrip():
    groups, frame, x, values = sample()
    target = np.full((6, 4), 5.)
    target[:, 0] = np.nan
    base = np.full((6, 4), 5.)
    model = ema_joint.fit(groups, values, target, base, 'ema_depression', epochs=1, seeds=(42,))
    prior = np.array([0., 10., 4., np.nan, 3., 6.])
    np.testing.assert_array_equal(model.raw(x, values, prior), prior)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).raw(x, values, prior), prior)
    assert model.raw(x.iloc[:0], values[:0], prior[:0]).shape == (0,)
    with pytest.raises(ValueError, match='P0'):
        model.raw(x, values)


@pytest.mark.parametrize('task', ['phq', 'ema'])
def test_empty_and_absent_neural_sensor_predictions(task):
    groups, frame, x, values = sample()
    labels = np.arange(6)%2 if task == 'phq' else np.full(6, 5.)
    prior = None if task == 'phq' else np.array([0., 10., 4., 3., 3., 6.])
    model = neural_comparators.fit(frame, x, values, labels, task, None, 'grud',
                                  p0_values=prior, epochs=1, seeds=(42,))
    expected = model.base.predict(x[list(model.base.cols)]) if task == 'phq' else prior
    np.testing.assert_array_equal(model.raw(x, np.full_like(values, np.nan), prior), expected)
    assert model.raw(x.iloc[:0], values[:0], None if prior is None else prior[:0]).shape == (0,)
    np.testing.assert_array_equal(model.raw(x, values, prior), pickle.loads(pickle.dumps(model)).raw(x, values, prior))
    with pytest.raises(ValueError, match='length'):
        model.raw(x, values[:, :-1], prior)


@pytest.mark.parametrize('kwargs', [{'epochs': 0}, {'epochs': -1}, {'epochs': True}, {'epochs': 1.5}, {'seeds': ()}, {'seeds': None}, {'seeds': (42, 42)}, {'seeds': (-1,)}])
def test_invalid_training_schedule_rejected(kwargs):
    groups, frame, x, values = sample()
    settings = {'epochs': 1, 'seeds': (42,)} | kwargs
    with pytest.raises(ValueError):
        neural_comparators.fit(frame, x, values, np.arange(6)%2, 'phq', None, 'grud', **settings)
    with pytest.raises(ValueError):
        ema_joint.fit(groups, values, np.full((6, 4), 5.), np.full((6, 4), 5.), 'ema_depression', **settings)


@pytest.mark.parametrize('invalid', [[2., 3., 2., 3., 2., 3.], [0., 1., np.inf, 0., 1., 0.]])
def test_invalid_phq_labels_rejected(invalid):
    groups, frame, x, values = sample()
    with pytest.raises(ValueError):
        neural_comparators.fit(frame, x, values, invalid, 'phq', None, 'grud', epochs=1, seeds=(42,))


@pytest.mark.parametrize('kind', ['labels', 'p0', 'shape', 'infinite_sensor'])
def test_invalid_ema_contract_rejected(kind):
    groups, frame, x, values = sample()
    target, prior = np.full((6, 4), 5.), np.full((6, 4), 5.)
    if kind == 'labels':
        target[0, 0] = 11
    elif kind == 'p0':
        prior[0, 0] = -1
    elif kind == 'shape':
        target = target[:-1]
    else:
        values[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        ema_joint.fit(groups, values, target, prior, 'ema_depression', epochs=1, seeds=(42,))


def test_reordered_feature_rows_are_rejected_before_training():
    groups, frame, x, values = sample()
    with pytest.raises(ValueError, match='Aligned'):
        neural_comparators.fit(frame, x.iloc[::-1], values, np.arange(6)%2, 'phq', None, 'grud', epochs=1, seeds=(42,))


def test_mtm_length_guard_precedes_external_source_loading():
    groups, frame, x, _ = sample()
    values = np.ones((6, 639, 2), np.float32)
    with pytest.raises(ValueError, match='638'):
        neural_comparators.fit(frame, x, values, np.arange(6)%2, 'phq', None, 'mtm', epochs=1, seeds=(42,))
    with pytest.raises(ValueError, match='638'):
        ema_joint.fit(groups, values, np.full((6, 4), 5.), np.full((6, 4), 5.), 'ema_depression', architecture='mtm', epochs=1, seeds=(42,))
