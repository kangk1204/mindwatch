"""Synthetic checks for causal histories and selection, without study files."""
import pickle

import numpy as np
import pandas as pd
import pytest

from mindwatch.research import history_decay as model
from mindwatch.research.state_heads import fit


def sample():
    events = pd.DataFrame({
        'subject_id': ['synthetic']*3,
        'date': pd.to_datetime(['2025-01-01', '2025-01-08', '2025-02-01']),
    })
    for domain in model.DOMAINS:
        events[domain] = [2., 8., 9.]
    meta = pd.DataFrame({'subject_id': ['synthetic'],
                         'index_date': pd.to_datetime(['2025-01-08 15:00'])})
    return meta, events


def test_same_day_future_and_other_subject_invariant():
    meta, events = sample()
    expected = model.history_features(meta, events)
    events.loc[events.date.ge('2025-01-08'), list(model.DOMAINS)] = 999
    future = events.loc[events.date.eq('2025-02-01')]
    other = events.iloc[:1].assign(subject_id='other')
    other[list(model.DOMAINS)] = -99
    changed = pd.concat([events, future, other], ignore_index=True)
    pd.testing.assert_frame_equal(expected, model.history_features(meta, changed))
    assert (expected.to_numpy() == 2).all()


def test_age_attenuation_independent_formula_and_missing():
    meta, events = sample()
    meta.index_date = pd.to_datetime(['2025-01-09'])
    recent = model.history_features(meta, events)
    values = np.array([2., 8.])
    ages = np.array([8., 1.])
    expected = values.mean()+2**(-1/14)*(np.average(values, weights=2**(-(ages-1)/14))-values.mean())
    np.testing.assert_allclose(recent.filter(like='ewm14'), expected)
    meta.index_date = pd.to_datetime(['2025-01-23'])
    older = model.history_features(meta, events)
    np.testing.assert_allclose(older.filter(like='ewm14')-5,
                               (recent.filter(like='ewm14')-5)*.5)
    assert model.history_features(meta, events.iloc[:0]).isna().all().all()


def test_duplicate_past_day_bad_values_and_timezone_refused():
    meta, events = sample()
    duplicate = events.iloc[:1].copy()
    duplicate.date += pd.Timedelta(hours=3)
    with pytest.raises(ValueError, match='Duplicate'):
        model.history_features(meta, pd.concat([events, duplicate]))
    with pytest.raises(ValueError, match='EMA must'):
        model.history_features(meta, events.assign(ema_sleep=20))
    with pytest.raises(ValueError, match='timezone'):
        model.history_features(meta.assign(index_date=pd.to_datetime(['2025-01-08T00:00:00Z'])), events)


def test_empty_metadata_and_duplicate_input_indexes():
    meta, events = sample()
    empty = model.history_features(meta.iloc[:0], events)
    assert list(empty.columns) == list(model.HISTORY) and len(empty) == 0
    repeated = pd.concat([meta, meta])
    result = model.history_features(repeated, events)
    assert len(result) == 2 and result.index.tolist() == [0, 0]


def trial(name, n, auc, brier, mae):
    row = dict(id=name, n_features=n, regression_head='ridge')
    for key, value in [('auc', auc), ('brier', brier), ('mae', mae), ('rmse', mae+1)]:
        row['mean_'+key] = value
        row['se_'+key] = .001
    return row


def test_joint_gates_honor_auc_brier_mae():
    a = trial('a', 2, .8, .1, 2.)
    b = trial('b', 3, .801, .09, 1.9)
    assert model.decision([a, b])['selected'] == 'b'
    b['mean_auc'] = .7
    assert model.decision([a, b])['selected'] == 'a'
    b.update(mean_auc=.8, mean_brier=.1, mean_mae=1.8)
    assert model.decision([a, b])['selected'] == 'b'


def test_selection_ties_are_input_order_invariant():
    a = trial('a', 2, .8, .1, 2.)
    b = trial('b', 3, .8, .1, 2.)
    b['se_auc'] = .1
    assert model.decision([a, b]) == model.decision([b, a])
    assert model.decision([a, b])['selected'] == 'a'


@pytest.mark.parametrize('bad', ['negative_se', 'nan', 'duplicate', 'features', 'empty', 'auc'])
def test_selection_rejects_invalid_evidence(bad):
    row = trial('a', 2, .8, .1, 2.)
    rows = [row]
    if bad == 'negative_se':
        row['se_mae'] = -.01
    elif bad == 'nan':
        row['mean_rmse'] = np.nan
    elif bad == 'duplicate':
        rows.append(row.copy())
    elif bad == 'features':
        row['n_features'] = True
    elif bad == 'auc':
        row['mean_auc'] = 2
    else:
        rows = []
    with pytest.raises(ValueError):
        model.decision(rows)


def test_regression_head_tie_prefers_ridge():
    rows = [dict(head=head, mean_mae=2., se_mae=.1, mean_rmse=3., se_rmse=.1)
            for head in ('absolute', 'ridge')]
    assert model.select_head(rows)['head'] == 'ridge'
    rows[0]['se_mae'] = -.1
    with pytest.raises(ValueError):
        model.select_head(rows)


def test_absolute_head_unseen_block_and_serialization():
    rng = np.random.default_rng(12)
    previous = rng.integers(1, 15, 60)
    x = pd.DataFrame({'h_previous_phq9': previous, 'history': rng.normal(size=60), 'empty': np.nan})
    target = np.clip(previous+np.rint(x.history), 0, 27)
    head = model.fit_h_absolute(x, target, ['p'+str(i//2) for i in range(60)], list(x))
    expected = head.predict(x)
    np.testing.assert_array_equal(expected, head.predict(x.assign(empty=20)))
    np.testing.assert_array_equal(expected, pickle.loads(pickle.dumps(head)).predict(x))
    assert head.diagnostic['converged']
    with pytest.raises(ValueError):
        model.fit_h_absolute(x, target, ['same']*59, list(x))
    with pytest.raises(ValueError, match='previous'):
        head.predict(x.assign(h_previous_phq9=np.nan))


def test_available_uses_only_retained_history():
    x = pd.DataFrame({model.HISTORY[0]: np.arange(20.), model.HISTORY[1]: np.nan})
    head = fit(x, np.arange(20) % 2, np.arange(20), list(x), 'class')
    changed = x.assign(**{model.HISTORY[0]: np.nan, model.HISTORY[1]: 5.})
    assert not model.available(head, changed).any()
