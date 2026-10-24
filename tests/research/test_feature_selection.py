"""Synthetic-only checks for refitting and temporal feature selection."""
import joblib
import numpy as np
import pandas as pd
import pytest

from mindwatch.features import FEATURE_COLUMNS
from mindwatch.models import fit_head
from mindwatch.research.feature_selection import (
    FeatureRecipe, ablation_recipes, choose_one_se, collection_pattern_recipes,
    select_on_training, temporal_folds,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(18)
    n = 960
    dates = pd.Timestamp('2024-01-01')+pd.to_timedelta(np.arange(n) % 240, unit='D')
    meta = pd.DataFrame({'subject_id': ['synthetic_'+str(i) for i in range(n)],
                         'index_date': dates, 'label_available_at': dates+pd.Timedelta(days=1)})
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (x.h_previous_phq9.to_numpy()+rng.normal(size=n) > 0).astype(int)
    return x, y, meta


@pytest.mark.parametrize('index', [0, 2])
def test_explicit_full_recipe_reproduces_default(synthetic, index):
    x, y, meta = synthetic
    recipe = collection_pattern_recipes()[index]
    expected = fit_head(x, y, meta.subject_id, recipe.family)
    actual = recipe.fit(x, y, meta.subject_id)
    np.testing.assert_array_equal(expected.predict(x), actual.predict(x))


@pytest.mark.parametrize('index', [1, 3])
def test_deleted_group_has_no_prediction_effect(synthetic, index, tmp_path):
    x, y, meta = synthetic
    recipe = collection_pattern_recipes()[index]
    model = recipe.fit(x, y, meta.subject_id)
    changed = x.copy()
    for c in set(FEATURE_COLUMNS)-set(recipe.columns):
        changed[c] = 1e6
    np.testing.assert_array_equal(model.predict(x), model.predict(changed))
    joblib.dump(model, tmp_path/'synthetic.joblib')
    restored = joblib.load(tmp_path/'synthetic.joblib')
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_single_and_group_ablation_definitions():
    recipe = collection_pattern_recipes()[0]
    singles = ablation_recipes(recipe)
    assert len(singles) == len(recipe.columns)+1
    assert all(len(r.columns) == len(recipe.columns)-1 for r in singles[1:])
    group = tuple(c for c in recipe.columns if 'phq9' in c)
    reduced = ablation_recipes(recipe, {'phq_history': group})[1]
    assert not any('phq9' in c for c in reduced.columns)
    with pytest.raises(ValueError):
        ablation_recipes(recipe, {'invalid': ('current_score',)})


def test_empty_and_training_unobserved_recipe_returns_constant(synthetic):
    x, y, meta = synthetic
    for columns in [(), ('h_previous_phq9',)]:
        frame = x.copy(); frame['h_previous_phq9'] = np.nan
        model = FeatureRecipe('synthetic', 'H', columns).fit(frame, y, meta.subject_id)
        changed = frame.copy(); changed['h_previous_phq9'] = 7
        np.testing.assert_array_equal(model.predict(frame), model.predict(changed))
        assert model.estimator is None


@pytest.mark.parametrize('columns', [('current_score',), ('h_previous_phq9', 'h_previous_phq9'), 'h_previous_phq9'])
def test_invalid_reduced_schema_is_rejected(synthetic, columns):
    x, y, meta = synthetic
    with pytest.raises(ValueError):
        fit_head(x, y, meta.subject_id, 'H', columns=columns)


def test_real_two_fold_selection_with_disjoint_people(synthetic):
    x, y, meta = synthetic
    result = select_on_training(x, y, meta, '2024-09-01', collection_pattern_recipes(), fallback='H_full')
    assert result['status'] == 'inner_selected'
    assert result['valid_folds'] == 2 and result['fit_count'] == 8
    assert len(result['trials']) == 4 and all(len(t['scores']) == 2 for t in result['trials'])
    for fold in result['folds']:
        for first, second in [('train', 'calibration'), ('train', 'validation'), ('calibration', 'validation')]:
            assert not set(meta.iloc[fold[first]].subject_id) & set(meta.iloc[fold[second]].subject_id)
        assert meta.iloc[fold['train']].label_available_at.max() < pd.Timestamp(fold['start'])-pd.Timedelta(days=14)
    # Future rows cannot silently be consumed by a selector expecting training-only input.
    contaminated = meta.copy(); contaminated.loc[0, 'label_available_at'] = pd.Timestamp('2024-09-01')
    with pytest.raises(ValueError, match='before the training boundary'):
        select_on_training(x, y, contaminated, '2024-09-01', collection_pattern_recipes(), fallback='H_full')


def test_reserved_people_and_unavailable_labels_are_rejected(synthetic):
    x, y, meta = synthetic
    with pytest.raises(ValueError, match='Excluded participant'):
        temporal_folds(meta, y, '2024-09-01', excluded_people=[meta.subject_id.iloc[0]])
    same_day = meta.copy(); same_day['label_available_at'] = same_day.index_date
    with pytest.raises(ValueError, match='date-only labels'):
        temporal_folds(same_day, y, '2024-09-01')
    intraday = meta.copy(); intraday.loc[0, 'index_date'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='Date-only'):
        temporal_folds(intraday, y, '2024-09-01')
    with pytest.raises(ValueError, match='row indices'):
        select_on_training(x.rename(index=lambda i: i+1), y, meta, '2024-09-01', collection_pattern_recipes(), fallback='H_full')


def test_support_failure_preserves_explicit_fallback(synthetic):
    x, y, meta = synthetic
    for fallback in ['H_full', 'compact_full']:
        result = select_on_training(x.iloc[:8], y[:8], meta.iloc[:8], '2024-09-01', collection_pattern_recipes(), fallback=fallback)
        assert result['status'] == 'warmup_fallback' and result['selected'] == fallback
        assert result['fit_count'] == 0


def test_one_se_uses_brier_and_has_order_independent_ties():
    def row(name, auc, br, size):
        return dict(name=name, mean_auc=auc, se_auc=.02, mean_brier=br, se_brier=.001, n_features=size)
    trials = [row('larger', .80, .15, 12), row('smaller', .79, .12, 4)]
    assert choose_one_se(trials)['selected'] == 'smaller'
    trials[1]['mean_auc'] = .65
    assert choose_one_se(trials)['selected'] == 'larger'
    tied = [row('a', .8, .12, 4), row('b', .8, .12, 4)]
    assert choose_one_se(tied) == choose_one_se(tied[::-1])
