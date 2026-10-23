import pytest
pytest.importorskip("torch")
"""Synthetic integration checks; no patient files or prior test results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.neural import flatten_sequence, past_only_delta
from mindwatch.research.temporal_study import (TemporalData, candidates, one_se,
                                   run_comparison, shuffled_sequence, tune)


def synthetic_data():
    rng = np.random.default_rng(84)
    dates = pd.date_range('2025-02-01', '2025-12-31', freq='D')
    n = len(dates)
    x = rng.normal(size=(n, 14, 2))
    mask = (rng.random(x.shape) > .15).astype(float)
    delta = past_only_delta(mask)
    x[mask == 0] = np.nan
    covariate = rng.normal(size=n)
    y = (covariate + rng.normal(size=n) > 0).astype(int)
    f = pd.DataFrame({'sample_id': [f'synthetic_sample_{i}' for i in range(n)],
                      'subject_id': [f'synthetic_person_{i}' for i in range(n)],
                      'index_date': dates, 'partition': 'development', 'y_phq9': y})
    values = np.column_stack([covariate, rng.normal(size=n)])
    packed = flatten_sequence(x, mask, delta)
    designs = {'sensor': {'tabular': values, 'tabular_names': ['sensor__mean', 'demo__age'],
                         'sequence': np.column_stack([packed, values[:, 1:]]),
                         'static_names': ['demo__age'], 'sequence_columns': packed.shape[1]},
               'demographic': {'tabular': values[:, 1:], 'tabular_names': ['demo__age']},
               'missingness': {'tabular': mask.reshape(n, -1), 'tabular_names': []},
               'history_only': {'tabular': values, 'tabular_names': ['history__previous_phq9', 'demo__age']}}
    return TemporalData(f, designs, {'synthetic': True, 'old_test_loaded': False})


def configuration():
    return {'feature_sets': ['sensor'], 'models': ['elasticnet'], 'include_mtm_sensor': False,
            'outer_windows': [('2025-10-01', '2025-11-01'), ('2025-11-01', '2025-12-01'), ('2025-12-01', '2026-01-01')],
            'window': 14, 'gap_days': 14, 'calibration_days': 28, 'inner_folds': 2,
            'seed': 42, 'target_sensitivity': .8, 'final_calibration_start': '2025-12-01',
            'development_end': '2026-01-01', 'order_control': False,
            'parameter_grids': {name: [{'C': .1, 'l1_ratio': 0.0}] for name in
                                ['demographic', 'history_only', 'missingness', 'elasticnet']}}


def test_forward_development_integration_and_freeze(tmp_path):
    data = synthetic_data()
    report = run_comparison(data, configuration(), tmp_path)
    assert report['status'] == 'completed_development_only'
    assert report['test_evaluated'] is False
    assert all(model['status'] == 'ok' for model in report['models'])
    assert report['nested_selectors']['sensor']['metrics']['n_predicted_visits'] == 76
    assert (tmp_path / 'selection.json').is_file()
    assert (tmp_path / 'frozen_sensor.joblib').is_file()
    assert not (tmp_path / 'final_results.json').exists()
    manifest = pd.read_parquet(tmp_path / 'development_cv_manifest.parquet')
    for _, group in manifest.groupby('outer_fold'):
        roles = {role: rows for role, rows in group.groupby('role')}
        assert roles['fit'].index_date.max() + pd.Timedelta(days=14) < roles['calibration'].index_date.min()
        assert roles['calibration'].index_date.max() + pd.Timedelta(days=14) < roles['validation'].index_date.min()
        assert not set(roles['fit'].subject_id) & set(roles['validation'].subject_id)
    frozen = pd.read_parquet(tmp_path / 'frozen_fit_manifest.parquet')
    reservation = pd.read_parquet(tmp_path / 'final_calibration_reservation.parquet')
    assert not set(reservation.subject_id) & set(manifest.subject_id)
    assert not set(reservation.subject_id) & set(frozen.query("role=='fit'").subject_id)
    assert set(frozen.query("role=='calibration'").subject_id) <= set(reservation.subject_id)
    assert frozen.query("role=='fit'").index_date.max() < pd.Timestamp('2025-11-17')
    assert frozen.query("role=='calibration'").index_date.min() >= pd.Timestamp('2025-12-01')
    for file in tmp_path.iterdir():
        if file.is_file():
            assert file.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        run_comparison(data, configuration(), tmp_path)


def test_non_development_rows_rejected_before_fitting(tmp_path):
    data = synthetic_data()
    data.frame.loc[0, 'partition'] = 'temporal_test'
    with pytest.raises(ValueError, match='Only development'):
        run_comparison(data, configuration(), tmp_path)
    assert not (tmp_path / 'development_cv_manifest.parquet').exists()


def test_order_control_preserves_mask_values_and_static_tail():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3, 14, 2))
    m = (rng.random(x.shape) > .4).astype(float)
    x[m == 0] = np.nan
    packed = flatten_sequence(x, m, past_only_delta(m))
    tail = rng.normal(size=(3, 4))
    xx = np.column_stack([packed, tail])
    changed = shuffled_sequence(xx, packed.shape[1], 14, 7)
    np.testing.assert_equal(changed[:, packed.shape[1]:], tail)
    for row in range(3):
        np.testing.assert_equal(np.sort(changed[row, :28]), np.sort(xx[row, :28]))
    changed_m = changed[:, 28:56].reshape(3, 14, 2)
    np.testing.assert_equal(changed[:, 56:84], past_only_delta(changed_m).reshape(3, -1))
    assert not np.array_equal(changed_m, m)


def test_one_se_uses_only_passed_scores_and_favors_simple_model():
    records = [{'name': 'demographic', 'model': 'demographic', 'status': 'ok', 'mean_auroc': .70, 'se_auroc': .01},
               {'name': 'sensor_grud', 'model': 'grud', 'status': 'ok', 'mean_auroc': .73, 'se_auroc': .04},
               {'name': 'failed', 'model': 'tcn', 'status': 'incomplete_forward_cv', 'mean_auroc': .99}]
    result = one_se(records)
    assert result['selected_name'] == 'demographic'
    assert result['best_mean_name'] == 'sensor_grud'


def test_candidates_do_not_offer_voice_or_old_test():
    all_candidates = candidates(configuration())
    assert all(c['model'] not in {'audio', 'text'} for c in all_candidates)
    assert {c['name'] for c in all_candidates} == {'demographic', 'history_only', 'missingness', 'sensor_elasticnet'}


def test_outer_outcome_cannot_change_inner_tuning():
    from mindwatch.research.temporal_splits import make_forward_fold
    data = synthetic_data()
    config = configuration()
    split = make_forward_fold(data.frame, '2025-12-01', '2026-01-01')
    candidate = {'name': 'sensor_elasticnet', 'scenario': 'sensor', 'model': 'elasticnet'}
    before = tune(data, candidate, split['fit'], config, 42)
    data.frame.loc[split['validation'], 'y_phq9'] = 1 - data.frame.loc[split['validation'], 'y_phq9']
    after = tune(data, candidate, split['fit'], config, 42)
    assert before == after


def test_abstention_keeps_all_labelled_validation_people_in_denominator(tmp_path):
    data = synthetic_data()
    data.frame['quality__sensor_eligible'] = 1
    # The same synthetic participant has an earlier otherwise-trainable row.
    data.frame.loc[data.frame.index_date.eq('2025-03-01'), 'subject_id'] = 'reserved_missing_person'
    missing = data.frame.index[data.frame.index_date.eq('2025-10-03')][0]
    data.frame.loc[missing, ['subject_id', 'quality__sensor_eligible']] = ['reserved_missing_person', 0]
    report = run_comparison(data, configuration(), tmp_path)
    table = pd.read_parquet(tmp_path / 'oof_predictions.parquet')
    held = table.query("model=='sensor_elasticnet' and outer_fold==0")
    assert len(held) == 31
    assert held.probability.isna().sum() == 1
    assert report['models'][-1]['pooled_oof_metrics']['n_labelled_visits'] == 76
    assert report['models'][-1]['pooled_oof_metrics']['n_predicted_visits'] == 75
    roles = pd.read_parquet(tmp_path / 'development_cv_manifest.parquet')
    assert not ((roles.outer_fold == 0) & (roles.role == 'fit') & (roles.subject_id == 'reserved_missing_person')).any()


def test_control_failure_cannot_leave_completed_status(tmp_path, monkeypatch):
    def broken_controls(*args):
        raise ValueError('Synthetic control failure')
    monkeypatch.setattr('mindwatch.research.temporal_study._controls', broken_controls)
    config = configuration()
    config['order_control'] = True
    report = run_comparison(synthetic_data(), config, tmp_path)
    assert report['status'] == 'completed_with_failures_development_only'
    assert report['controls']['status'] == 'failed'
    assert report['controls']['exception_type'] == 'ValueError'
    assert json.loads((tmp_path / 'selection.json').read_text())['status'] == report['status']


def test_checkpoint_failure_cannot_count_as_successful_fold(tmp_path, monkeypatch):
    from mindwatch.research import temporal_study
    original = temporal_study._checkpoint
    def broken_checkpoint(path, model):
        if path.name == 'fold_0_sensor_elasticnet.joblib':
            raise OSError('Synthetic storage failure')
        return original(path, model)
    monkeypatch.setattr(temporal_study, '_checkpoint', broken_checkpoint)
    report = run_comparison(synthetic_data(), configuration(), tmp_path)
    record = next(r for r in report['models'] if r['name'] == 'sensor_elasticnet')
    assert record['status'] == 'incomplete_forward_cv'
    assert len(record['fold_aurocs']) == 2
    assert report['status'] == 'completed_with_failures_development_only'


def test_reserved_calibration_outcomes_cannot_change_model_selection(tmp_path):
    data = synthetic_data()
    before = run_comparison(data, configuration(), tmp_path / 'before')
    reservation = pd.read_parquet(tmp_path / 'before' / 'final_calibration_reservation.parquet')
    rows = data.frame.subject_id.isin(reservation.subject_id)
    data.frame.loc[rows, 'y_phq9'] = 1 - data.frame.loc[rows, 'y_phq9']
    after = run_comparison(data, configuration(), tmp_path / 'after')
    assert [r['fold_aurocs'] for r in before['models']] == [r['fold_aurocs'] for r in after['models']]
    assert before['nested_selectors'] == after['nested_selectors']
    assert before['final_models']['sensor']['selected_name'] == after['final_models']['sensor']['selected_name']
    assert before['final_models']['sensor']['tuning'] == after['final_models']['sensor']['tuning']
