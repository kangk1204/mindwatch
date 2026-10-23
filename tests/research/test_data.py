import json
import numpy as np
import pandas as pd
import pytest
from mindwatch.research.clinical import validate_scale
from mindwatch.research.features import window_features, elapsed_delta, clock_hours, personal_change
from mindwatch.research.privacy import get_key, token, write_json
from mindwatch.research.splits import make_temporal_split, lock_split, prognosis_split


def visits():
    return pd.DataFrame({'sample_id': ['v1', 'v2'], 'subject_id': ['a', 'b'],
                         'index_date': pd.to_datetime(['2026-01-15', '2026-01-20']),
                         'enrolled_date': pd.to_datetime(['2025-01-01', '2025-07-01']),
                         'age_group': ['40대', '70~79'], 'sex': ['1', '2']})


def test_item_scores_are_verified_not_trusted():
    df = pd.DataFrame({'q1': [1, 1, 7, 1.5], 'q2': [2, 2, 1, 1], 'score': [3, 4, 8, 2.5]})
    total, audit = validate_scale(df, ['q1', 'q2'], 'score', 3)
    assert total.notna().tolist() == [True, False, False, False]
    assert audit['provided_mismatch'] == 1


def test_no_future_or_current_date_in_windows():
    daily = pd.DataFrame({'subject_id': ['a']*4, 'date': pd.to_datetime(['2026-01-01', '2026-01-14', '2026-01-15', '2026-01-16']),
                          'steps': [0., 10., 100000., 1e8], 'ema_depression': [1., 2., 10., 100.]})
    features, array = window_features(visits(), daily, 14, 2)
    assert features.loc[0, 'sensor__steps__mean'] == 5
    assert features.loc[0, 'ema__ema_depression__last'] == 2
    assert np.nanmax(array['values'][0]) == 10
    assert features.loc[0, 'quality__sensor_eligible'] == 1
    assert features.loc[1, 'quality__sensor_eligible'] == 0
    assert array['mask'][0, 0, 0] == 1  # true zero is observed


def test_delta_uses_previous_mask_only():
    assert elapsed_delta(np.array([[1], [0], [0], [1], [0]])).ravel().tolist() == [1, 1, 2, 3, 1]


def test_clock_midnight_and_invalid_values():
    got = clock_hours(pd.Series(['23:30:00', '00:30', '24:00', '12:90', 'bad']))
    assert got.iloc[:2].tolist() == [23.5, .5]
    assert got.iloc[2:].isna().all()


def test_subject_and_time_disjoint_split_and_lock(tmp_path):
    v = visits()
    extra = v.iloc[[0]].copy()
    extra['index_date'] = pd.Timestamp('2026-03-01')
    extra['sample_id'] = 'v3'
    v = pd.concat([v, extra], ignore_index=True)
    manifest, spec = lock_split(v, tmp_path, .5)
    assert 'excluded_after_cutoff' in set(manifest.partition)
    test = set(manifest.loc[manifest.partition.eq('temporal_test'), 'subject_id'])
    dev = set(manifest.loc[manifest.partition.eq('development'), 'subject_id'])
    assert not (test & dev)
    changed = v.copy();changed.loc[0, 'index_date'] = pd.Timestamp('2025-01-01')
    with pytest.raises(ValueError):
        lock_split(changed, tmp_path, .5)


def test_prognosis_future_label_is_purged():
    v = pd.DataFrame({'sample_id': ['a'], 'partition': ['development']})
    f = pd.DataFrame({'sample_id': ['a'], 'target_date': pd.to_datetime(['2026-04-01'])})
    assert prognosis_split(f, v, '2026-03-01').partition.iloc[0].startswith('excluded')


def test_hmac_and_permissions(tmp_path):
    k = get_key(tmp_path)
    assert token(k, 'person') == token(k, 'person')
    assert token(k, 'person') != token(k, 'person', 'sample')
    write_json(tmp_path / 'safe.json', {'nan': np.nan})
    assert json.loads((tmp_path / 'safe.json').read_text()) == {'nan': None}
    assert (tmp_path / 'safe.json').stat().st_mode & 0o077 == 0


def test_personal_change_never_uses_current_value():
    daily = pd.DataFrame({'subject_id': ['p']*30, 'date': pd.date_range('2026-01-01', periods=30), 'steps': np.arange(30.)})
    first = personal_change(daily)
    daily.loc[29, 'steps'] = 1000.
    second = personal_change(daily)
    assert np.allclose(first.change_score.iloc[:29], second.change_score.iloc[:29], equal_nan=True)


def test_daily_duplicates_across_chunk_boundary_are_rejected(tmp_path):
    import zipfile
    from mindwatch.research.sources import Source
    from mindwatch.research.features import aggregate_daily
    dates = list(pd.date_range('2025-01-01', periods=128)) + [pd.Timestamp('2025-01-01')]
    csv = pd.DataFrame({'date': dates, 'synthetic_id': np.arange(129.)}).to_csv(index=False)
    path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('synthetic.csv',csv)
    source=Source('A01',path,0,'steps','2026-01-01',len(csv),0,'sensor')
    with pytest.raises(ValueError,match='multiple_rows_per_day'):
        aggregate_daily([source],{'synthetic_id':'pseudonym'},exploratory=True)
