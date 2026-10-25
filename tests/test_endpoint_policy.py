"""Synthetic-only tests for separately configured risk and score endpoints."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch import endpoint_policy as endpoint
from mindwatch import lightweight as core
from mindwatch import lightweight_policy as history


@pytest.fixture
def records():
    query = pd.DataFrame({'sample_id': [f'synthetic_q{i}' for i in range(3)],
        'subject_id': ['synthetic_train', 'synthetic_cal', 'synthetic_new'],
        'index_date': pd.Timestamp('2020-03-01 15:00')})
    surveys = pd.DataFrame({'subject_id': query.subject_id,
        'index_date': pd.Timestamp('2020-01-01'), 'phq9': [26., 8., 4.],
        'gad7': [7., 3., 1.], 'isi': [11., 9., 6.]})
    ema = pd.DataFrame({'subject_id': ['synthetic_train'],
        'date': [pd.Timestamp('2020-02-28')], **{domain: [2.] for domain in core.DOMAINS}})
    return query, surveys, ema


@pytest.fixture
def model():
    n = len(core.FEATURES)
    classifier = core.NumericHead(core.FEATURES, np.ones(n, bool), np.ones(n),
        np.zeros(n), np.ones(n), np.full(n, .02), -.7)
    # Deliberately force clipping before the endpoint shrink operation.
    regressor = replace(classifier, coef=np.zeros(n), intercept=100.)
    primary = core.FrozenLightweight(classifier, regressor,
        frozenset({'synthetic_train'}), frozenset({'synthetic_cal'}),
        'synthetic_primary', 'synthetic_data', calibration_intercept=.3,
        calibration_slope=.7, threshold=.43, calibration_status='synthetic_fixed')
    fallback = history.NumericHistoryHead(('h_previous_phq9',), np.ones(1, bool),
        np.ones(1), (), np.zeros(1), np.ones(1), np.array([.09]), -.8)
    routed = history.FrozenHistoryPolicy(primary, fallback,
        replace(fallback, coef=np.zeros(1), intercept=-100.), 'synthetic_history')
    return endpoint.FrozenEndpointPolicy(routed, .4, 'persistence',
        'synthetic_risk', 'synthetic_score', frozenset({'synthetic_excluded'}),
        'synthetic_risk_source', 'synthetic_score_source')


def test_classifier_calibration_threshold_unchanged_and_clip_then_shrink(records, model):
    q, s, e = records
    meta, x, _ = history.features_from_history(q, s, e)
    base = model.classifier.predict(meta, x)
    out = model.predict(meta, x)
    for name in ('raw_prediction', 'prediction', 'threshold', 'calibration_status',
                 'population_exposure', 'history_fallback_used'):
        pd.testing.assert_series_equal(out[name], base[name])
    np.testing.assert_allclose(out.predicted_phq9, [26.4, 8., 4.], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(out.predicted_phq9_change, out.predicted_phq9-x.h_previous_phq9)
    assert out.classification_route.tolist() == ['compact', 'synthetic_history', 'synthetic_history']
    assert out.regression_route.tolist() == ['shrunken_base_change', 'persistence', 'persistence']
    assert out.fallback_reason_scope.eq('classification').all()
    assert out.regression_fallback_reason.iloc[1:].eq('no_recent_ema_use_previous_phq').all()
    assert out.classification_model_version.eq('synthetic_risk').all()
    assert out.regression_model_version.eq('synthetic_score').all()
    assert out.classification_source.eq('synthetic_risk_source').all()
    # Calling the endpoint wrapper did not alter the original frozen model.
    pd.testing.assert_frame_equal(base, model.classifier.predict(meta, x))


@pytest.mark.parametrize('weight', [0., .6, 1.])
def test_required_weight_partial_ema_and_optional_base_regressor_route(records, model, weight):
    q, s, e = records
    e.loc[:, list(core.DOMAINS[1:])] = np.nan
    candidate = replace(model, ema_change_weight=weight, no_ema_score_route='base_regressor')
    base = candidate.classifier.predict_from_history(q, s, e)
    out = candidate.predict_from_history(q, s, e)
    assert out.predicted_phq9.iloc[0] == 26.+weight*(27.-26.)
    np.testing.assert_array_equal(out.predicted_phq9.iloc[1:], base.predicted_phq9.iloc[1:])
    assert out.regression_route.iloc[1:].eq('base_regressor').all()
    assert out.regression_used_modalities.iloc[1:].eq('past_surveys').all()
    assert out.regression_ema_used.iloc[0] == (weight > 0)


def test_observed_but_unretained_ema_not_reported_as_effective_score_input(records, model):
    q, s, e = records
    regressor = replace(model.classifier.primary.change_regressor,
                        seen=np.array([True, *([False]*(len(core.FEATURES)-1))]))
    primary = replace(model.classifier.primary, change_regressor=regressor)
    candidate = replace(model, classifier=replace(model.classifier, primary=primary))
    out = candidate.predict_from_history(q, s, e)
    assert not out.regression_ema_used.any()
    assert out.regression_used_modalities.eq('past_phq').all()


def test_unit_weight_preserves_base_scores_exactly(records, model):
    q, s, e = records
    primary = replace(model.classifier.primary, change_regressor=replace(
        model.classifier.primary.change_regressor, intercept=-.137))
    routed = replace(model.classifier, primary=primary, fallback_change_regressor=replace(
        model.classifier.fallback_change_regressor, intercept=.263))
    candidate = replace(model, classifier=routed, ema_change_weight=1., no_ema_score_route='base_regressor')
    expected = routed.predict_from_history(q, s, e)
    actual = candidate.predict_from_history(q, s, e)
    np.testing.assert_array_equal(actual.predicted_phq9, expected.predicted_phq9)


@pytest.mark.parametrize('mode', ['disabled_regression', 'classifier_imputation',
                                  'classifier_only_ema', 'separate_head_usage'])
def test_usage_metadata_tracks_only_active_endpoint_heads(records, model, mode):
    q, s, e = records
    q = q.iloc[:1]
    n = len(core.FEATURES)
    class_seen = np.array([True, *([False]*(n-1))])
    reg_seen = np.ones(n, bool)
    weight = 0.
    if mode in ('classifier_imputation', 'separate_head_usage'):
        class_seen[1:3] = True
        reg_seen[:] = False; reg_seen[0] = True; reg_seen[3:5] = True
        e.loc[:, list(core.DOMAINS)] = np.nan
        e[core.DOMAINS[1]] = 2.
        if mode == 'separate_head_usage': weight = .6
    elif mode == 'classifier_only_ema':
        class_seen[:] = True
        reg_seen[1:] = False
    primary = replace(model.classifier.primary,
        classifier=replace(model.classifier.primary.classifier, seen=class_seen),
        change_regressor=replace(model.classifier.primary.change_regressor, seen=reg_seen))
    candidate = replace(model, classifier=replace(model.classifier, primary=primary), ema_change_weight=weight)
    out = candidate.predict_from_history(q, s, e)
    raw = candidate.classifier.predict_from_history(q, s, e)
    np.testing.assert_array_equal(out.prediction, raw.prediction)
    class_uses = mode == 'classifier_only_ema'
    reg_uses = mode == 'separate_head_usage'
    assert out.classification_ema_used.iloc[0] == class_uses
    assert out.regression_ema_used.iloc[0] == reg_uses
    assert out.ema_used.iloc[0] == (class_uses or reg_uses)
    assert out.used_modalities.iloc[0] == ('past_phq+ema' if class_uses or reg_uses else 'past_phq')
    assert out.classification_status.iloc[0] == ('ok_ema_missing_imputed' if mode in
        ('classifier_imputation', 'separate_head_usage') else 'ok')
    assert out.status.iloc[0] == ('ok_ema_missing_imputed' if mode == 'classifier_imputation' else 'ok')
    assert out.classification_fallback_reason.iloc[0] == ('ema_missing_frozen_training_imputation'
        if mode in ('classifier_imputation', 'separate_head_usage') else '')
    if weight == 0:
        assert out.regression_ema_retained_features.eq(0).all()
        assert out.regression_ema_used_features.eq(0).all()
    if mode == 'disabled_regression':
        assert out.ema_retained_features.eq(0).all() and out.ema_used_features.eq(0).all()
    if mode == 'classifier_imputation':
        assert out.ema_retained_features.eq(2).all() and out.ema_used_features.eq(0).all()


def test_raw_history_causality_provenance_and_optional_auxiliary_surveys(records, model):
    q, s, e = records
    expected = model.predict_from_history(q, s, e)
    future_s = s.assign(index_date=pd.Timestamp('2020-03-01 10:00'), phq9=999., gad7=999., isi=999.)
    future_e = e.assign(date=pd.Timestamp('2020-03-01 10:00'), **{domain: 999. for domain in core.DOMAINS})
    changed = model.predict_from_history(q, pd.concat([s, future_s]), pd.concat([e, future_e]))
    pd.testing.assert_frame_equal(changed, expected)
    unrelated_s = s.assign(subject_id='synthetic_unrelated')
    unrelated_e = e.assign(subject_id='synthetic_unrelated')
    pd.testing.assert_frame_equal(expected, model.predict_from_history(q,
        pd.concat([s, unrelated_s]), pd.concat([e, unrelated_e])))
    optional = model.predict_from_history(q, s.drop(columns=['gad7', 'isi']), e)
    assert optional.past_gad7_available.eq(False).all()
    meta, x, _ = history.features_from_history(q, s, e)
    prepared = model.predict(meta, x)
    for name in ('prediction', 'raw_prediction', 'predicted_phq9', 'threshold'):
        np.testing.assert_array_equal(prepared[name], expected[name])
    assert prepared.input_provenance_status.eq('not_provided').all()
    assert prepared.latest_phq9_input_date.isna().all()
    assert expected.input_provenance_status.eq('history_derived').all()


def test_missing_prior_empty_inputs_and_population_exposure(records, model):
    q, s, e = records
    out = model.predict_from_history(q, s.iloc[:0], e)
    assert out.status.eq('insufficient_data').all()
    assert out.prediction.isna().all() and out.predicted_phq9.isna().all()
    assert out.regression_route.eq('none').all()
    assert out.regression_change_weight.isna().all()
    assert model.predict_from_history(q.iloc[:0], s, e).empty
    actual = model.predict_from_history(q, s, e)
    assert actual.policy.tolist() == ['known', 'calibration_only', 'new']
    for index in (0, 1):
        with pytest.raises(ValueError, match='cannot be declared new'):
            model.predict_from_history(q.iloc[index:index+1], s, e, policy='new')
    with pytest.raises(ValueError, match='training membership'):
        model.predict_from_history(q.iloc[2:], s, e, policy='known')


def test_excluded_queries_stop_before_features_or_model(records, model, monkeypatch):
    q, s, e = records
    meta, x, _ = history.features_from_history(q, s, e)
    blocked = q.assign(subject_id='synthetic_excluded')
    def forbidden(*args, **kwargs):
        raise AssertionError('Excluded queries must not reach features or models')
    monkeypatch.setattr(history, 'features_from_history', forbidden)
    monkeypatch.setattr(history.FrozenHistoryPolicy, 'predict', forbidden)
    with pytest.raises(ValueError, match='Excluded participant'):
        model.predict_from_history(blocked, s, e)
    with pytest.raises(ValueError, match='Excluded participant'):
        model.predict(blocked, x)
    endpoint.guard_queries(pd.DataFrame({'subject_id': ['synthetic_new']}), model.excluded_people)


@pytest.mark.parametrize('kind', ['target_query', 'target_feature', 'schema', 'index', 'infinity'])
def test_prepared_input_contract(records, model, kind):
    q, s, e = records
    meta, x, _ = history.features_from_history(q, s, e)
    if kind == 'target_query': meta = meta.assign(phq9=10)
    elif kind == 'target_feature': x = x.assign(target=1)
    elif kind == 'schema': x = x.loc[:, list(core.FEATURES)]
    elif kind == 'index': x.index = x.index+1
    else: x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError): model.predict(meta, x)


@pytest.mark.parametrize('change', [
    {'ema_change_weight': -1}, {'ema_change_weight': 2}, {'ema_change_weight': np.inf},
    {'ema_change_weight': np.nan}, {'ema_change_weight': True}, {'ema_change_weight': '.4'},
    {'ema_change_weight': 1j}, {'no_ema_score_route': 'guess'}, {'classification_model_version': ''},
    {'regression_model_version': None}, {'classification_source': ''}, {'excluded_people': 'synthetic'},
    {'excluded_people': frozenset({''})}, {'classifier': None}])
def test_explicit_frozen_contract(model, change):
    with pytest.raises((TypeError, ValueError)): replace(model, **change)


def test_private_numeric_roundtrip_and_no_pickle(records, model, tmp_path, monkeypatch):
    q, s, e = records
    directory = tmp_path.resolve()/'release'
    endpoint.save_release(model, directory)
    original = np.load
    calls = []
    def numeric_load(*args, **kwargs):
        calls.append(kwargs.get('allow_pickle'))
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'load', numeric_load)
    restored = endpoint.load_release(directory, queries=q)
    pd.testing.assert_frame_equal(model.predict_from_history(q, s, e), restored.predict_from_history(q, s, e))
    assert calls == [False, False]
    assert restored.excluded_people == model.excluded_people
    assert all(path.stat().st_mode & 0o077 == 0 for path in directory.rglob('*'))
    with pytest.raises(FileExistsError): endpoint.save_release(model, directory)


def test_release_rejects_excluded_queries_before_numeric_load(records, model, tmp_path, monkeypatch):
    q, _, _ = records
    directory = tmp_path.resolve()/'release'
    endpoint.save_release(model, directory)
    def forbidden(*args, **kwargs): raise AssertionError('Numeric model must not be loaded')
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(history, 'load_release', forbidden)
    with pytest.raises(ValueError, match='Excluded participant'):
        endpoint.load_release(directory, queries=q.assign(subject_id='synthetic_excluded'))


def test_release_refuses_git_and_symlink_destinations(model, tmp_path):
    checkout = tmp_path.resolve()/'checkout'; checkout.mkdir(); (checkout/'.git').mkdir()
    with pytest.raises(ValueError): endpoint.save_release(model, checkout/'release')
    alias = tmp_path.resolve()/'alias'; alias.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(ValueError): endpoint.save_release(model, alias/'release')
    private = tmp_path.resolve()/'private'; private.mkdir()
    alias_private = checkout/'private_alias'; alias_private.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError): endpoint.save_release(model, alias_private/'release')
    assert not (private/'release').exists()


@pytest.mark.parametrize('kind', ['checksum', 'sources', 'traversal', 'missing_member', 'contract',
                                  'member_symlink', 'directory_symlink', 'manifest_symlink'])
def test_release_integrity_checks(model, tmp_path, kind):
    folder = tmp_path.resolve()/'release'
    endpoint.save_release(model, folder)
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if kind == 'checksum': (folder/'endpoint.json').write_text('{}')
    elif kind in ('sources', 'traversal', 'missing_member'):
        if kind == 'sources': manifest['sources']['endpoint_policy.py'] = 'changed'
        elif kind == 'traversal': manifest['files']['../outside'] = 'changed'
        else: manifest['files'].pop('endpoint.json')
        manifest_path.write_text(json.dumps(manifest))
    elif kind == 'contract':
        spec = json.loads((folder/'endpoint.json').read_text()); spec['unexpected'] = True
        (folder/'endpoint.json').write_text(json.dumps(spec))
        manifest['files']['endpoint.json'] = core._digest(folder/'endpoint.json')
        manifest_path.write_text(json.dumps(manifest))
    elif kind == 'directory_symlink':
        moved = folder.parent/'moved'; folder.rename(moved); folder.symlink_to(moved, target_is_directory=True)
    else:
        path = folder/('classifier' if kind == 'member_symlink' else 'manifest.json')
        moved = folder.parent/'moved'; path.rename(moved); path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError): endpoint.load_release(folder)
