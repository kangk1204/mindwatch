"""Synthetic-only numeric export, routing and causal history tests."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline

from mindwatch import lightweight as core
from mindwatch import lightweight_policy as policy


@pytest.fixture
def records():
    people = [f'synthetic_{i}' for i in range(4)]
    query = pd.DataFrame({'subject_id': people, 'sample_id': [f'query_{i}' for i in range(4)],
                         'index_date': pd.Timestamp('2020-03-01 15:00')})
    surveys = pd.DataFrame([{'subject_id': person, 'index_date': pd.Timestamp(date),
        'phq9': float(7+i+j), 'gad7': float(3+i+j), 'isi': float(9+i+j)}
        for i, person in enumerate(people) for j, date in enumerate(('2020-01-01','2020-02-01'))])
    ema = pd.DataFrame([{'subject_id': person, 'date': pd.Timestamp('2020-02-29'),
        **{domain: float(i+1) for domain in core.DOMAINS}} for i, person in enumerate(people[:2])])
    return query, surveys, ema


@pytest.fixture
def model():
    n = len(core.FEATURES)
    head = core.NumericHead(core.FEATURES, np.ones(n, bool), np.ones(n), np.zeros(n),
                            np.ones(n), np.ones(n)*.01, -.5)
    primary = core.FrozenLightweight(head, head, frozenset({'synthetic_0'}), frozenset(),
        'synthetic', 'synthetic', calibration_intercept=.2, threshold=.4, calibration_slope=.8)
    names = ('h_previous_phq9','h_previous_gad7','h_previous_isi')
    fallback = policy.NumericHistoryHead(names, np.ones(3, bool), np.ones(3), (1, 2),
        np.zeros(5), np.ones(5), np.array([.1,.02,.03,.2,-.1]), -.3)
    return policy.FrozenHistoryPolicy(primary, fallback, fallback, 'synthetic_survey')


def test_routing_only_on_complete_ema_absence(records, model):
    q, s, e = records
    meta, x, _ = policy.features_from_history(q, s, e)
    out = model.predict(meta, x)
    expected = model.primary.predict(meta, x.loc[:, list(core.FEATURES)])
    np.testing.assert_array_equal(out.prediction.iloc[:2], expected.prediction.iloc[:2])
    np.testing.assert_array_equal(out.predicted_phq9.iloc[:2], expected.predicted_phq9.iloc[:2])
    raw = expit(model.fallback_classifier.linear(x.iloc[2:]))
    np.testing.assert_allclose(out.raw_prediction.iloc[2:], raw)
    np.testing.assert_array_equal(out.threshold, expected.threshold)
    assert out.history_fallback_used.tolist() == [False, False, True, True]
    assert out.inference_route.tolist() == ['compact','compact','synthetic_survey','synthetic_survey']
    assert out.used_modalities.iloc[2:].eq('past_surveys').all()
    assert out.status.iloc[2:].eq('ok_history_fallback').all()
    assert out.ema_used.iloc[2:].eq(False).all()
    assert out.ema_retained_features.iloc[2:].eq(0).all()
    # One observed EMA domain still takes the compact route.
    e.loc[:, list(core.DOMAINS[1:])] = np.nan
    partial = model.predict_from_history(q, s, e)
    assert partial.history_fallback_used.tolist() == [False, False, True, True]


def test_raw_history_strict_past_and_optional_auxiliary_scales(records, model):
    q, s, e = records
    expected = model.predict_from_history(q, s, e)
    future = s.assign(index_date=pd.Timestamp('2020-03-01'), phq9=999, gad7=999, isi=999)
    changed = pd.concat([s, future, future], ignore_index=True)
    pd.testing.assert_frame_equal(expected, model.predict_from_history(q, changed, e))
    no_aux = model.predict_from_history(q, s.drop(columns=['gad7','isi']), e)
    assert no_aux.past_gad7_available.eq(False).all()
    assert no_aux.past_isi_available.eq(False).all()
    assert no_aux.fallback_auxiliary_imputation.tolist() == [False, False, True, True]
    np.testing.assert_array_equal(no_aux.prediction.iloc[:2], expected.prediction.iloc[:2])
    assert np.isfinite(no_aux.prediction).all()


def test_missing_prior_and_prepared_schema(records, model):
    q, s, e = records
    absent = model.predict_from_history(q, s.iloc[:0], e.iloc[:0])
    assert absent.prediction.isna().all()
    assert absent.inference_route.eq('none').all()
    assert not absent.history_fallback_used.any()
    meta, compact, _ = core.features_from_history(q, s, e)
    with pytest.raises(ValueError, match='Full ordered'):
        model.predict(meta, compact)
    with pytest.raises(ValueError, match='Current outcomes'):
        model.predict_from_history(q.assign(phq9=10), s, e)


def test_latest_auxiliary_survey_provenance_distinguishes_phq_age(records,model):
    q,s,e = records
    s.loc[s.index_date.eq(pd.Timestamp('2020-02-01')),'phq9'] = np.nan
    out = model.predict_from_history(q,s,e)
    assert out.prior_phq_age_days.eq(60).all()
    assert out.latest_phq9_input_date.eq(pd.Timestamp('2020-01-01')).all()
    assert out.latest_gad7_input_date.eq(pd.Timestamp('2020-02-01')).all()
    assert out.latest_survey_input_date.iloc[:2].eq(pd.Timestamp('2020-01-01')).all()
    assert out.latest_survey_input_date.iloc[2:].eq(pd.Timestamp('2020-02-01')).all()
    meta,x,_ = policy.features_from_history(q,s,e)
    prepared = model.predict(meta,x)
    assert prepared.latest_phq9_input_date.isna().all()
    assert prepared.latest_gad7_input_date.isna().all()


def test_empty_queries_preserve_policy_schema(records,model):
    q,s,e = records
    meta,x,provenance=policy.features_from_history(q.iloc[:0],s,e)
    assert x.empty and tuple(x.columns) == policy.FEATURE_COLUMNS
    assert 'latest_gad7_input_date' in provenance
    out=model.predict_from_history(q.iloc[:0],s,e)
    assert out.empty and 'inference_route' in out


@pytest.mark.parametrize('scale,maximum', [('gad7',21),('isi',28)])
@pytest.mark.parametrize('value', [-1, 29, 1.5, np.inf, -np.inf, 'invalid'])
def test_invalid_past_auxiliary_scores_rejected(records, model, scale, maximum, value):
    q,s,e = records
    s[scale] = s[scale].astype(object)
    s.loc[0,scale] = value
    with pytest.raises((ValueError, TypeError)):
        model.predict_from_history(q,s,e)


@pytest.mark.parametrize('scale,maximum', [('gad7',21),('isi',28)])
def test_auxiliary_bounds_missing_and_future_are_causal(records, model, scale, maximum):
    q,s,e = records
    s[scale] = s[scale].astype(object)
    s.loc[0,scale] = 0
    s.loc[1,scale] = maximum
    s.loc[2,scale] = np.nan
    expected = model.predict_from_history(q,s,e)
    future = s.iloc[:2].assign(index_date=pd.Timestamp('2020-03-01 10:00'))
    future[scale] = ['invalid', np.inf]
    pd.testing.assert_frame_equal(expected,model.predict_from_history(q,pd.concat([s,future],ignore_index=True),e))
    s.loc[1,scale] = maximum+1
    with pytest.raises(ValueError): model.predict_from_history(q,s,e)


def _export(pipeline, names, seen):
    imputer, scaler, estimator = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
    return policy.NumericHistoryHead(names, seen, imputer.statistics_, tuple(imputer.indicator_.features_),
        scaler.mean_, scaler.scale_, np.asarray(estimator.coef_).reshape(-1), float(np.asarray(estimator.intercept_).reshape(-1)[0]))


def test_numeric_history_export_matches_sklearn_without_clipping():
    rng = np.random.default_rng(17)
    names = ('h_previous_phq9','h_previous_gad7','h_previous_isi')
    x = pd.DataFrame(rng.uniform(1, 10, (80,3)), columns=names)
    x.iloc[::4,1] = np.nan
    x.iloc[:,2] = np.nan
    seen = np.isfinite(x.to_numpy()).any(0)
    y = np.arange(len(x)) % 2
    query = x.copy()
    query.iloc[:,2] = 100.
    # The frozen observed mask must block the newly observed third channel.
    blocked = query.copy(); blocked.iloc[:,2] = np.nan
    for estimator in (LogisticRegression(C=.01, solver='lbfgs'), Ridge(alpha=10)):
        fit = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                            StandardScaler(), estimator).fit(x.to_numpy(), y)
        numeric = _export(fit, names, seen)
        if isinstance(estimator, LogisticRegression):
            expected = fit.predict_proba(blocked.to_numpy())[:,1]
            actual = expit(numeric.linear(query))
        else:
            expected = fit.predict(blocked.to_numpy())
            actual = numeric.linear(query)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    # A large deviation is intentionally not subjected to the compact clip.
    head = policy.NumericHistoryHead(names[:1], np.ones(1,bool), np.zeros(1), (),
                                    np.zeros(1), np.ones(1), np.ones(1), 0.)
    assert head.linear(pd.DataFrame({names[0]: [100.]}))[0] == 100.


def test_policy_roundtrip_and_no_public_artifacts(records, model, tmp_path):
    q, s, e = records
    directory = tmp_path/'private_release'
    policy.save_release(model, directory)
    restored = policy.load_release(directory)
    pd.testing.assert_frame_equal(model.predict_from_history(q,s,e), restored.predict_from_history(q,s,e))
    assert all(path.stat().st_mode & 0o077 == 0 for path in directory.rglob('*'))
    with pytest.raises(FileExistsError): policy.save_release(model, directory)
    checkout = tmp_path/'checkout'; checkout.mkdir(); (checkout/'.git').mkdir()
    with pytest.raises(ValueError): policy.save_release(model, checkout/'weights')


@pytest.mark.parametrize('kind', ['checksum', 'source', 'primary_symlink', 'traversal'])
def test_policy_tamper_rejected(model, tmp_path, kind):
    folder = tmp_path/'release'
    policy.save_release(model, folder)
    if kind == 'checksum': (folder/'policy.json').write_text('{}')
    elif kind == 'primary_symlink':
        original = folder/'primary'; target = tmp_path/'moved'; original.rename(target); original.symlink_to(target)
    else:
        path = folder/'manifest.json'; manifest = json.loads(path.read_text())
        if kind == 'source': manifest['source_sha256'] = 'changed'
        else: manifest['files']['../outside'] = 'changed'
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): policy.load_release(folder)


@pytest.mark.parametrize('kind', ['indicator', 'scale', 'shape', 'features'])
def test_history_numeric_contract(model, kind):
    updates = {'indicator': {'indicator_indices': (99,)}, 'scale': {'scale': np.zeros(5)},
               'shape': {'coef': np.zeros(1)}, 'features': {'features': ('target',)}}
    with pytest.raises(ValueError): replace(model.fallback_classifier, **updates[kind])
