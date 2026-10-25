"""Synthetic, runtime-generated checks; no private records or fitted weights."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from mindwatch.lightweight import (DOMAINS, FEATURES, FrozenLightweight,
                                  NumericHead, features_from_history,
                                  load_release, save_release)


@pytest.fixture
def histories():
    people = [f"synthetic_person_{i}" for i in range(4)]
    query = pd.DataFrame({"sample_id": [f"synthetic_query_{i}" for i in range(4)],
                          "subject_id": people,
                          "index_date": pd.Timestamp("2020-03-01 15:00")})
    surveys = pd.DataFrame([{"subject_id": person, "index_date": pd.Timestamp("2020-01-01"),
                             "phq9": 4+i} for i, person in enumerate(people)])
    ema = pd.DataFrame([{"subject_id": person, "date": pd.Timestamp("2020-03-01")-pd.Timedelta(days=lag),
                         **{d: float((i+j+lag) % 11) for j, d in enumerate(DOMAINS)}}
                        for i, person in enumerate(people) for lag in (20, 14, 7, 1)])
    return query, surveys, ema


@pytest.fixture
def frozen():
    rng = np.random.default_rng(9)
    def head():
        n = len(FEATURES)
        return NumericHead(FEATURES, np.ones(n, dtype=bool), rng.uniform(1, 4, n),
                           rng.uniform(1, 4, n), rng.uniform(1, 3, n), rng.normal(0, .1, n), .1)
    return FrozenLightweight(head(), head(), frozenset({"synthetic_person_0"}),
                             frozenset({"synthetic_person_1"}), "synthetic_model", "synthetic_inputs",
                             calibration_intercept=.2, threshold=.5, calibration_status="synthetic")


def test_day_boundary_window_and_last_valid_survey(histories):
    q, s, e = histories
    s = pd.concat([s, s.assign(index_date=pd.Timestamp("2020-02-01"), phq9=np.nan)], ignore_index=True)
    meta, x, provenance = features_from_history(q, s, e)
    assert tuple(x.columns) == FEATURES
    np.testing.assert_array_equal(x.h_previous_phq9, np.arange(4, 8))
    for i in range(len(q)):
        for j, d in enumerate(DOMAINS):
            assert x.loc[i, f"e_{d}_last"] == (i+j+1) % 11
            assert x.loc[i, f"e_{d}_median"] == np.median([(i+j+lag) % 11 for lag in (14, 7, 1)])
    assert provenance.latest_survey_input_date.eq(pd.Timestamp("2020-01-01")).all()
    assert provenance.latest_ema_input_date.lt(meta.index_date.dt.normalize()).all()


def test_future_and_same_day_records_cannot_change_inputs_or_predictions(histories, frozen):
    q, s, e = histories
    expected = frozen.predict_from_history(q, s, e)
    future_surveys = s.assign(index_date=pd.Timestamp("2020-03-01"), phq9=999)
    future_ema = e.assign(date=pd.Timestamp("2020-03-01"), **{d: 999 for d in DOMAINS})
    # Invalid values and duplicate future dates are irrelevant to earlier inputs.
    changed_s = pd.concat([s, future_surveys, future_surveys], ignore_index=True)
    changed_e = pd.concat([e, future_ema, future_ema], ignore_index=True)
    pd.testing.assert_frame_equal(expected, frozen.predict_from_history(q, changed_s, changed_e))


def test_repeated_dataframe_indices_do_not_break_linkage(histories, frozen):
    q, s, e = histories
    expected = frozen.predict_from_history(q, s, e)
    s.index = np.zeros(len(s), dtype=int)
    e.index = np.zeros(len(e), dtype=int)
    pd.testing.assert_frame_equal(expected, frozen.predict_from_history(q, s, e))


def test_numeric_probabilities_totals_and_exposure(histories, frozen):
    q, s, e = histories
    meta, x, _ = features_from_history(q, s, e)
    out = frozen.predict(meta, x)
    def independent_linear(head):
        result = []
        for _, row in x.iterrows():
            value = head.intercept
            for i, column in enumerate(head.features):
                raw_value = row[column]
                if not head.seen[i] or not np.isfinite(raw_value):
                    raw_value = head.median[i]
                standardized = min(12., max(-12., (raw_value-head.mean[i])/head.scale[i]))
                value += standardized*head.coef[i]
            result.append(value)
        return np.asarray(result)
    raw = expit(independent_linear(frozen.classifier))
    np.testing.assert_allclose(out.raw_prediction, raw)
    np.testing.assert_allclose(out.prediction, expit(logit(raw)+frozen.calibration_intercept))
    expected = np.clip(x.h_previous_phq9+independent_linear(frozen.change_regressor), 0, 27)
    np.testing.assert_allclose(out.predicted_phq9, expected)
    np.testing.assert_allclose(out.predicted_phq9_change, expected-x.h_previous_phq9)
    assert out.policy.tolist() == ["known", "calibration_only", "new", "new"]
    with pytest.raises(ValueError, match="Population-seen"):
        frozen.predict(meta, x, policy="new")
    with pytest.raises(ValueError, match="training membership"):
        frozen.predict(meta, x, policy="known")
    assert frozen.predict(meta.iloc[2:], x.iloc[2:], policy="new").policy.eq("new").all()


def test_missing_prior_and_missing_ema_keep_denominator(histories, frozen):
    q, s, e = histories
    out = frozen.predict_from_history(q, s.iloc[1:], e.iloc[:0])
    assert len(out) == len(q)
    assert out.loc[0, "status"] == "insufficient_data"
    assert out.loc[0, ["prediction", "predicted_phq9", "predicted_phq9_change", "threshold"]].isna().all()
    assert out.loc[1:, "status"].eq("ok_ema_missing_imputed").all()
    assert np.isfinite(out.loc[1:, "prediction"]).all()
    assert not out.ema_used.any()


def test_never_observed_training_channel_is_blocked(histories, frozen):
    q, s, e = histories
    meta, x, _ = features_from_history(q, s, e)
    mask = frozen.classifier.seen.copy()
    mask[-1] = False
    model = replace(frozen, classifier=replace(frozen.classifier, seen=mask),
                    change_regressor=replace(frozen.change_regressor, seen=mask))
    expected = model.predict(meta, x)
    x.iloc[:, -1] = 10
    pd.testing.assert_frame_equal(expected, model.predict(meta, x))


def _subset_head(head, indices):
    return NumericHead(tuple(head.features[i] for i in indices),
                       **{name: getattr(head, name)[indices] for name in ("seen", "median", "mean", "scale", "coef")},
                       intercept=head.intercept)


@pytest.mark.parametrize("kind", ["phq_only", "ema_unseen"])
def test_unused_ema_inputs_do_not_claim_model_use(histories, frozen, kind):
    q, s, e = histories
    if kind == "phq_only":
        classifier = _subset_head(frozen.classifier, [0])
        regressor = _subset_head(frozen.change_regressor, [0])
    else:
        seen = np.zeros(len(FEATURES), dtype=bool)
        seen[0] = True
        classifier = replace(frozen.classifier, seen=seen)
        regressor = replace(frozen.change_regressor, seen=seen)
    model = replace(frozen, classifier=classifier, change_regressor=regressor)
    out = model.predict_from_history(q, s, e)
    assert out.ema_observed_features.eq(len(FEATURES)-1).all()
    assert out.ema_retained_features.eq(0).all()
    assert out.ema_used_features.eq(0).all()
    assert not out.ema_used.any()
    assert out.used_modalities.eq("past_phq").all()
    assert out.status.eq("ok").all()
    assert out.fallback_reason.eq("").all()
    without_ema = model.predict_from_history(q, s, e.iloc[:0])
    np.testing.assert_array_equal(out.prediction, without_ema.prediction)
    np.testing.assert_array_equal(out.predicted_phq9, without_ema.predicted_phq9)
    assert without_ema.status.eq("ok").all()


@pytest.mark.parametrize("retained_head", ["classifier", "change_regressor"])
def test_ema_use_is_union_of_retained_seen_head_inputs(histories, frozen, retained_head):
    q, s, e = histories
    heads = {"classifier": _subset_head(frozen.classifier, [0]),
             "change_regressor": _subset_head(frozen.change_regressor, [0])}
    heads[retained_head] = _subset_head(getattr(frozen, retained_head), [0, 1])
    model = replace(frozen, **heads)
    out = model.predict_from_history(q, s, e)
    assert out.ema_retained_features.eq(1).all()
    assert out.ema_used_features.eq(1).all()
    assert out.ema_used.all()
    assert out.used_modalities.eq("past_phq+ema").all()
    e[DOMAINS[0]] = np.nan
    missing = model.predict_from_history(q, s, e)
    assert missing.ema_observed_features.gt(0).all()
    assert not missing.ema_used.any()
    assert missing.status.eq("ok_ema_missing_imputed").all()
    assert missing.used_modalities.eq("past_phq").all()


@pytest.mark.parametrize("case", ["target_query", "target_feature", "misaligned", "column_order", "infinite", "fractional_phq", "ema_range", "duplicate_query"])
def test_invalid_prepared_inputs_rejected(histories, frozen, case):
    q, s, e = histories
    meta, x, _ = features_from_history(q, s, e)
    if case == "target_query":
        meta["phq9"] = 0
    elif case == "target_feature":
        x["current_phq9"] = 0
    elif case == "misaligned":
        x.index = x.index+1
    elif case == "column_order":
        x = x[x.columns[::-1]]
    elif case == "infinite":
        x.iloc[0, 1] = np.inf
    elif case == "fractional_phq":
        x.iloc[0, 0] = 1.5
    elif case == "ema_range":
        x.iloc[0, 1] = 11
    else:
        meta.loc[1, "sample_id"] = meta.loc[0, "sample_id"]
    with pytest.raises(ValueError):
        frozen.predict(meta, x)


@pytest.mark.parametrize("case", ["duplicate", "invalid_score", "timezone", "numeric_key"])
def test_invalid_history_rejected(histories, case):
    q, s, e = histories
    if case == "duplicate":
        e = pd.concat([e, e.iloc[-1:]])
    elif case == "invalid_score":
        s.loc[0, "phq9"] = 28
    elif case == "timezone":
        e["date"] = e.date.dt.tz_localize("UTC")
    else:
        e["subject_id"] = np.arange(len(e))
    with pytest.raises(ValueError):
        features_from_history(q, s, e)


def test_empty_query_supported(histories, frozen):
    q, s, e = histories
    assert frozen.predict_from_history(q.iloc[:0], s, e).empty


def test_zero_calibration_preserves_saturated_raw_probability(histories, frozen):
    q, s, e = histories
    model = replace(frozen, classifier=replace(frozen.classifier, intercept=1000.), calibration_intercept=0.)
    out = model.predict_from_history(q, s, e)
    np.testing.assert_array_equal(out.prediction, out.raw_prediction)


def test_membership_requires_a_collection(frozen):
    with pytest.raises(ValueError, match="collection"):
        replace(frozen, training_people="synthetic_person_0")


@pytest.mark.parametrize("case", ["shape", "scale", "finite", "seen", "features"])
def test_invalid_numeric_parameters_rejected(frozen, case):
    updates = {"shape": {"coef": np.zeros(1)}, "scale": {"scale": np.zeros(len(FEATURES))},
               "finite": {"mean": np.full(len(FEATURES), np.nan)},
               "seen": {"seen": np.ones(len(FEATURES))}, "features": {"features": ("target",)}}
    with pytest.raises(ValueError):
        replace(frozen.classifier, **updates[case])


def test_private_numeric_release_roundtrip(histories, frozen, tmp_path):
    q, s, e = histories
    folder = tmp_path/"private_release"
    save_release(frozen, folder)
    loaded = load_release(folder)
    pd.testing.assert_frame_equal(frozen.predict_from_history(q, s, e), loaded.predict_from_history(q, s, e))
    assert (folder.stat().st_mode & 0o077) == 0
    assert all((p.stat().st_mode & 0o077) == 0 for p in folder.iterdir())
    with pytest.raises(FileExistsError):
        save_release(frozen, folder)


@pytest.mark.parametrize("case", ["checksum", "source", "traversal", "symlink"])
def test_release_corruption_fails_closed(frozen, tmp_path, case):
    folder = tmp_path/"release"
    save_release(frozen, folder)
    if case == "checksum":
        (folder/"model.json").write_text("{}")
    elif case == "symlink":
        path = folder/"model.json"
        contents = path.read_bytes()
        path.unlink()
        target = tmp_path/"redirect.json"
        target.write_bytes(contents)
        path.symlink_to(target)
    else:
        manifest = json.loads((folder/"manifest.json").read_text())
        if case == "source":
            manifest["source_sha256"] = "changed"
        else:
            manifest["files"]["../outside"] = "changed"
        (folder/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_release(folder)


def test_fitted_arrays_never_saved_under_git(frozen, tmp_path):
    root = tmp_path/"checkout"
    root.mkdir()
    (root/".git").mkdir()
    with pytest.raises(ValueError, match="outside every Git"):
        save_release(frozen, root/"release")
    assert not (root/"release").exists()


def test_reliability_is_descriptive_and_not_a_predictor(histories, frozen):
    q, s, e = histories
    full = frozen.predict_from_history(q, s, e)
    assert full.input_provenance_status.eq('history_derived').all()
    assert full.prior_phq_age_days.eq(60).all()
    assert full.prior_phq_age_band.eq('within_90_days').all()
    assert full.ema_observed_days_14d.eq(3).all()
    assert full.ema_observed_domains_14d.eq(4).all()
    assert full.ema_last_age_days_14d.eq(1).all()
    assert full.ema_history_status.eq('all_domains').all()
    for domain in DOMAINS:
        assert full[f'{domain}_observed_days_14d'].eq(3).all()
    shifted = s.assign(index_date=pd.Timestamp('2019-01-01'))
    old = frozen.predict_from_history(q, shifted, e)
    assert old.prior_phq_age_band.eq('over_180_days').all()
    np.testing.assert_array_equal(full.prediction, old.prediction)
    np.testing.assert_array_equal(full.predicted_phq9, old.predicted_phq9)
    assert full.input_reliability_scope.eq('descriptive_availability_not_validated_confidence').all()


def test_prepared_features_do_not_invent_reliability(histories, frozen):
    q, s, e = histories
    meta, x, _ = features_from_history(q, s, e)
    out = frozen.predict(meta, x)
    assert out.input_provenance_status.eq('not_provided').all()
    assert out.prior_phq_age_band.eq('unknown').all()
    assert out.ema_history_status.eq('unknown').all()
    assert out[['prior_phq_age_days', 'ema_observed_days_14d',
                'ema_observed_domains_14d', 'ema_last_age_days_14d']].isna().all().all()


def test_partial_domains_and_missing_histories_are_explicit(histories, frozen):
    q, s, e = histories
    partial = e.copy()
    partial[list(DOMAINS[1:])] = np.nan
    out = frozen.predict_from_history(q, s, partial)
    assert out.ema_observed_domains_14d.eq(1).all()
    assert out.ema_history_status.eq('partial_domains').all()
    missing = frozen.predict_from_history(q, s.iloc[:0], e.iloc[:0])
    assert missing.prior_phq_age_days.isna().all()
    assert missing.prior_phq_age_band.eq('missing_prior_phq').all()
    assert missing.ema_observed_days_14d.eq(0).all()
    assert missing.ema_last_age_days_14d.isna().all()
    assert missing.ema_history_status.eq('none_in_window').all()


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, True, 1j, '1'])
def test_calibration_slope_must_be_positive_finite_real(frozen, value):
    with pytest.raises(ValueError, match='calibration slope'):
        replace(frozen, calibration_slope=value)


def test_slope_calibration_and_roundtrip(histories, frozen, tmp_path):
    q, s, e = histories
    model = replace(frozen, calibration_slope=.7)
    out = model.predict_from_history(q, s, e)
    expected = expit(.7*logit(np.clip(out.raw_prediction, 1e-7, 1-1e-7))+.2)
    np.testing.assert_allclose(out.prediction, expected, rtol=0, atol=1e-15)
    path = tmp_path/'slope_release'
    save_release(model, path)
    restored = load_release(path)
    assert restored.calibration_slope == .7
    pd.testing.assert_frame_equal(out, restored.predict_from_history(q, s, e))
