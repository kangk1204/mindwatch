"""Runtime-generated synthetic tests only; no study records or fitted artifacts."""
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from mindwatch.features import (DOMAINS, FEATURE_COLUMNS, H_COLUMNS, build_dataset,
                               build_features, ema_forecasts, history_features)
from mindwatch.models import calibrate, fit_head, person_weights
from mindwatch.pipeline import (evaluate_frozen, load_frozen, predict_frozen,
                                private_output, train_and_freeze)
from mindwatch.validation import assert_disjoint, assign_roles, classification_metrics


def toy(prefix, people=12):
    visits, events = [], []
    for person in range(people):
        sid = f"synthetic_{prefix}_{person}"
        for wave, date in enumerate(pd.date_range("2020-01-01", periods=4, freq="120D")):
            score = 6+8*(person % 2)+((wave+person) % 3)-1
            visits.append(dict(subject_id=sid, index_date=date, phq9=score,
                               gad7=person % 8+wave, isi=person % 10+wave))
            for lag in (5, 12):
                events.append(dict(subject_id=sid, date=date-pd.Timedelta(days=lag),
                                   **{d: float((person+wave+lag+i) % 11) for i, d in enumerate(DOMAINS)}))
    return pd.DataFrame(visits), pd.DataFrame(events)


@pytest.fixture
def sample():
    return toy("features")


@pytest.fixture
def frozen(tmp_path):
    tr, te = toy("train")
    va, ve = toy("selection")
    ca, ce = toy("calibration")
    tests, ee = toy("test")
    events = pd.concat([te, ve, ce, ee], ignore_index=True)
    folder = tmp_path/"private_run"
    train_and_freeze(tr, va, ca, events, folder)
    return folder, tests, events, tr


def test_four_visits_yield_three_samples(sample):
    v, e = sample
    m, x = build_dataset(v, e)
    assert m.groupby("subject_id").size().eq(3).all()
    assert tuple(x) == FEATURE_COLUMNS and len(H_COLUMNS) == 22
    assert m.latest_survey_input_date.lt(m.index_date).all()
    assert m.latest_EMA_input_date.lt(m.index_date).all()


def test_current_and_future_scores_do_not_change_earlier_features(sample):
    v, e = sample
    cutoff = sorted(v.index_date.unique())[1]
    m, x = build_features(v, e)
    changed = v.copy()
    changed.loc[changed.index_date >= cutoff, ["phq9", "gad7", "isi"]] = [26, 20, 27]
    mm, xx = build_features(changed, e)
    pd.testing.assert_frame_equal(x[m.index_date <= cutoff], xx[mm.index_date <= cutoff])


def test_future_deleted_or_changed_ema_invariant(sample):
    v, e = sample
    cutoff = sorted(v.index_date.unique())[1]
    m, x = build_features(v, e)
    changed = e.copy()
    changed.loc[changed.date >= cutoff, list(DOMAINS)] = 10.
    for new in (changed, e[e.date < cutoff]):
        mm, xx = build_features(v, new)
        pd.testing.assert_frame_equal(x[m.index_date <= cutoff], xx[mm.index_date <= cutoff])


def test_same_day_ema_excluded(sample):
    v, e = sample
    same = v[["subject_id", "index_date"]].rename(columns={"index_date": "date"})
    for d in DOMAINS:
        same[d] = 10.
    _, before = build_features(v, e)
    _, after = build_features(v, pd.concat([e, same], ignore_index=True))
    pd.testing.assert_frame_equal(before, after)


def test_first_visit_defines_eligibility(sample):
    v, e = sample
    sid = v.subject_id.iloc[0]
    first = v[v.subject_id == sid].index_date.min()
    v.loc[(v.subject_id == sid) & (v.index_date == first), "phq9"] = 4
    v.loc[(v.subject_id == sid) & (v.index_date > first), "phq9"] = 26
    meta, _ = build_dataset(v, e)
    assert sid not in set(meta.subject_id)


@pytest.mark.parametrize("kind", ["duplicate_visit", "duplicate_ema", "time_of_day", "timezone", "range", "fractional_score", "inf"])
def test_invalid_inputs_rejected(sample, kind):
    v, e = sample
    if kind == "duplicate_visit":
        v = pd.concat([v, v.iloc[:1]], ignore_index=True)
    elif kind == "duplicate_ema":
        e = pd.concat([e, e.iloc[:1]], ignore_index=True)
    elif kind == "time_of_day":
        v.loc[0, "index_date"] += pd.Timedelta(hours=1)
    elif kind == "timezone":
        v["index_date"] = v.index_date.dt.tz_localize("UTC")
    elif kind == "range":
        v.loc[0, "phq9"] = 30
    elif kind == "fractional_score":
        v["phq9"] = v.phq9.astype(float)
        v.loc[0, "phq9"] = 5.5
    else:
        e.loc[0, DOMAINS[0]] = np.inf
    with pytest.raises(ValueError):
        build_dataset(v, e)


def test_unlabeled_query_supported(sample):
    v, e = sample
    last = v.index_date.max()
    _, expected = build_features(v, e)
    v.loc[v.index_date == last, ["phq9", "gad7", "isi"]] = np.nan
    _, got = build_features(v, e)
    pd.testing.assert_frame_equal(expected, got)


def test_subject_roles_and_seed(sample):
    v, _ = sample
    b = v.groupby("subject_id").head(1)[["subject_id", "phq9"]].rename(columns={"phq9": "baseline_phq9"})
    one = assign_roles(b)
    two = assign_roles(b.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(one.sort_values("subject_id").reset_index(drop=True),
                                  two.sort_values("subject_id").reset_index(drop=True))
    groups = [g.subject_id for _, g in one.groupby("role")]
    assert_disjoint(*groups)
    with pytest.raises(ValueError):
        assign_roles(b.assign(future_target=1))
    with pytest.raises(ValueError):
        assert_disjoint(["same"], ["same"])


def test_equal_total_person_weight():
    ids = pd.Series(["synthetic_a"]*3+["synthetic_b"])
    w = pd.Series(person_weights(ids))
    assert w.groupby(ids).sum().nunique() == 1


@pytest.mark.parametrize("recipe", ["H", "compact"])
def test_fit_only_transform_and_unknown_input_block(sample, tmp_path, recipe):
    v, e = sample
    m, x = build_dataset(v, e)
    x.loc[:, "h_previous_isi"] = np.nan
    head = fit_head(x, m.target, m.subject_id, recipe)
    before = head.predict(x)
    changed = x.copy()
    changed["h_previous_isi"] = 28.
    np.testing.assert_array_equal(before, head.predict(changed))
    with pytest.raises(ValueError):
        head.predict(x.assign(target=m.target))
    with pytest.raises(ValueError):
        head.predict(x[list(reversed(x.columns))])
    path = tmp_path/"trusted.joblib"
    joblib.dump(head, path)
    np.testing.assert_array_equal(before, joblib.load(path).predict(x))


def test_calibration_support_and_threshold():
    y = np.tile([0, 1], 20)
    p = np.linspace(.1, .9, len(y))
    ids = np.array([f"synthetic_cal_{i}" for i in range(len(y))])
    ca = calibrate(y, p, ids)
    assert ca.status == "intercept_calibrated"
    assert np.mean(ca.predict(p)[y == 1] >= ca.threshold) >= .8
    assert calibrate(y[:2], p[:2], ids[:2]).status == "insufficient_calibration_support"


def test_repeated_rows_do_not_satisfy_core_calibration_person_threshold():
    y=np.repeat([0.,1.],30);p=np.repeat([.2,.8],30)
    ids=np.repeat(['synthetic_negative','synthetic_positive'],30)
    result=calibrate(y,p,ids)
    assert result.status=='insufficient_calibration_support'
    np.testing.assert_array_equal(result.predict(p),p)


@pytest.mark.parametrize('overlap',[('train','selection'),('train','calibration'),('selection','calibration')])
def test_training_rejects_every_development_role_overlap(tmp_path,overlap):
    pairs={role:toy(role) for role in ['train','selection','calibration']}
    source,target=overlap
    pairs[target]=(pairs[source][0].copy(),pairs[source][1].copy())
    events=pd.concat([value[1] for value in pairs.values()]).drop_duplicates(['subject_id','date'])
    with pytest.raises(ValueError,match='Participant overlap'):
        train_and_freeze(pairs['train'][0],pairs['selection'][0],pairs['calibration'][0],events,tmp_path/'private_overlap')
    assert not (tmp_path/'private_overlap').exists()


def test_compact_fit_discards_sparse_and_constant_features():
    from mindwatch.features import E_COLUMNS
    n=40;x=pd.DataFrame(3.,index=range(n),columns=FEATURE_COLUMNS)
    x['h_previous_phq9']=np.arange(n)%20
    x[E_COLUMNS[0]]=np.nan
    x.loc[:9,E_COLUMNS[0]]=np.arange(10)
    x[E_COLUMNS[1]]=3.
    model=fit_head(x,np.arange(n)%2,['synthetic_'+str(i) for i in range(n)],'compact')
    assert model.columns==('h_previous_phq9',)


def test_freeze_then_fixed_evaluation_and_disjointness(frozen):
    folder, test, ema, train = frozen
    locked = (folder/"models.joblib").read_bytes()
    metrics = evaluate_frozen(folder, test, ema)
    assert len(metrics) == 5 and sum(z["selected_before_test"] for z in metrics) == 1
    assert locked == (folder/"models.joblib").read_bytes()
    assert (folder/"models.joblib").stat().st_mode & 0o777 == 0o600
    assert json.loads((folder/"freeze.json").read_text())["test_used_for_selection"] is False
    with pytest.raises(ValueError, match="overlap"):
        evaluate_frozen(folder, train, ema)
    with pytest.raises(FileExistsError):
        evaluate_frozen(folder, test, ema)


def test_prediction_invariance_without_current_labels(frozen):
    folder, visits, ema, _ = frozen
    cutoff = visits.index_date.max()
    expected = predict_frozen(folder, visits, ema)
    visits.loc[visits.index_date == cutoff, ["phq9", "gad7", "isi"]] = np.nan
    got = predict_frozen(folder, visits, ema)
    np.testing.assert_array_equal(expected.prediction, got.prediction)


def test_tampered_model_refused(frozen):
    folder, *_ = frozen
    with (folder/"models.joblib").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_frozen(folder)


def test_git_output_and_symlink_output_refused(tmp_path):
    repo = tmp_path/"other_repo"
    (repo/".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="Git"):
        private_output(repo/"artifacts")
    link = tmp_path/"link"
    link.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="Git"):
        private_output(link/"artifacts")


def test_p0_uses_only_past_real_responses(sample):
    _, e = sample
    original = ema_forecasts(e, DOMAINS[0])
    cut = sorted(e.date.unique())[3]
    changed = e.copy()
    changed.loc[changed.date >= cut, DOMAINS[0]] = 10.
    new = ema_forecasts(changed, DOMAINS[0])
    pd.testing.assert_frame_equal(original.loc[original.date <= cut, ["last", "P0"]],
                                  new.loc[new.date <= cut, ["last", "P0"]])


def test_metrics_agree_with_direct_weighted_formula():
    ids = ["synthetic_a", "synthetic_a", "synthetic_b", "synthetic_c"]
    y, p = np.array([0, 1, 1, 0]), np.array([.1, .7, .8, .3])
    out = classification_metrics(y, p, ids)
    assert out["Brier"] == pytest.approx(np.average((y-p)**2, weights=person_weights(ids)))
    assert out["AUROC"] == 1.


def test_cli_separate_train_evaluate(tmp_path):
    events = []
    for role in ("train", "selection", "calibration", "test"):
        visits, ema = toy("cli_"+role, people=8)
        visits.to_csv(tmp_path/f"{role}.csv", index=False)
        events.append(ema)
    pd.concat(events).to_csv(tmp_path/"ema.csv", index=False)
    prefix = [sys.executable, "-m", "mindwatch.cli"]
    run = tmp_path/"cli_run"
    args = [str(a) for role in ("train", "selection", "calibration")
            for a in (f"--{role}-visits", tmp_path/f"{role}.csv")]
    subprocess.run(prefix+["train", *args, "--ema", str(tmp_path/"ema.csv"), "--output", str(run)], check=True, capture_output=True)
    subprocess.run(prefix+["evaluate", "--test-visits", str(tmp_path/"test.csv"),
                           "--ema", str(tmp_path/"ema.csv"), "--run", str(run)], check=True, capture_output=True)
    assert (run/"evaluation.json").exists()
