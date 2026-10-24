"""Synthetic checks; no study outcomes or reserved participants are loaded."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.state_model import (CausalStateSpace, DOMAINS, DomainParameters,
                         _innovation_loss, _panel, _propagate, _update)


def events(n=12, days=10):
    rng=np.random.default_rng(2026)
    rows=[]
    for i in range(n):
        level=rng.uniform(2,7)
        for j in range(days):
            rows.append({"subject_id":f"p{i}","date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=7*j+i%3),
                **{d:float(np.clip(level+.3*np.sin(j)+rng.normal(0,.4),0,10)) for d in DOMAINS}})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("personalized",[False,True])
def test_actual_boundary_before_numeric_validation_and_fit_isolation(personalized):
    e=events();cutoff=pd.Timestamp("2025-02-15");allowed={f"p{i}" for i in range(8)}
    model=CausalStateSpace(personalized,maxiter=20).fit(e,allowed,cutoff)
    changed=e.copy()
    forbidden=~changed.subject_id.isin(allowed)|changed.date.ge(cutoff)
    changed.loc[forbidden,list(DOMAINS)]=99999
    alternate=CausalStateSpace(personalized,maxiter=20).fit(changed,allowed,cutoff)
    assert model.parameters_==alternate.parameters_
    assert model.fit_content_sha256_==alternate.fit_content_sha256_
    assert model.training_subjects_<=allowed
    assert pd.Timestamp(model.diagnostics_["fit_latest_date"])<cutoff


def test_target_day_and_future_ema_cannot_change_prediction():
    e=events();model=CausalStateSpace(True,maxiter=15).fit(e,{"p0","p1","p2"},"2025-02-01")
    anchor=pd.Timestamp("2025-02-12 15:00")
    meta=pd.DataFrame({"subject_id":["p8"],"index_date":[anchor]},index=[100])
    baseline=model.transform(meta,e)
    extra={"subject_id":"p8","date":anchor.normalize(),**{d:10. for d in DOMAINS}}
    mutation=pd.concat([e,pd.DataFrame([extra])],ignore_index=True)
    mutation.loc[mutation.date.ge(anchor.normalize()),list(DOMAINS)]=99999
    pd.testing.assert_frame_equal(baseline,model.transform(meta,mutation))
    pd.testing.assert_frame_equal(baseline,model.transform(meta,e[e.date.lt(anchor.normalize())]))


def test_unknown_user_past_is_used_without_population_refit():
    e=events();model=CausalStateSpace(True,maxiter=20).fit(e,{"p0","p1","p2","p3"},"2025-04-01")
    before=pickle.dumps(model)
    original=model.forecast_one("p10","2025-04-01",e)
    changed=e.copy();changed.loc[changed.subject_id.eq("p10"),list(DOMAINS)]=0
    alternate=model.forecast_one("p10","2025-04-01",changed)
    assert original[DOMAINS[0]]["level"] != alternate[DOMAINS[0]]["level"]
    assert pickle.dumps(model)==before
    assert "p10" not in model.training_subjects_


@pytest.mark.parametrize("personalized",[False,True])
def test_no_history_and_unobserved_training_channel(personalized):
    e=events();e[DOMAINS[-1]]=np.nan
    model=CausalStateSpace(personalized,maxiter=20).fit(e,set(e.subject_id),"2025-04-01")
    x=e.copy();x[DOMAINS[-1]]=10
    meta=pd.DataFrame({"subject_id":["missing","p0"],"index_date":pd.to_datetime(["2025-04-01"]*2)})
    result=model.transform(meta,x)
    assert result.iloc[0].isna().all()
    assert result.filter(like=DOMAINS[-1]).isna().all().all()
    assert np.isfinite(result.iloc[1,:-3]).all()


def test_constant_channel_is_stable_and_marked():
    e=events();e[DOMAINS[0]]=3.
    model=CausalStateSpace(True,maxiter=20).fit(e,set(e.subject_id),"2025-04-01")
    state=model.forecast_one("p0","2025-04-01",e)[DOMAINS[0]]
    assert state["level"]==3 and state["deviation"]==0 and state["uncertainty"]==0
    assert state["status"]=="constant_training_channel"


def test_independent_scalar_update_and_joseph_covariance():
    p=DomainParameters(4.,14.,2.,3.,0.,1,2,0,True,0,None,"fixed_test")
    mean=np.array([0.,0.]);covariance=np.diag([2.,0.])
    m,c,v,s=_update(mean,covariance,7.,p)
    assert v==3 and s==5
    np.testing.assert_allclose(m,[1.2,0.],rtol=0,atol=1e-14)
    np.testing.assert_allclose(c,np.diag([1.2,0.]),rtol=0,atol=1e-14)
    mm,cc=_propagate(m,c,14.,p)
    np.testing.assert_allclose(mm,[.6,0.],rtol=0,atol=1e-14)
    np.testing.assert_allclose(cc,np.diag([1.8,0.]),rtol=0,atol=1e-14)


def test_random_intercept_is_shrunk_and_covariance_positive():
    e=events();e.loc[e.subject_id.eq("p0"),list(DOMAINS)]=9.
    m=CausalStateSpace(True,maxiter=25).fit(e,set(e.subject_id),"2025-04-01")
    state=m.forecast_one("p0","2026-04-01",e)[DOMAINS[0]]
    p=m.parameters_[DOMAINS[0]]
    assert p.mean<state["baseline"]<9.
    assert abs(state["deviation"])<.2
    assert np.linalg.eigvalsh(state["covariance"]).min()>=-1e-12


def test_vectorized_likelihood_matches_independent_recursion():
    e=events(n=4,days=5);domain=DOMAINS[0];panel=_panel(e,domain)
    p=DomainParameters(4.,12.,1.5,.7,2.1,4,20,4,True,0,None,"fixed_test")
    losses=[]
    for _,g in e.groupby("subject_id"):
        m=np.zeros(2);c=np.diag([p.dynamic_variance,p.intercept_variance]);prev=g.date.iloc[0];ll=[]
        for row in g.itertuples():
            m,c=_propagate(m,c,(row.date-prev).days,p)
            m,c,res,var=_update(m,c,getattr(row,domain),p)
            ll.append(.5*(np.log(2*np.pi*var)+res*res/var));prev=row.date
        losses.append(np.mean(ll))
    got=_innovation_loss(np.log([12,1.5,.7,2.1]),panel,4.,True)
    np.testing.assert_allclose(got,np.mean(losses),rtol=0,atol=1e-13)


def test_person_equal_likelihood_with_unequal_history_lengths():
    e=events(n=4,days=10)
    e=e.loc[~(e.subject_id.eq("p0")&e.date.gt("2025-01-08"))]
    domain=DOMAINS[0];theta=np.log([14.,2.,1.,1.])
    whole=_innovation_loss(theta,_panel(e,domain),4.,True)
    individual=[_innovation_loss(theta,_panel(g,domain),4.,True) for _,g in e.groupby("subject_id")]
    np.testing.assert_allclose(whole,np.mean(individual),rtol=0,atol=1e-13)


def test_row_order_and_serialization_are_exact():
    e=events();m=CausalStateSpace(True,maxiter=20).fit(e,set(e.subject_id),"2025-04-01")
    f=pd.DataFrame({"subject_id":["p5","p0","p5"],"index_date":pd.to_datetime(["2025-03-01","2025-02-01","2025-01-08"])},index=[9,1,9])
    a=m.transform(f,e);b=pickle.loads(pickle.dumps(m)).transform(f,e)
    pd.testing.assert_frame_equal(a,b)
    for i,row in enumerate(f.itertuples()):
        one=m.forecast_one(row.subject_id,row.index_date,e)
        for d in DOMAINS:
            assert a.iloc[i][f"ss_{d}_level"]==one[d]["level"]


def test_integer_and_string_identifiers_not_merged():
    e=events(n=2);e["subject_id"]=e.subject_id.map({"p0":1,"p1":"1"})
    m=CausalStateSpace().fit(e,{1},"2025-04-01")
    assert m.training_subjects_==frozenset({1})
    expected=e.loc[e.subject_id.map(type).eq(int),DOMAINS[0]].mean()
    assert m.parameters_[DOMAINS[0]].mean==pytest.approx(expected)


def test_duplicate_day_and_ambiguous_timezone_rejected():
    e=events(n=2);dup=pd.concat([e,e.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicate"):
        CausalStateSpace().fit(dup,set(dup.subject_id),"2025-04-01")
    with pytest.raises(ValueError,match="timezone"):
        CausalStateSpace().fit(e,set(e.subject_id),"2025-04-01T00:00Z")
    with pytest.raises(ValueError,match="Fit population"):
        CausalStateSpace().transform(pd.DataFrame(),e)


def test_empty_transform_preserves_schema():
    e=events();m=CausalStateSpace(maxiter=10).fit(e,{"p0"},"2025-04-01")
    f=pd.DataFrame({"subject_id":pd.Series([],dtype=str),"index_date":pd.Series([],dtype="datetime64[ns]")})
    assert list(m.transform(f,e).columns)==list(m.feature_columns)
