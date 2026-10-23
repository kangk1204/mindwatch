"""Synthetic extension tests: raw parser, temporal features, OOF and local routing."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mindwatch.features import DOMAINS
from mindwatch.research.sensor_engineering import sensor_features,select_columns
from mindwatch.research.hourly_features import hourly_view
from mindwatch.research.ema_personalization import adaptive_one,event_history
from mindwatch.research.foundation import contexts,forecast
from mindwatch.research.missingness_stress import features as stress_features,mask_windows
from mindwatch.research.oof_sensor import fit_offset,time_oof
from mindwatch.research.routing import PopulationRouter
from mindwatch.research.registry import freeze_router,predict_registry


def tables():
    anchors=pd.DataFrame({"sample_id":["synthetic_anchor"],"subject_id":["synthetic_person"],"index_date":[pd.Timestamp("2020-03-01")]})
    daily=pd.DataFrame({"subject_id":"synthetic_person","date":pd.date_range("2020-01-01","2020-03-10"),"steps":np.arange(70,dtype=float)})
    return anchors,daily


def test_sensor_future_rows_and_personal_history():
    a,d=tables();before=sensor_features(a,d,["steps"])
    changed=d.copy();changed.loc[changed.date>=a.index_date.iloc[0],"steps"]=1e9
    pd.testing.assert_frame_equal(before,sensor_features(a,changed,["steps"]))
    assert before.s_available.all() and np.isfinite(before.s_fe_steps__personal_z).all()
    with pytest.raises(ValueError):sensor_features(a,d.assign(phq9=10),["phq9"])
    with pytest.raises(ValueError):sensor_features(a,pd.concat([d,d.iloc[:1]]),["steps"])


def test_sensor_selector_removes_redundancy():
    rng=np.random.default_rng(3);x=rng.normal(size=60)
    frame=pd.DataFrame({"s_fe_first":x,"s_fe_duplicate":x,"s_fe_other":rng.normal(size=60)})
    chosen=select_columns(frame,x,8)
    assert len(set(chosen)&{"s_fe_first","s_fe_duplicate"})==1


def test_hourly_circadian_and_strict_past():
    a,_=tables();times=pd.date_range("2020-01-15","2020-03-02",freq="h")
    table=pd.DataFrame({"subject_id":"synthetic_person","hour":times,"median":60+5*np.sin(times.hour*2*np.pi/24),"n":60})
    x,v,c=hourly_view(a,{"heart_rate":table},["heart_rate"])
    changed=table.copy();changed.loc[changed.hour>=a.index_date.iloc[0],"median"]=1e8
    xx,vv,cc=hourly_view(a,{"heart_rate":changed},["heart_rate"])
    pd.testing.assert_frame_equal(x,xx);np.testing.assert_array_equal(v,vv);np.testing.assert_array_equal(c,cc)
    assert v.shape==(1,336,1) and c.sum()==336*60
    assert np.isfinite(x.s_hour_heart_rate__circadian_amplitude).all()
    with pytest.raises(ValueError):hourly_view(a,{"heart_rate":pd.concat([table,table.iloc[:1]])},["heart_rate"])


def test_ema_actual_events_and_next_context():
    a,_=tables();dates=pd.date_range("2020-01-01",periods=12,freq="7D")
    d=pd.DataFrame({"subject_id":"synthetic_person","date":dates,**{n:np.arange(12)%11 for n in DOMAINS}})
    before=event_history(a,d);ctx=contexts(a,d)
    changed=d.copy();changed.loc[changed.date>=a.index_date.iloc[0],list(DOMAINS)]=10
    pd.testing.assert_frame_equal(before,event_history(a,changed));np.testing.assert_array_equal(ctx,contexts(a,changed))
    assert before.e_event8_ema_depression__n.iloc[0]==8
    assert 0<=adaptive_one([1,2,np.nan,3,4])<=10


def test_frozen_forecast_keeps_missing_domains():
    class Fake:
        def forecast(self,x,**kwargs):
            class Output:pass
            out=Output();out.median=np.full((len(x),4,1),5.);return out
    history=np.full((2,4,112),np.nan,np.float32);history[0,0,-1]=3.
    out=forecast("timestral5m",Fake(),history)
    assert out[0,0]==5 and np.isnan(out[0,1:]).all() and np.isnan(out[1]).all()


def test_missingness_stress_removes_actual_values():
    values=np.arange(2*14*2).reshape(2,14,2).astype(float)
    for scenario in ("retain75","retain50","retain25","missing3","missing7"):
        z=mask_windows(values,scenario);observed=np.isfinite(z)
        np.testing.assert_array_equal(z[observed],values[observed])
        assert observed.sum()<values.size
    assert stress_features(values).shape==(2,6)
    with pytest.raises(ValueError):mask_windows(values,"unknown")


def test_oof_sensor_never_uses_its_row_or_future_label():
    frame=pd.DataFrame({"sample_id":[f"synthetic_sample_{i}" for i in range(8)],"subject_id":[f"synthetic_person_{i}" for i in range(8)],"index_date":pd.to_datetime(["2020-01-01"]*4+["2020-04-01"]*4),"label_available_at":pd.to_datetime(["2020-01-02"]*4+["2020-04-02"]*4),"target":[0,1,0,1,0,1,0,1]})
    x=pd.DataFrame({"h_previous_phq9":np.arange(8,dtype=float)})
    factory=lambda:make_pipeline(SimpleImputer(),LogisticRegression())
    # Pipeline routes sample weights explicitly in this harmless test adapter.
    class Parent:
        def fit(self,x,y,sample_weight):self.model=factory().fit(x,y,logisticregression__sample_weight=sample_weight);return self
        def predict_proba(self,x):return self.model.predict_proba(x)
    p,trace=time_oof(frame,x,[(np.arange(4),np.arange(4,8))],Parent,policy="new")
    assert np.isnan(p[:4]).all() and np.isfinite(p[4:]).all() and len(trace)==1
    changed=frame.copy();changed.loc[4:,"target"]=1-changed.loc[4:,"target"]
    pp,_=time_oof(changed,x,[(np.arange(4),np.arange(4,8))],Parent,policy="new")
    np.testing.assert_array_equal(p,pp)
    with pytest.raises(ValueError):time_oof(frame,x,[(np.arange(4,8),np.arange(4))],Parent)


def test_offset_masks_unseen_channels_and_exact_fallback():
    rng=np.random.default_rng(1);x=pd.DataFrame({"s_first":rng.normal(size=40),"s_never":np.nan})
    base=np.full(40,.4);y=np.arange(40)%2;ids=[f"synthetic_{i}" for i in range(40)]
    model=fit_offset(x,y,ids,base,np.ones(40))
    expected=model.predict(base,x,np.ones(40));changed=x.copy();changed["s_never"]=1e10
    np.testing.assert_array_equal(expected,model.predict(base,changed,np.ones(40)))
    np.testing.assert_array_equal(base,model.predict(base,x*float("nan"),np.ones(40)))
    np.testing.assert_array_equal(base,model.predict(base,x,np.zeros(40)))


class ConstantRisk:
    def __init__(self,value):self.value=value
    def predict(self,x):return np.full(len(x),self.value)


def test_population_router_registry_and_prepared_contract(tmp_path):
    meta=pd.DataFrame({"sample_id":["synthetic_a","synthetic_b"],"subject_id":["synthetic_seen","synthetic_new"],"index_date":pd.to_datetime(["2020-03-01"]*2)})
    x=pd.DataFrame({"h_previous_phq9":[6.,7.]})
    router=PopulationRouter(ConstantRisk(.3),ConstantRisk(.6),("synthetic_seen",))
    out=router.predict(meta,x);assert out.policy.tolist()==["known","new"] and out.prediction.tolist()==[.3,.6]
    with pytest.raises(ValueError):router.predict(meta.assign(target=1),x)
    with pytest.raises(ValueError):router.predict(meta,x,policy="new")
    folder=tmp_path/"frozen_registry";freeze_router(router,tuple(x),folder,"synthetic",meta,x)
    pd.testing.assert_series_equal(out.prediction,predict_registry(folder,meta,x).prediction)
    with pytest.raises(ValueError):predict_registry(folder,meta,x.assign(h_previous_phq9=20))


def test_hourly_stream_parser_local_synthetic_csv(tmp_path):
    if not (shutil.which("clang++") or shutil.which("g++")):pytest.skip("C++17 compiler unavailable")
    from mindwatch.research.raw_hourly import aggregate_csv
    path=tmp_path/"synthetic.csv"
    path.write_text('timestamp,synthetic_column\n2020-01-01 00:00:00,0\n2020-01-01 00:01:00,2\n2020-01-01 01:00:00,4\n')
    output=tmp_path/"hours.csv";audit=aggregate_csv(path,output,1)
    result=pd.read_csv(output)
    assert result.n.tolist()==[2,1] and result["median"].tolist()==[1.,4.] and result.zero.sum()==1 and audit["observed"]==3


def test_optional_masked_representation_ignores_unseen_channel():
    torch=pytest.importorskip("torch")
    from mindwatch.research.representations import fit_representation
    original=torch.get_num_threads();torch.set_num_threads(1)
    try:
        a=np.random.default_rng(6).normal(size=(6,14,2)).astype(np.float32);a[:,:,1]=np.nan
        rep=fit_representation(a,[f"synthetic_{i//2}" for i in range(6)],epochs=1,seeds=(42,))
        first=rep.transform(a);changed=a.copy();changed[:,:,1]=1e9
        np.testing.assert_array_equal(first,rep.transform(changed))
        assert rep.trace[0]["one_window_per_person"] and first.shape==(6,16)
    finally:torch.set_num_threads(original)


def test_corrected_export_parsers():
    from mindwatch.research.parsing import robust_numeric,robust_clock,robust_dates
    np.testing.assert_allclose(robust_numeric(pd.Series(['1,234','0','bad'])),[1234,0,np.nan],equal_nan=True)
    np.testing.assert_allclose(robust_clock(pd.Series(['12:30 AM','오후 1:15','24:00'])),[.5,13.25,np.nan],equal_nan=True)
    assert robust_dates(pd.Series(['20200101','2020/01/02'])).dt.day.tolist()==[1,2]


def test_optional_joint_neural_comparator_fallback_and_unseen():
    torch=pytest.importorskip('torch')
    from mindwatch.research.neural_comparators import fit
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        f=pd.DataFrame({'subject_id':[f'synthetic_{i}' for i in range(8)]})
        x=pd.DataFrame({'e_ema_depression_last':np.arange(8,dtype=float)})
        v=np.random.default_rng(2).normal(size=(8,14,2)).astype(np.float32);v[:,:,1]=np.nan;y=np.arange(8)%2
        model=fit(f,x,v,y,'phq','phq9','grud',epochs=1,seeds=(42,))
        expected=model.raw(x,v);changed=v.copy();changed[:,:,1]=1e9
        np.testing.assert_array_equal(expected,model.raw(x,changed))
        empty=np.full_like(v,np.nan);np.testing.assert_array_equal(model.base.predict(x[list(model.base.cols)]),model.raw(x,empty))
    finally:torch.set_num_threads(threads)


def test_joint_ema_fit_mask_and_exact_fallback():
    torch=pytest.importorskip('torch')
    from mindwatch.research.ema_joint import fit,DOMAINS
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        rng=np.random.default_rng(77);values=rng.normal(size=(6,14,2)).astype(np.float32);values[:,:,1]=np.nan
        targets=rng.uniform(0,10,(6,4));targets[0,1]=np.nan;baseline=np.full((6,4),5.)
        model=fit([f'synthetic_{i}' for i in range(6)],values,targets,baseline,DOMAINS[0],epochs=1,seeds=(42,))
        x=pd.DataFrame({'e_previous':np.zeros(6)});before=model.raw(x,values,baseline[:,0]);changed=values.copy();changed[:,:,1]=1e6
        np.testing.assert_array_equal(before,model.raw(x,changed,baseline[:,0]))
        np.testing.assert_array_equal(baseline[:,0],model.raw(x,np.full_like(values,np.nan),baseline[:,0]))
    finally:torch.set_num_threads(threads)


def test_participant_weighted_associations_and_strict_past():
    from mindwatch.research.associations import weighted_rank,rho,between_within
    np.testing.assert_allclose(weighted_rank(np.array([1.,1.,3.]),np.array([1.,2.,1.])),[1.5,1.5,3.5])
    frame=pd.DataFrame({'subject_id':np.repeat([f'synthetic_{i}' for i in range(30)],5),'index_date':np.tile(pd.date_range('2000-01-01',periods=5),30),'sensor':np.arange(150.),'target':np.arange(150.)})
    assert rho(frame.sensor.to_numpy(),frame.target.to_numpy(),frame.subject_id.to_numpy())[0]==pytest.approx(1.)
    result=between_within(frame,'sensor','target');assert result['between'][0]==pytest.approx(1.)
    assert result['within_strict_past'][1]==60


def test_research_private_writers_reject_git_destination(tmp_path):
    from mindwatch.research.privacy import write_json,write_parquet
    from mindwatch.research.benchmarks import _checkpoint
    repo=tmp_path/'checkout';repo.mkdir();(repo/'.git').mkdir()
    for call in [lambda:write_json(repo/'report.json',{'synthetic':True}),lambda:write_parquet(repo/'rows.parquet',pd.DataFrame({'x':[1]})),lambda:_checkpoint(repo/'model.joblib',{'synthetic':True})]:
        with pytest.raises(ValueError):call()
    assert not (repo/'report.json').exists() and not (repo/'rows.parquet').exists() and not (repo/'model.joblib').exists()
