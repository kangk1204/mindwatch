"""Synthetic orchestration checks, independent of confidential cohort values."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch.research import monitoring_study as study
from mindwatch.research.privacy import write_json


def data_fixture():
    dates=pd.to_datetime(['2025-07-10']*60+['2025-09-01']*30+['2025-10-10']*20)
    n=len(dates)
    y=np.arange(n)%2
    frame=pd.DataFrame({'sample_id':[f's{i}' for i in range(n)],'subject_id':[f'p{i%60}' for i in range(n)],
        'index_date':dates,'label_available_at':dates+pd.Timedelta(days=1),'partition':'development',
        'baseline_phq9':6.,'history_available':True,'sensor_available':True,'ema_available':True,
        'quality__sensor_observed_days':14.,'y_phq9':y,'y_gad7':y,'y_isi':y,'phq9':y*10.,
        'gad7':y*10.,'isi':y*15.})
    design={'tabular':np.column_stack([y*8+2,y*5+2,y*7+3]),
            'tabular_names':['history__previous_phq9','history__previous_gad7','history__previous_isi']}
    return SimpleNamespace(frame=frame,designs={'A':design},provenance={'synthetic':True})


def config_fixture():
    return {'seed':42,'seeds':[42],'window':14,'entry_threshold':5,'development_end':'2026-01-01',
        'outer_windows':[['2025-10-01','2025-11-01']],'validation_modes':['known_user'],
        'feature_sets':['A'],'gap_days':14,'calibration_days':28,'inner_folds':2,
        'warmup_classification':{'model':'elasticnet','parameters':{'C':.01,'l1_ratio':0.}},
        'warmup_regression':{'model':'ridge','parameters':{'alpha':10.}},
        'parameter_grids':{'elasticnet':[{'C':.01,'l1_ratio':0.}],'ridge':[{'alpha':10.}]}}


def test_support_counts_people_not_repeated_visits():
    data=data_fixture()
    data.frame['subject_id']='one'
    _,good=study._support(data,np.arange(60),'A','y_phq9','classification',config_fixture())
    assert not good


def test_support_parquet_read_only_boolean_views_do_not_mutate_history(tmp_path):
    data=data_fixture()
    path=tmp_path/'synthetic.parquet'
    data.frame.to_parquet(path,index=False)
    data.frame=pd.read_parquet(path)
    data.frame.loc[::2,'sensor_available']=False
    selected,_=study._support(data,np.arange(60),'B','y_phq9','classification',config_fixture())
    assert len(selected)==30
    assert data.frame.history_available.all()


def test_warmup_uses_declared_model_even_with_other_candidate_scores():
    records=[{'name':'A_elasticnet','model':'elasticnet','feature_set':'A',
              'tuning':{'selection_status':'warmup_fixed_model','mean_score':None}},
             {'name':'A_tcn','model':'tcn','feature_set':'A',
              'tuning':{'selection_status':'warmup_fixed_model','mean_score':.99}}]
    chosen=study.choose_candidate(records,'A',config_fixture(),'classification')
    assert chosen['name']=='A_elasticnet'
    assert chosen['selection_status']=='warmup_fixed_model'


def test_one_se_prefers_simple_without_outer_scores():
    records=[{'name':'A_elasticnet','model':'elasticnet','feature_set':'A',
              'tuning':{'selection_status':'nested_inner_selected','mean_score':.79,'se_score':.02}},
             {'name':'A_catboost','model':'catboost','feature_set':'A',
              'tuning':{'selection_status':'nested_inner_selected','mean_score':.80,'se_score':.03}}]
    assert study.choose_candidate(records,'A',config_fixture(),'classification')['model']=='elasticnet'


def test_outer_outcomes_cannot_change_inner_tuning():
    data=data_fixture()
    # Actual expanding folds and multiple candidates, rather than warmup fallback.
    data.frame=pd.concat([data.frame.iloc[:60].copy() for _ in range(7)],ignore_index=True)
    data.frame['sample_id']=['synthetic-'+str(i) for i in range(len(data.frame))]
    data.frame['index_date']=pd.date_range('2024-01-01',periods=len(data.frame))
    data.frame['label_available_at']=data.frame.index_date+pd.Timedelta(days=1)
    data.designs['A']['tabular']=np.tile(data.designs['A']['tabular'][:60],(7,1))
    fit=np.arange(360)
    config=config_fixture()
    config['parameter_grids']['elasticnet']=[{'C':.01,'l1_ratio':0.},{'C':.1,'l1_ratio':.5}]
    spec={'feature_set':'A','model':'elasticnet','name':'A_elasticnet'}
    before=study.tune_candidate(data,spec,fit,config)
    assert before['selection_status']=='nested_inner_selected'
    assert before['valid_inner_folds']==2
    assert len(before['trials'])==2
    assert all(len(trial['scores'])==2 and not trial['failures'] for trial in before['trials'])
    data.frame.loc[360:,'y_phq9']=1-data.frame.loc[360:,'y_phq9']
    after=study.tune_candidate(data,spec,fit,config)
    assert before==after


@pytest.mark.parametrize('model,parameters,expected',[
    ('ridge',[{'alpha':1.},{'alpha':10.}],{'alpha':10.}),
    ('elasticnet',[{'C':1.},{'C':.01}],{'C':.01}),
    ('catboost',[{'depth':5},{'depth':3}],{'depth':3}),
    ('grud',[{'hidden_dim':64,'epochs':40},{'hidden_dim':32,'epochs':20}],{'hidden_dim':32,'epochs':20}),
])
def test_complexity_order_is_independent_of_grid_order(model,parameters,expected):
    for grid in [parameters,parameters[::-1]]:
        assert min(grid,key=lambda p:study._parameter_complexity(model,p))==expected


def test_tuner_one_se_prefers_stronger_ridge_independent_of_grid_order(monkeypatch):
    from mindwatch.research import monitoring_models
    data=data_fixture()
    data.frame=pd.concat([data.frame.iloc[:60].copy() for _ in range(7)],ignore_index=True)
    data.frame['sample_id']=['synthetic-'+str(i) for i in range(len(data.frame))]
    data.frame['index_date']=pd.date_range('2024-01-01',periods=len(data.frame))
    data.frame['label_available_at']=data.frame.index_date+pd.Timedelta(days=1)
    data.designs['A']['tabular']=np.tile(data.designs['A']['tabular'][:60],(7,1))
    calls=[]
    def fit(d,train,calibration,spec,*args,**kwargs):
        calls.append(spec['parameters']['alpha'])
        assert not len(calibration)
        assert max(train)<360
        return SimpleNamespace(metadata={},predict=lambda d,rows:pd.DataFrame({'score':np.full(len(rows),5.)}))
    monkeypatch.setattr(monitoring_models,'fit_monitoring_bundle',fit)
    for grid in [[{'alpha':1.},{'alpha':10.}],[{'alpha':10.},{'alpha':1.}]]:
        config=config_fixture();config['parameter_grids']['ridge']=grid
        tuned=study.tune_candidate(data,{'feature_set':'A','model':'ridge','name':'A_ridge'},
                                   np.arange(360),config,target='phq9',task='regression')
        assert tuned['selection_status']=='nested_inner_selected'
        assert tuned['parameters']=={'alpha':10.}
        assert all(len(trial['scores'])==2 for trial in tuned['trials'])
    assert len(calls)==8


def test_regression_equal_participant_weight_and_abstention():
    m=study.regression_metrics([0,0,10],[1,1,10],['a','a','b'])
    assert m['mae']==pytest.approx(.5)
    assert study.regression_metrics([0,10],[np.nan,10],['a','b'])['coverage']==.5


def test_prediction_rows_cannot_silently_disappear():
    data=data_fixture()
    with pytest.raises(ValueError,match='every evaluation row'):
        study._prediction_table(data,[90,91],pd.DataFrame({'probability':[.3]}),
            'known_user',0,'A','A','y_phq9','classification','warmup')


def test_joint_inference_retains_separate_provenance_without_labels():
    data=data_fixture()
    data.frame=data.frame.drop(columns=['y_phq9','y_gad7','y_isi','phq9','gad7','isi'])
    class Bundle:
        def __init__(self,task):self.task=task
        def predict(self,data,rows):
            key='probability' if self.task=='classification' else 'score'
            return pd.DataFrame({key:[.2]*len(rows),'model_version':self.task+'_version',
                                 'used_modalities':'A','fallback_reason':'none'})
    pred=study.predict_monitoring(Bundle('classification'),Bundle('regression'),data,[90,91])
    assert len(pred)==2
    assert pred.classification_model_version.eq('classification_version').all()
    assert pred.regression_model_version.eq('regression_version').all()


def test_prognosis_uses_released_current_phq_only_and_records_unmatured_months(tmp_path):
    from mindwatch.research.monitoring_data import MonitoringData
    pf=pd.DataFrame({'sample_id':['q1','q2'],'subject_id':['a','b'],
        'index_date':pd.to_datetime(['2025-07-01','2025-07-02']),
        'index_label_available_at':pd.to_datetime(['2025-07-02','2025-07-03']),
        'target_date':pd.to_datetime(['2025-11-01','2025-11-02']),
        'label_available_at':pd.to_datetime(['2025-11-02','2025-11-03']),
        'partition':'development','history_available':True,'sensor_available':True,'ema_available':False,
        'phq9':[7.,8.],'future_phq9':[8.,9.],'future_ge10':[0.,0.],'worsening_ge5':[0.,0.]})
    names=['history__forecast_current_phq9','history__forecast_current_gad7','sensor__steps']
    designs={b:{'tabular':np.array([[7.,5.,100.],[8.,5.,200.]]),'tabular_names':names} for b in ('A','B')}
    data=MonitoringData(pd.DataFrame(),{}, {},pf,designs)
    report={}
    study._prognosis(data,tmp_path,config_fixture(),report)
    assert report['prognosis']['records'][0]['status']=='no_mature_validation_pairs'
    assert report['prognosis']['records'][0]['validation_pairs']==0


def test_forecast_calibration_uses_outcome_release_not_recent_input_date():
    frame=pd.DataFrame({'sample_id':['a','b','c'],'subject_id':['a','b','c'],
        'partition':'development','index_date':pd.to_datetime(['2025-03-01','2025-05-01','2025-10-10']),
        'target_date':pd.to_datetime(['2025-07-01','2025-09-01','2026-02-10']),
        'label_available_at':pd.to_datetime(['2025-07-02','2025-09-02','2026-02-11'])})
    split=study.forecast_fold(frame,'2025-10-01','2025-11-01')
    assert split['fit'].tolist()==[0]
    assert split['calibration'].tolist()==[1]
    assert split['validation'].tolist()==[2]
    frame.loc[0,'label_available_at']=pd.Timestamp('2025-12-01')
    assert not len(study.forecast_fold(frame,'2025-10-01','2025-11-01')['fit'])


def test_checkpoint_failure_does_not_commit_success(tmp_path,monkeypatch):
    from mindwatch.research import monitoring_data,monitoring_models
    data=data_fixture();config=config_fixture()
    lock={'config':config,'dataset':'synthetic','source_hashes':{},'preserved_hashes':{},'folds':[]}
    write_json(tmp_path/'development_report.json',{'status':'prepared'})
    monkeypatch.setattr(study,'_load_locked',lambda output:lock)
    monkeypatch.setattr(study,'_sources',lambda root:{})
    monkeypatch.setattr(monitoring_data,'load_monitoring_data',lambda *a,**k:data)
    monkeypatch.setattr(study,'_specs',lambda c,task='classification',auxiliary=False:[
        {'name':'A_ridge' if task=='regression' else 'A_elasticnet','model':'ridge' if task=='regression' else 'elasticnet','feature_set':'A'}])
    monkeypatch.setattr(study,'tune_candidate',lambda d,s,*a,**k:{'parameters':{},'selection_status':'warmup_fixed_model'})
    class Bundle:
        metadata={}
        def predict(self,d,rows):
            return pd.DataFrame({'probability':np.repeat(.5,len(rows)),'score':np.repeat(5.,len(rows)),'threshold':.5})
    monkeypatch.setattr(monitoring_models,'fit_monitoring_bundle',lambda *a,**k:Bundle())
    calls=[]
    def checkpoint(path,value):
        calls.append(path)
        if len(calls)==1:raise OSError('synthetic checkpoint failure')
    monkeypatch.setattr(study,'_checkpoint',checkpoint)
    for fn in ('_final_snapshots','_controls','_prognosis'):
        monkeypatch.setattr(study,fn,lambda *a,**k:None)
    result=study.develop(tmp_path)
    assert result['status']=='completed_with_failures'
    assert result['records'][0]['status']=='failed'
    stored=pd.read_parquet(tmp_path/'candidate_oof.parquet')
    assert not ((stored.endpoint=='y_phq9')&(stored.model=='A_elasticnet')).any()
    assert json.loads((tmp_path/'preservation_audit.json').read_text())['all_prior_artifacts_unchanged']


def test_control_failure_cannot_be_completed(tmp_path,monkeypatch):
    from mindwatch.research import monitoring_data,monitoring_models
    original=data_fixture();config=config_fixture()
    config['controls']={'feature_set':'D','model':'tcn'}
    data=monitoring_data.MonitoringData(original.frame,{'D':{'sequence':np.zeros((len(original.frame),14*11*3))}},original.provenance)
    monkeypatch.setattr(study,'tune_candidate',lambda *a,**k:{'parameters':{'hidden_dim':32,'epochs':1}})
    def fail(*a,**k):raise OSError('synthetic control fit failure')
    monkeypatch.setattr(monitoring_models,'fit_monitoring_bundle',fail)
    report={'status':'running','failures':[]}
    study._controls(data,tmp_path,config,report)
    assert report['controls']['status']=='failed'
    assert report['failures'] and all(x['stage']=='control' for x in report['failures'])
