"""Task/availability/partition integration using synthetic prepared artifacts."""
import numpy as np
import pandas as pd
import pytest
from mindwatch.research.pipeline import load_model_frame
from mindwatch.research.multimodal_run import fusion_inputs, finalize_multimodal
from mindwatch.research.privacy import sha256,write_json,write_parquet
from test_pipeline import prepared_synthetic


def speech_dataset(tmp_path):
    dataset,visits,_,_=prepared_synthetic(tmp_path)
    n=len(visits)
    speech=pd.DataFrame({'sample_id':visits.sample_id,
        'audio__free_speech__wavlm_0000':np.ones(n),
        'audio__free_speech__wavlm_0001':np.ones(n)*2,
        'audio__fixed_reading__wavlm_0000':np.ones(n)*3,
        'text__free_speech__klue_0000':np.ones(n)*4})
    speech.loc[1,[c for c in speech if c!='sample_id']]=np.nan
    path=dataset/'speech_features_14d.parquet'
    write_parquet(path,speech)
    write_json(dataset/'speech_pooling_14d.json',{'metadata_gates_verified':True,'features_sha256':sha256(path)})
    write_json(dataset/'data_manifest.json',{'analysis_status':'exploratory_metadata_unresolved','unresolved_metadata':['synthetic']})
    values=np.ones((n,14,2),dtype=np.float32)
    np.savez_compressed(dataset/'sequences_14d.npz',sample_ids=visits.sample_id.to_numpy(dtype=str),
                        values=values,mask=np.ones_like(values),delta=np.ones_like(values))
    return dataset


def test_union_loader_preserves_ineligible_visits_and_sealed_membership(tmp_path):
    dataset=speech_dataset(tmp_path)
    frame,_=load_model_frame(dataset,feature_set='sensor+audio+text',cohort='all_labelled')
    assert len(frame)==120
    assert frame.partition.eq('development').all()
    assert frame.quality__sensor_eligible.eq(0).sum()==1


def test_fusion_clears_ineligible_sensor_but_retains_reviewed_audio_only_visit(tmp_path):
    pytest.importorskip('torch')
    dataset=speech_dataset(tmp_path)
    frame,batch,spec=fusion_inputs(dataset)
    assert len(frame)==120
    assert batch.mask[0].sum()==0 and batch.recording_available[0]
    assert batch.audio_available[0] and not batch.audio_available[1]
    assert batch.text_available[0] and not batch.text_available[1]
    assert batch.audio.shape==(120,2) and batch.text.shape==(120,1)
    assert not any('fixed_reading' in c for c in spec['audio_columns'])
    _,reading,reading_spec=fusion_inputs(dataset,task='fixed_reading')
    assert reading.text is None and reading_spec['text_columns']==[]


def test_final_multimodal_gate_runs_before_any_temporal_loader(tmp_path,monkeypatch):
    from mindwatch.research import multimodal_run
    monkeypatch.setattr(multimodal_run,'load_model_frame',lambda *a,**k: (_ for _ in ()).throw(AssertionError('temporal loader called')))
    with pytest.raises(ValueError,match='frozen_selection_required'):
        finalize_multimodal(tmp_path,tmp_path/'nonexistent','stacking')
    with pytest.raises(ValueError,match='frozen_selection_required'):
        finalize_multimodal(tmp_path,tmp_path/'nonexistent','fusion')
