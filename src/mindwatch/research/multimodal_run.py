"""Reviewed dataset adapters for stacking and small joint-representation models."""
from pathlib import Path
from dataclasses import fields
import json
import numpy as np
import yaml
from .pipeline import load_model_frame
from .privacy import sha256, write_json


def modality_columns(columns):
    result={name:[c for c in columns if c.startswith(name+'__')]
            for name in ['sensor','audio','text','ema','history']}
    result['sensor'] += [c for c in columns if c.startswith(('demo__','quality__sensor'))]
    return {k:v for k,v in result.items() if v}


def develop_stacking(dataset, output, config_path, feature_set='sensor+audio+text', window=14):
    from .stacking import StackingConfig, run_stacking_development
    frame, columns=load_model_frame(dataset,window,'y_phq9',feature_set,cohort='all_labelled')
    config=yaml.safe_load(Path(config_path).read_text())
    config['target_sensitivity']=config.pop('sensitivity_target',.8)
    config['bootstrap_replicates']=config.pop('n_bootstrap',2000)
    manifest=json.loads((Path(dataset)/'data_manifest.json').read_text())
    config['protocol_verified']=manifest['analysis_status']=='confirmatory_ready'
    config['protocol_unresolved']=manifest['unresolved_metadata']
    accepted={f.name for f in fields(StackingConfig)}
    report=run_stacking_development(frame,modality_columns(columns),'y_phq9',output,
                                    {k:v for k,v in config.items() if k in accepted})
    write_json(Path(output)/'dataset_input_spec.json',{'runner':'stacking','window':window,'feature_set':feature_set,
        'cohort':'all_labelled','data_manifest_sha256':sha256(Path(dataset)/'data_manifest.json'),
        'speech_receipt_sha256':sha256(Path(dataset)/f'speech_pooling_{window}d.json')})
    return report


def fusion_inputs(dataset, window=14, task='free_speech', phase='development'):
    """Keep task-specific frozen vectors; never fill absent tasks as observed.

    Only WavLM and KLUE vectors enter this DL adapter. Handcrafted and TF-IDF
    comparisons belong to the late-fusion API. Incomplete vectors abstain from
    that branch. Ineligible sensor windows have masks cleared, while reviewed
    speech-only visits remain available to the union model.
    """
    from .neural import NeuralBatch
    dataset=Path(dataset)
    frame,columns=load_model_frame(dataset,window,'y_phq9','sensor+audio+text',phase=phase,cohort='all_labelled')
    audio=sorted(c for c in columns if c.startswith(f'audio__{task}__wavlm_'))
    text=sorted(c for c in columns if c.startswith('text__free_speech__klue_')) if task=='free_speech' else []
    if not audio:
        raise ValueError('reviewed_task_specific_wavlm_vectors_unavailable')
    demo=sorted(c for c in columns if c.startswith('demo__'))
    with np.load(dataset/f'sequences_{window}d.npz',allow_pickle=False) as archive:
        positions={key:i for i,key in enumerate(archive['sample_ids'])}
        indices=np.asarray([positions[key] for key in frame.sample_id])
        values=archive['values'][indices].copy()
        mask=archive['mask'][indices].copy()
        delta=archive['delta'][indices].copy()
    eligible=frame.quality__sensor_eligible.eq(1).to_numpy()
    mask[~eligible]=0
    values[~eligible]=np.nan
    a=frame[audio].to_numpy(dtype=np.float32,copy=True)
    t=frame[text].to_numpy(dtype=np.float32,copy=True) if text else None
    a_seen=np.isfinite(a).all(axis=1)
    t_seen=np.isfinite(t).all(axis=1) if t is not None else np.zeros(len(frame),bool)
    recording=a_seen|t_seen
    quality=np.column_stack((mask.mean(axis=(1,2)),a_seen.astype(float),t_seen.astype(float))).astype(np.float32)
    batch=NeuralBatch(values,mask,delta,a,t,frame[demo].to_numpy(dtype=np.float32),
                      a_seen,t_seen,recording,quality)
    spec={'runner':'fusion','window':window,'task':task,'audio_columns':audio,'text_columns':text,
          'demographic_columns':demo,'sensor_features':values.shape[-1],'cohort':'all_labelled',
          'speech_receipt_sha256':sha256(dataset/f'speech_pooling_{window}d.json'),
          'data_manifest_sha256':sha256(dataset/'data_manifest.json'),
          'quality_definition':'sensor observed fraction; complete reviewed embedding availability, not ASR accuracy'}
    return frame.reset_index(drop=True),batch,spec


def develop_fusion(dataset, output, config_path, window=14, task='free_speech'):
    from .neural import NeuralConfig
    from .fusion_run import FusionRunConfig, run_fusion_development
    frame,batch,spec=fusion_inputs(dataset,window,task)
    raw=yaml.safe_load(Path(config_path).read_text())
    neural=NeuralConfig(n_features=spec['sensor_features'],audio_dim=len(spec['audio_columns']),
        text_dim=len(spec['text_columns']),demo_dim=len(spec['demographic_columns']),
        hidden_dim=64,dropout=.2,epochs=raw.get('neural_epochs',20),
        batch_size=raw.get('neural_batch_size',64),device=raw.get('neural_device','cpu'),seed=raw.get('seed',42))
    manifest=json.loads((Path(dataset)/'data_manifest.json').read_text())
    config=FusionRunConfig(neural=neural,outer_folds=raw.get('outer_folds',5),inner_folds=raw.get('inner_folds',3),
        calibration_fraction=raw.get('calibration_fraction',.2),seed=raw.get('seed',42),
        bootstrap_replicates=raw.get('n_bootstrap',2000),target_sensitivity=raw.get('sensitivity_target',.8),
        speech_gate_verified=True,speech_gate_reference='verified_pooling_receipt_sha256:'+spec['speech_receipt_sha256'],
        protocol_verified=manifest['analysis_status']=='confirmatory_ready',protocol_unresolved=manifest['unresolved_metadata'])
    # The verified flag above can only be reached after loader receipt/hash checks.
    report=run_fusion_development(frame,batch,output,config)
    write_json(Path(output)/'dataset_input_spec.json',spec)
    return report


def finalize_multimodal(dataset, output, runner):
    output,dataset=Path(output),Path(dataset)
    selection=output/('stacking_selection.json' if runner=='stacking' else 'selection.json')
    if not selection.is_file():
        raise ValueError('frozen_selection_required_before_loading_temporal_partition')
    spec=json.loads((output/'dataset_input_spec.json').read_text())
    if spec['runner']!=runner or spec['data_manifest_sha256']!=sha256(dataset/'data_manifest.json'):
        raise ValueError('dataset_or_runner_changed_since_development')
    if runner=='stacking':
        from .stacking import finalize_stacking
        if spec['speech_receipt_sha256']!=sha256(dataset/f"speech_pooling_{spec['window']}d.json"):
            raise ValueError('speech_pooling_changed_since_development')
        frame,_=load_model_frame(dataset,spec['window'],'y_phq9',spec['feature_set'],
                                 phase='finalize',cohort='all_labelled')
        return finalize_stacking(frame,output)
    if runner!='fusion':
        raise ValueError('unknown_multimodal_runner')
    from .fusion_run import finalize_fusion
    frame,batch,current=fusion_inputs(dataset,spec['window'],spec['task'],phase='finalize')
    if current!=spec:
        raise ValueError('fusion_inputs_or_metadata_changed_since_development')
    return finalize_fusion(frame,batch,output)
