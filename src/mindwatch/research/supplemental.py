"""Explicit exploratory calendar amendment; the original test people stay sealed."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from .privacy import write_json,write_parquet,secure_dir,sha256,artifact_digest
from .pipeline import environment,code_digest


def supplemental_membership(visits,original_split,cutoff='2026-01-01'):
    required={'sample_id','subject_id','index_date'}
    if not required.issubset(visits) or not {'sample_id','subject_id','partition'}.issubset(original_split):
        raise ValueError('explicit_original_participant_membership_required')
    if visits.sample_id.duplicated().any() or original_split.sample_id.duplicated().any():
        raise ValueError('duplicate_sample_membership')
    cutoff=pd.Timestamp(cutoff)
    if pd.isna(cutoff):raise ValueError('explicit_calendar_cutoff_required')
    reserved=set(original_split.loc[original_split.partition.eq('temporal_test'),'subject_id'])
    original_non_test=set(original_split.loc[~original_split.subject_id.isin(reserved),'subject_id'])
    if not set(visits.subject_id).issubset(reserved|original_non_test):
        raise ValueError('subject_missing_from_original_locked_split')
    kept=visits.loc[visits.subject_id.isin(original_non_test)].copy()
    result=kept[list(sorted(required))].copy()
    result['partition']=np.where(result.index_date<cutoff,'development','excluded_after_supplemental_cutoff')
    if set(result.subject_id)&reserved:raise AssertionError('original_test_subject_leaked')
    return result.reset_index(drop=True),reserved


def combine_supplemental_test(development,original_temporal,cutoff='2026-01-01'):
    cutoff=pd.Timestamp(cutoff)
    if not development.partition.eq('development').all() or not original_temporal.partition.eq('temporal_test').all():
        raise ValueError('explicit_original_roles_required')
    if not development.index_date.lt(cutoff).all():
        raise ValueError('development_calendar_overlap')
    temporal=original_temporal.loc[original_temporal.index_date.ge(cutoff)].copy()
    if set(development.subject_id)&set(original_temporal.subject_id):
        raise ValueError('original_test_participant_overlap')
    frame=pd.concat([development,temporal],ignore_index=True)
    if frame.sample_id.duplicated().any():raise ValueError('duplicate_evaluation_sample')
    return frame


def prepare_supplemental(source,output,amendment,cutoff='2026-01-01'):
    source,output,amendment=Path(source),Path(output),Path(amendment)
    if output.exists():raise FileExistsError('supplemental_version_exists')
    if not amendment.is_file():raise ValueError('recorded_amendment_required')
    original_split=pd.read_parquet(source/'split_manifest.parquet')
    # Read the harmless membership columns first; test outcomes are not copied.
    member=pd.read_parquet(source/'visits.parquet',columns=['sample_id','subject_id','index_date'])
    split,reserved=supplemental_membership(member,original_split,cutoff)
    samples=set(split.sample_id);people=set(split.subject_id)
    secure_dir(output)
    write_parquet(output/'split_manifest.parquet',split)
    config=json.loads((source/'analysis_config.json').read_text())
    config['supplemental_calendar_cutoff']=cutoff
    config['metadata_evidence_status']='protocol_documented_but_export_units_and_audio_recordings_not_verified'
    canonical=split.sort_values('sample_id').to_csv(index=False,date_format='%Y-%m-%d').encode()
    digest=hashlib.sha256(canonical).hexdigest()
    spec={'cutoff':cutoff,'digest':digest,'partition_counts':split.partition.value_counts().to_dict(),
          'original_test_subjects_reserved':len(reserved),'original_split_sha256':sha256(source/'split_manifest.parquet'),
          'test_policy':'no_temporal_test_partition_created; original_test_subject_outcomes_not_copied',
          'amendment_sha256':sha256(amendment),'selection_used_outcome_values':False}
    write_json(output/'split.json',spec)
    from .demographics import age_band_series
    from .features import demo_features
    corrected_visits=None
    for name in ['visits.parquet','history.parquet','features_14d.parquet','features_7d.parquet','features_28d.parquet']:
        frame=pd.read_parquet(source/name)
        frame=frame.loc[frame.sample_id.isin(samples)].copy()
        if name=='visits.parquet':
            frame['age_group'],formats=age_band_series(frame.age_group,frame.index_date)
            corrected_visits=frame
            write_json(output/'demographic_correction.json',{'age_formats':formats.value_counts().to_dict(),
                'unknown_bands':int(frame.age_group.eq('unknown').sum()),'DOB_not_serialized':True,
                'original_unknown_age_bug_corrected':True})
        elif name.startswith('features_'):
            frame=frame.drop(columns=[c for c in frame if c.startswith('demo__')]).merge(demo_features(corrected_visits),on='sample_id',validate='one_to_one')
        write_parquet(output/name,frame)
    # Retention needs dates of kept people only. No raw sensor reprocessing.
    daily=pd.read_parquet(source/'daily.parquet')
    write_parquet(output/'daily.parquet',daily.loc[daily.subject_id.isin(people)])
    for window in [7,14,28]:
        with np.load(source/f'sequences_{window}d.npz',allow_pickle=False) as a:
            keep=np.isin(a['sample_ids'],list(samples))
            arrays={key:(a[key][keep] if key!='feature_names' else a[key]) for key in a.files}
        path=output/f'sequences_{window}d.npz'
        np.savez_compressed(path,**arrays);path.chmod(0o600)
    prognosis=pd.read_parquet(source/'prognosis.parquet').drop(columns='partition')
    prognosis=prognosis.loc[prognosis.sample_id.isin(samples)].merge(split[['sample_id','partition']],on='sample_id',validate='one_to_one')
    purge=prognosis.partition.eq('development')&prognosis.target_date.ge(pd.Timestamp(cutoff))
    prognosis.loc[purge,'partition']='excluded_future_outcome_after_cutoff'
    write_parquet(output/'prognosis.parquet',prognosis)
    write_json(output/'analysis_config.json',config)
    write_json(output/'environment.json',environment())
    frozen=json.loads((source/'data_manifest.json').read_text())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_status':'exploratory_amended_calendar',
        'code_sha256':code_digest(),'source_dataset_manifest_sha256':sha256(source/'data_manifest.json'),
        'split_digest':digest,'data_digest':artifact_digest(output/'visits.parquet',output/'features_14d.parquet',output/'split_manifest.parquet'),
        'unresolved_metadata':frozen['unresolved_metadata'],
        'original_test_subjects_reserved':len(reserved),'clinical_validation':False,
        'all_prepared_artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.suffix in {'.parquet','.npz'}}}
    write_json(output/'data_manifest.json',manifest)
    features=pd.read_parquet(output/'features_14d.parquet',columns=['sample_id','quality__sensor_eligible'])
    eligible=split.merge(features,on='sample_id').query("partition=='development' and quality__sensor_eligible==1")
    counts=eligible.groupby('subject_id').size()
    summary={'status':'prepared_supplemental','eligible_development_visits':len(eligible),'eligible_development_subjects':len(counts),
        'repeated_subjects':int(counts.ge(2).sum()),'original_test_subjects_reserved':len(reserved),'cutoff':cutoff,
        'original_primary_split_modified':False,'temporal_test_evaluated':False,'clinical_validation':False}
    write_json(output/'progress.json',summary)
    return summary
