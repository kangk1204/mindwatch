"""Local preparation and development orchestration; final test is explicit."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .clinical import build_visits
from .features import aggregate_daily, window_features, personal_change
from .privacy import secure_dir, get_key, sha256, write_json, write_parquet, artifact_digest
from .sources import discover, load_clinical_raw
from .splits import lock_split, prognosis_split


def environment():
    packages = {}
    for name in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'torch', 'catboost', 'interpret-core',
                 'tabpfn', 'transformers', 'av', 'librosa', 'pyarrow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages}


def code_digest():
    h = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob('*.py')):
        h.update(file.name.encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def prepare(original, output, config_path, identity_dir=None):
    original, output = Path(original).resolve(), Path(output).resolve()
    if output == original or original in output.parents:
        raise ValueError('derived_output_must_not_modify_original_directory')
    config = yaml.safe_load(Path(config_path).read_text())
    secure_dir(output)
    if (output / 'data_manifest.json').exists():
        raise ValueError('dataset_version_exists_use_new_output_directory')
    key = get_key(Path(identity_dir) if identity_dir else output.parent)
    sources = discover(original)
    paths = sorted({s.path for s in sources})
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    write_json(output / 'progress.json', {'stage': 'source_freeze', 'status': 'running'})
    frozen = [{'source_alias': f'archive_{i+1:02d}', 'path': str(p), 'size': p.stat().st_size,
               'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha256(p)} for i, p in enumerate(paths)]
    raw = load_clinical_raw(sources)
    visits, history, prognosis, raw_mapping, clinical_audit = build_visits(raw, key)
    write_parquet(output / 'visits.parquet', visits)
    write_parquet(output / 'history.parquet', history)
    split, split_spec = lock_split(visits, output, config['test_fraction'])
    write_parquet(output / 'prognosis.parquet', prognosis_split(prognosis, split, split_spec['cutoff']))
    write_json(output / 'clinical_audit.json', clinical_audit)
    write_json(output / 'progress.json', {'stage': 'daily_features', 'status': 'running'})
    daily, daily_audit = aggregate_daily(sources, raw_mapping, config.get('sensor_registry'),
                                        config.get('exploratory_unverified_export_units', False))
    write_parquet(output / 'daily.parquet', daily)
    write_json(output / 'daily_audit.json', daily_audit)
    counts = {}
    for window in config['windows']:
        features, arrays = window_features(visits, daily, window, config['minimum_sensor_days'])
        features = features.merge(history, on='sample_id', validate='one_to_one')
        write_parquet(output / f'features_{window}d.parquet', features)
        arrpath = output / f'sequences_{window}d.npz'
        np.savez_compressed(arrpath, **arrays)
        arrpath.chmod(0o600)
        eligible = set(features.loc[features.quality__sensor_eligible.eq(1), 'sample_id'])
        joined = split.loc[split.sample_id.isin(eligible)]
        counts[str(window)] = {'sensor_eligible_visits': len(eligible),
                               'partition_counts': joined.partition.value_counts().to_dict(),
                               'eligible_subjects': joined.subject_id.nunique()}
    # Read-only source check follows all reads; changed data never complete a version.
    changed = [p for p in paths if before[str(p)] != (p.stat().st_size, p.stat().st_mtime_ns)]
    if changed:
        raise ValueError('source_changed_preparation_not_complete')
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    gates = config.get('metadata_gates', {})
    main_ready = bool(config.get('confirmatory_ready')) and bool(gates) and all(gates.values())
    summary = {'status': 'prepared', 'analysis_status': 'confirmatory_ready' if main_ready else 'exploratory_metadata_unresolved',
               'subjects': clinical_audit['subjects'], 'visits': len(visits), 'windows': counts,
               'cutoff': split_spec['cutoff'], 'all_source_size_mtime_unchanged': True,
               'unresolved_metadata': [k for k, v in gates.items() if not v],
               'clinical_validation': False}
    write_json(output / 'source_manifest.json', frozen)
    write_json(output / 'environment.json', environment())
    write_json(output / 'analysis_config.json', config)
    write_json(output / 'data_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': [x['sha256'] for x in frozen],
        'config_sha256': config_sha, 'code_sha256': code_digest(), 'split_digest': split_spec['digest'],
        'data_digest': artifact_digest(output / 'visits.parquet', output / 'features_14d.parquet',
                                       output / 'split_manifest.parquet'),
        'all_prepared_artifact_sha256': {p.name: sha256(p) for p in sorted(output.iterdir())
                                         if p.suffix in {'.parquet', '.npz'}},
        **summary})
    write_json(output / 'progress.json', summary)
    return summary


def load_model_frame(dataset, window=14, target='y_phq9', feature_set='sensor', phase='development', cohort='sensor_eligible'):
    """Select partition membership before reading outcomes into a modeling frame.

    Development retains the previously defined labelled sensor-eligible cohort.
    Finalization keeps that exact development cohort for fingerprint validation,
    and includes every labelled temporal visit so abstention has a full denominator.
    These are logical access gates; the local source parquet is not encrypted.
    """
    if phase not in {'development', 'finalize'}:
        raise ValueError('invalid_model_loading_phase')
    if cohort not in {'sensor_eligible', 'all_labelled'}:
        raise ValueError('invalid_model_loading_cohort')
    dataset = Path(dataset)
    split = pd.read_parquet(dataset / 'split_manifest.parquet')
    allowed = ['development'] if phase == 'development' else ['development', 'temporal_test']
    split = split.loc[split.partition.isin(allowed)].copy()
    sample_ids = set(split.sample_id)
    # Columnar reads may load row groups internally; no target predicates,
    # statistics or conversion are applied before the partition restriction.
    visits = pd.read_parquet(dataset / 'visits.parquet')
    visits = visits.loc[visits.sample_id.isin(sample_ids)].copy()
    features = pd.read_parquet(dataset / f'features_{window}d.parquet')
    features = features.loc[features.sample_id.isin(sample_ids)].copy()
    frame = visits.merge(features, on='sample_id', validate='one_to_one').merge(
        split[['sample_id', 'partition']], on='sample_id', validate='one_to_one')
    if 'audio' in feature_set or 'text' in feature_set:
        speech_path = dataset / f'speech_features_{window}d.parquet'
        if not speech_path.exists():
            raise ValueError('reviewed_speech_features_unavailable_do_not_silently_run_sensor_only')
        receipt_path = dataset / f'speech_pooling_{window}d.json'
        if not receipt_path.is_file():
            raise ValueError('reviewed_speech_pooling_receipt_missing')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('metadata_gates_verified') is not True or not receipt.get('features_sha256') or receipt['features_sha256'] != sha256(speech_path):
            raise ValueError('reviewed_speech_feature_provenance_invalid')
        speech = pd.read_parquet(speech_path)
        frame = frame.merge(speech, on='sample_id', how='left', validate='one_to_one')
    labelled = frame[target].notna()
    sensor_eligible = frame.quality__sensor_eligible.eq(1)
    eligibility_summary = {'phase': phase, 'coverage_denominator': 'all_labelled_visits_in_supplied_partition',
                           'training_cohort': cohort, 'partitions': {}}
    for partition in allowed:
        member = frame.partition.eq(partition)
        eligibility_summary['partitions'][partition] = {
            'all_visits': int(member.sum()), 'all_participants': int(frame.loc[member, 'subject_id'].nunique()),
            'labelled_visits': int((member & labelled).sum()),
            'labelled_participants': int(frame.loc[member & labelled, 'subject_id'].nunique()),
            'labelled_sensor_eligible_visits': int((member & labelled & sensor_eligible).sum()),
            'labelled_sensor_ineligible_visits': int((member & labelled & ~sensor_eligible).sum()),
        }
    keep = labelled if cohort == 'all_labelled' else labelled & (sensor_eligible | frame.partition.eq('temporal_test'))
    frame = frame.loc[keep].copy()
    prefixes = ['demo__'] if feature_set == 'demo' else ['sensor__', 'demo__', 'quality__']
    if 'ema' in feature_set:
        prefixes.append('ema__')
    if 'history' in feature_set:
        prefixes.append('history__')
    if 'audio' in feature_set:
        prefixes.append('audio__')
    if 'text' in feature_set:
        prefixes.append('text__')
    columns = [c for c in frame if any(c.startswith(p) for p in prefixes)]
    if not columns:
        raise ValueError('no_eligible_predictor_columns')
    if any(c in columns for c in ['phq9', 'gad7', 'isi', target, 'subject_id', 'sample_id', 'region_group']):
        raise ValueError('forbidden_predictor_column')
    frame.attrs['cohort_eligibility_summary'] = eligibility_summary
    return frame, columns


def pool_speech(dataset, recordings, window=14):
    from .multimodal import pool_to_visits
    dataset = Path(dataset)
    if (dataset / f'speech_features_{window}d.parquet').exists() or (dataset / f'speech_pooling_{window}d.json').exists():
        raise FileExistsError('speech_pooling_version_exists_create_new_dataset_version')
    features, summary = pool_to_visits(pd.read_parquet(dataset / 'visits.parquet'),
                                      pd.read_parquet(recordings), window)
    if summary['matched_visits']:
        features_path = dataset / f'speech_features_{window}d.parquet'
        write_parquet(features_path, features)
        summary['features_sha256'] = sha256(features_path)
        summary['source_recordings_sha256'] = sha256(Path(recordings))
    write_json(dataset / f'speech_pooling_{window}d.json', summary)
    return summary


def run_development(dataset, output, config_path, feature_set='sensor', target='y_phq9', window=14):
    from .benchmarks import run_development as benchmark
    from .benchmarks import BenchmarkConfig
    from dataclasses import fields
    dataset, output = Path(dataset), Path(output)
    config = yaml.safe_load(Path(config_path).read_text())
    frame, columns = load_model_frame(dataset, window, target, feature_set)
    if feature_set == 'demo':
        config['observation_feature_columns'] = columns
    config['protocol_verified'] = bool(json.loads((dataset / 'data_manifest.json').read_text()).get('analysis_status') == 'confirmatory_ready')
    config['target_sensitivity'] = config.pop('sensitivity_target', .80)
    config['bootstrap_replicates'] = config.pop('n_bootstrap', 2000)
    manifest = json.loads((dataset / 'data_manifest.json').read_text())
    config['protocol_unresolved'] = manifest.get('unresolved_metadata', [])
    allowed = {field.name for field in fields(BenchmarkConfig)}
    config = {k: v for k, v in config.items() if k in allowed}
    return benchmark(frame, columns, target, output, config)


def inventory_speech(original, dataset):
    from .speech import inventory_audio
    dataset = Path(dataset)
    raw = load_clinical_raw(discover(Path(original)))
    key = get_key(dataset.parent)
    from .privacy import token
    mapping = {str(v): token(key, str(v)) for _, frame in raw for v in frame['_raw_id'] if str(v)}
    return inventory_audio(Path(original), mapping, dataset / 'speech')
