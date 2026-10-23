"""Run actual sequence comparators through the common nested evaluation engine."""
from __future__ import annotations
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .benchmarks import BenchmarkConfig, run_development
from .sequence_arrays import flatten_sequence
from .pipeline import load_model_frame
from .privacy import write_json, sha256


def sequence_frame(dataset, window=14, target='y_phq9', phase='development'):
    dataset = Path(dataset)
    frame, _ = load_model_frame(dataset, window, target, 'sensor', phase=phase)
    with np.load(dataset / f'sequences_{window}d.npz', allow_pickle=False) as archive:
        positions = {value: i for i, value in enumerate(archive['sample_ids'])}
        order = np.asarray([positions[value] for value in frame.sample_id])
        packed = flatten_sequence(archive['values'][order], archive['mask'][order], archive['delta'][order])
    names = [f'sensor__seq_{i:05d}' for i in range(packed.shape[1])]
    demo = [name for name in frame if name.startswith('demo__')]
    base = frame[['sample_id', 'subject_id', 'index_date', 'partition', target, 'quality__sensor_eligible', *demo]].reset_index(drop=True)
    base = pd.concat([base, pd.DataFrame(packed, columns=names)], axis=1)
    base.attrs['cohort_eligibility_summary'] = frame.attrs.get('cohort_eligibility_summary')
    return base, names + demo, len(demo)


def develop_sequences(dataset, output, config_path, window=14, target='y_phq9'):
    config = yaml.safe_load(Path(config_path).read_text())
    config['target_sensitivity'] = config.pop('sensitivity_target', .8)
    config['bootstrap_replicates'] = config.pop('n_bootstrap', 2000)
    config['sequence_length'] = window
    frame, columns, n_demo = sequence_frame(dataset, window, target)
    config['neural_demographics'] = n_demo
    config['n_demographics'] = n_demo
    source_dir = config.get('mtm_source_dir')
    if source_dir:
        config['mtm_source_dir'] = str(Path(source_dir).resolve())
    manifest = json.loads((Path(dataset) / 'data_manifest.json').read_text())
    config['protocol_verified'] = manifest['analysis_status'] == 'confirmatory_ready'
    config['protocol_unresolved'] = manifest['unresolved_metadata']
    allowed = {field.name for field in fields(BenchmarkConfig)}
    result = run_development(frame, columns, target, output, {k: v for k, v in config.items() if k in allowed})
    write_json(Path(output) / 'input_provenance.json', {
        'dataset_digest': manifest['data_digest'], 'sequence_length': window,
        'sequence_sha256': sha256(Path(dataset)/f'sequences_{window}d.npz'),
        'features_sha256': sha256(Path(dataset)/f'features_{window}d.parquet'),
        'sequence_channels': (len(columns) - n_demo) // (3 * window),
        'demographic_columns': n_demo, 'development_eligibility': 'same_sensor_window_as_tabular',
        'delta_initialization': 'one_day_cold_start; subsequent_delta_past_mask_only',
        'test_evaluated': False})
    return result
