"""Development-only, nested chronological comparisons of nonvoice models.

Old temporal-test outcomes are never loaded. Saved OOF predictions are private;
reported estimates describe exploratory development validation, not a new test.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from .benchmarks import (BenchmarkConfig, _calibrated, _calibrator, _checkpoint,
                         _estimator, _fit, _probability)
from .evaluation import binary_metrics, paired_cluster_bootstrap, select_sensitivity_threshold
from .pipeline import load_model_frame
from .privacy import secure_dir, sha256, write_json, write_parquet


@dataclass
class TemporalData:
    frame: pd.DataFrame
    designs: dict
    provenance: dict


def load_temporal_data(dataset: str | Path, window: int = 14, *, development_end=None) -> TemporalData:
    from .sequence_arrays import flatten_sequence, past_only_delta
    from .temporal_features import temporal_features

    dataset = Path(dataset)
    spec = json.loads((dataset / 'split.json').read_text())
    reserved = spec.get('original_test_subjects_reserved')
    if type(reserved) is not int or reserved < 1:
        raise ValueError('Use the amended development dataset with original test people excluded.')
    frame, columns = load_model_frame(dataset, window, 'y_phq9', 'sensor+ema+history',
                                      phase='development', cohort='all_labelled')
    frame = frame.reset_index(drop=True)
    end = pd.Timestamp(development_end or spec.get('cutoff'))
    if pd.isna(end):
        raise ValueError('Explicit development end date required.')
    if not frame.partition.eq('development').all() or frame.index_date.ge(end).any():
        raise ValueError('Only observations before the locked development end are allowed.')
    if not np.isin(frame.y_phq9.to_numpy(), [0, 1]).all():
        raise ValueError('Observed binary PHQ targets are required.')
    with np.load(dataset / f'sequences_{window}d.npz', allow_pickle=False) as archive:
        ids = archive['sample_ids']
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate sequence sample keys.')
        positions = {key: index for index, key in enumerate(ids)}
        order = np.asarray([positions[key] for key in frame.sample_id])
        values, mask = archive['values'][order], archive['mask'][order]
        channels = archive['feature_names'].tolist()
    if not np.isin(mask, [0, 1]).all() or not np.isfinite(values[mask.astype(bool)]).all():
        raise ValueError('Invalid observed sequence values or masks.')
    actual_eligible = mask.astype(bool).any(axis=2).sum(axis=1) >= 4
    if not np.array_equal(actual_eligible, frame.quality__sensor_eligible.eq(1).to_numpy()):
        raise ValueError('Sensor eligibility changed between feature and sequence data.')
    delta = past_only_delta(mask)
    packed = flatten_sequence(values, mask, delta)
    signal, quality = temporal_features(values, mask, channels)
    tab = pd.concat([frame[columns].reset_index(drop=True), signal, quality], axis=1)
    if tab.columns.duplicated().any():
        raise ValueError('Temporal feature names collide.')
    designs = {}
    for scenario in ('sensor', 'sensor+ema', 'sensor+ema+history'):
        prefixes = ['sensor__', 'demo__', 'quality__']
        static_prefixes = ['demo__']
        for modality in ('ema', 'history'):
            if modality in scenario:
                prefixes.append(modality + '__')
                static_prefixes.append(modality + '__')
        names = [c for c in tab if c.startswith(tuple(prefixes))]
        static = [c for c in tab if c.startswith(tuple(static_prefixes))]
        designs[scenario] = {
            'tabular': tab[names].to_numpy(dtype=float), 'tabular_names': names,
            'sequence': np.column_stack([packed, tab[static].to_numpy(dtype=float)]),
            'static_names': static, 'sequence_columns': packed.shape[1],
        }
    demo = [c for c in frame if c.startswith('demo__')]
    history = [c for c in frame if c.startswith(('demo__', 'history__'))]
    designs['demographic'] = {'tabular': frame[demo].to_numpy(dtype=float), 'tabular_names': demo}
    designs['history_only'] = {'tabular': frame[history].to_numpy(dtype=float), 'tabular_names': history}
    # This comparator receives collection patterns and demographics only.
    names = [f'quality__mask_{i}' for i in range(mask[0].size)]
    names += [f'quality__delta_{i}' for i in range(mask[0].size)] + demo
    designs['missingness'] = {
        'tabular': np.column_stack([mask.reshape(len(mask), -1), delta.reshape(len(mask), -1), frame[demo]]),
        'tabular_names': names,
    }
    provenance = {
        'dataset_manifest_sha256': sha256(dataset / 'data_manifest.json'),
        'split_sha256': sha256(dataset / 'split_manifest.parquet'),
        'sequences_sha256': sha256(dataset / f'sequences_{window}d.npz'),
        'features_sha256': sha256(dataset / f'features_{window}d.parquet'),
        'original_test_people_excluded': reserved, 'window': window, 'channels': channels,
        'labelled_participants': int(frame.subject_id.nunique()), 'labelled_visits': len(frame),
        'sensor_eligible_participants': int(frame.loc[actual_eligible].subject_id.nunique()),
        'sensor_eligible_visits': int(actual_eligible.sum()),
        'delta_initialization': 'zero; past masks only',
        'eligibility': frame.attrs.get('cohort_eligibility_summary'),
        'feature_names': {key: {k: v for k, v in value.items() if k.endswith('names')} for key, value in designs.items()},
        'old_test_loaded': False,
    }
    return TemporalData(frame, designs, provenance)


def candidates(config):
    result = [{'name': name, 'scenario': name, 'model': name} for name in ('demographic', 'missingness', 'history_only')]
    for scenario in config['feature_sets']:
        for model in config['models']:
            result.append({'name': scenario.replace('+', '_') + '_' + model, 'scenario': scenario, 'model': model})
    if config.get('include_mtm_sensor'):
        result.append({'name': 'sensor_mtm', 'scenario': 'sensor', 'model': 'mtm'})
    return result


def design(data, candidate):
    source = data.designs[candidate['scenario']]
    return source['sequence' if candidate['model'] in {'grud', 'tcn', 'mtm'} else 'tabular']


def estimator(data, candidate, parameters, config, seed):
    model = candidate['model']
    if model in {'grud', 'tcn', 'mtm'}:
        from .temporal_features import TemporalFusionClassifier
        source = data.designs[candidate['scenario']]
        return TemporalFusionClassifier(
            sequence_length=config['window'], n_static_features=len(source['static_names']),
            static_feature_names=source['static_names'], architecture=model,
            mtm_source_dir=config.get('mtm_source_dir'),
            batch_size=config.get('neural_batch_size', 64), device=config.get('neural_device', 'cpu'),
            seed=seed, **parameters)
    family = 'elasticnet' if model in {'history_only', 'missingness'} else model
    return _estimator(family, parameters, seed, BenchmarkConfig())


def _support(frame, rows):
    y = frame.iloc[rows].y_phq9.to_numpy(dtype=float)
    return bool(len(rows) > 0 and len(np.unique(y)) == 2)


def _simple_rank(record):
    rank = {'demographic': 0, 'history_only': 1, 'missingness': 2,
            'elasticnet': 3, 'catboost': 4, 'grud': 5, 'tcn': 6, 'mtm': 7}
    return rank[record['model']], record['name']


def one_se(records, mean_key='mean_auroc', se_key='se_auroc'):
    valid = [r for r in records if r.get(mean_key) is not None and r.get('status') == 'ok']
    if not valid:
        return {'status': 'no_selectable_model'}
    best = max(valid, key=lambda r: r[mean_key])
    boundary = best[mean_key] - best.get(se_key, 0.0)
    chosen = min([r for r in valid if r[mean_key] >= boundary], key=_simple_rank)
    return {'status': 'ok', 'selected_name': chosen['name'], 'best_mean_name': best['name'],
            'one_se_boundary': boundary, 'rule': 'simplest within one SE; three dependent time blocks are a heuristic'}


def tune(data, candidate, rows, config, seed):
    from .temporal_splits import forward_splits
    f = data.frame.iloc[rows].reset_index(drop=True)
    splits = forward_splits(f, n_splits=config['inner_folds'], gap_days=config['gap_days'])
    if len(splits) != config['inner_folds'] or any(not _support(f, a) or not _support(f, b) for a, b in splits):
        return {'status': 'insufficient_chronological_inner_support', 'folds': len(splits)}
    xx, yy, groups = design(data, candidate), data.frame.y_phq9.to_numpy(), data.frame.subject_id.to_numpy()
    records = []
    for parameters in config['parameter_grids'][candidate['model']]:
        scores = []
        try:
            for a, b in splits:
                train, valid = rows[a], rows[b]
                fitted = _fit(estimator(data, candidate, parameters, config, seed), xx[train], yy[train], groups[train])
                metric = binary_metrics(yy[valid], _probability(fitted, xx[valid]), groups[valid], None)['auroc']
                if metric is None:
                    raise ValueError('Undefined inner AUROC.')
                scores.append(metric)
            records.append({'status': 'ok', 'parameters': parameters, 'scores': scores,
                            'mean_auroc': float(np.mean(scores)), 'se_auroc': float(np.std(scores, ddof=1) / np.sqrt(len(scores)))})
        except Exception as exc:
            records.append({'status': 'failed', 'parameters': parameters, 'exception_type': type(exc).__name__})
    good = [r for r in records if r['status'] == 'ok']
    if not good:
        return {'status': 'all_candidates_failed', 'candidates': records}
    best = max(good, key=lambda r: r['mean_auroc'])
    return {**best, 'folds': len(splits), 'candidates': records}


def _fit_calibrate(data, candidate, parameters, fit, calibration, config, seed, matrix=None):
    xx = design(data, candidate) if matrix is None else matrix
    frame = data.frame
    yy, groups = frame.y_phq9.to_numpy(), frame.subject_id.to_numpy()
    fitted = _fit(estimator(data, candidate, parameters, config, seed), xx[fit], yy[fit], groups[fit])
    calibrator = _calibrator(yy[calibration], _probability(fitted, xx[calibration]), groups[calibration])
    operating = select_sensitivity_threshold(yy[calibration], _calibrated(calibrator, _probability(fitted, xx[calibration])), groups[calibration], config['target_sensitivity'])
    if operating['threshold'] is None:
        raise ValueError('Calibration operating point unavailable.')
    return fitted, calibrator, operating


def _prior_fingerprints(private_root):
    names = ['primary_sensor_phq_age_corrected_v3', 'primary_sensor_ema_phq_age_corrected_v3',
             'primary_sequences_age_corrected_v3', 'supplemental_sensor_phq_v3',
             'supplemental_sensor_ema_phq_v3', 'supplemental_sensor_ema_history_phq_v3']
    files = [private_root / name / filename for name in names for filename in
             ('selection.json', 'frozen_development_model.joblib', 'final_results.json')]
    files += [private_root / folder / 'split_manifest.parquet' for folder in
              ('dataset_v1', 'dataset_primary_age_corrected_v3', 'dataset_supplemental_v3')]
    # Preserve the initial time-series run when a corrected version is fitted.
    previous = private_root / 'temporal_v5'
    if previous.is_dir():
        files += list(previous.glob('*.joblib'))
        files += [previous / name for name in ('selection.json', 'development_report.json')]
    return {str(path.relative_to(private_root)): sha256(path) for path in files if path.is_file()}


def shuffled_sequence(matrix, sequence_columns, window, seed):
    from .sequence_arrays import flatten_sequence, past_only_delta
    x = np.asarray(matrix)
    width = sequence_columns // (3 * window)
    block = window * width
    values = x[:, :block].reshape(-1, window, width).copy()
    mask = x[:, block:2 * block].reshape(-1, window, width).copy()
    rng = np.random.default_rng(seed)
    for row in range(len(x)):
        order = rng.permutation(window)
        values[row], mask[row] = values[row, order], mask[row, order]
    return np.column_stack([flatten_sequence(values, mask, past_only_delta(mask)), x[:, sequence_columns:]])


def run_comparison(data: TemporalData, config: dict, output: Path):
    from .temporal_splits import make_forward_fold

    output = secure_dir(output)
    if (output / 'development_report.json').exists():
        raise FileExistsError('Use a new output directory; prior results are immutable.')
    frame = data.frame.reset_index(drop=True)
    if not frame.partition.eq('development').all():
        raise ValueError('Only development partition can enter temporal modeling.')
    data.frame = frame
    specs = candidates(config)
    outer, manifest = [], []
    eligible = frame.quality__sensor_eligible.eq(1).to_numpy() if 'quality__sensor_eligible' in frame else np.ones(len(frame), bool)
    # Reserve calibration PEOPLE before any CV, tuning or model selection.
    # Selection uses only dates, pseudonymous membership and a fixed seed.
    final_cal_start = pd.Timestamp(config['final_calibration_start'])
    final_period = (frame.index_date >= final_cal_start) & (frame.index_date < pd.Timestamp(config['development_end']))
    pool = np.sort(frame.loc[final_period, 'subject_id'].unique())
    reserve_count = int(np.ceil(len(pool) * config.get('final_calibration_fraction', .5)))
    if not 0 < reserve_count < len(pool):
        raise ValueError('Final calibration reservation must leave comparison participants.')
    reserved = np.random.default_rng(config['seed'] + 731).permutation(pool)[:reserve_count]
    is_reserved = frame.subject_id.isin(reserved).to_numpy()
    comparison_positions = np.flatnonzero(~is_reserved)
    comparison_frame = frame.iloc[comparison_positions].reset_index(drop=True)
    reservation = frame.loc[is_reserved, ['sample_id', 'subject_id', 'index_date']].copy()
    reservation['role'] = 'reserved_final_calibration_person_excluded_from_all_model_selection'
    write_parquet(output / 'final_calibration_reservation.parquet', reservation)
    for fold, (start, end) in enumerate(config['outer_windows']):
        split = make_forward_fold(comparison_frame, start, end, gap_days=config['gap_days'], calibration_days=config['calibration_days'])
        for role in ('fit', 'calibration', 'validation'):
            split[role] = comparison_positions[split[role]]
        split['all_validation'] = split['validation'].copy()
        split['metadata']['sensor_eligible_roles'] = {}
        for role in ('fit', 'calibration', 'validation'):
            split[role] = split[role][eligible[split[role]]]
            split['metadata']['sensor_eligible_roles'][role] = {
                'visits': len(split[role]), 'participants': int(frame.iloc[split[role]].subject_id.nunique())}
        if any(not _support(frame, split[role]) for role in ('fit', 'calibration', 'validation')):
            raise ValueError('Fixed chronological fold has insufficient class support; no fallback split.')
        outer.append(split)
        for role in ('fit', 'calibration', 'validation'):
            indices = split['all_validation'] if role == 'validation' else split[role]
            part = frame.iloc[indices][['sample_id', 'subject_id', 'index_date']].copy()
            part['sensor_eligible'] = eligible[indices]
            part['outer_fold'], part['role'] = fold, role
            manifest.append(part)
    write_parquet(output / 'development_cv_manifest.parquet', pd.concat(manifest, ignore_index=True))
    report = {'status': 'running', 'analysis_type': 'exploratory_nested_forward_development',
              'test_evaluated': False, 'clinical_validation': False, 'config': config,
              'provenance': data.provenance, 'folds': [s['metadata'] for s in outer],
              'calibration_reservation': {'all_reserved_people': len(reserved),
                  'reserved_labelled_visits': int(is_reserved.sum()),
                  'comparison_sensor_eligible_people': int(frame.loc[~is_reserved & eligible].subject_id.nunique()),
                  'comparison_sensor_eligible_visits': int((~is_reserved & eligible).sum()),
                  'selection_used_outcomes': False, 'independent_of_all_model_selection': True},
              'models': [], 'nested_selectors': {}, 'final_models': {}, 'controls': {}}
    write_json(output / 'development_report.json', report)
    predictions, records, timings = [], [], []
    yy, groups = frame.y_phq9.to_numpy(), frame.subject_id.to_numpy()
    for candidate in specs:
        record = {**candidate, 'status': 'running', 'folds': [], 'fold_aurocs': []}
        xx = design(data, candidate)
        for fold, split in enumerate(outer):
            begun = time.monotonic()
            fit, cal, valid = (split[r] for r in ('fit', 'calibration', 'validation'))
            tuning = tune(data, candidate, fit, config, config['seed'] + fold)
            fold_record = {'fold': fold, 'tuning': tuning, 'status': tuning['status']}
            if tuning['status'] == 'ok':
                try:
                    fitted, calibrator, operating = _fit_calibrate(data, candidate, tuning['parameters'], fit, cal, config, config['seed'] + fold)
                    p = _calibrated(calibrator, _probability(fitted, xx[valid]))
                    all_valid = split['all_validation']
                    full_p = pd.Series(p, index=valid).reindex(all_valid).to_numpy()
                    metrics = binary_metrics(yy[all_valid], full_p, groups[all_valid], operating['threshold'])
                    if metrics['auroc'] is None:
                        raise ValueError('Undefined outer AUROC.')
                    fold_record.update(status='ok', metrics=metrics, operating_point=operating)
                    pred = frame.iloc[all_valid][['sample_id', 'subject_id', 'index_date']].copy()
                    pred['target'], pred['probability'] = yy[all_valid], full_p
                    pred['threshold'], pred['model'], pred['outer_fold'] = operating['threshold'], candidate['name'], fold
                    _checkpoint(output / f'fold_{fold}_{candidate["name"]}.joblib',
                                {'model': fitted, 'calibrator': calibrator, 'threshold': operating['threshold']})
                    record['fold_aurocs'].append(metrics['auroc'])
                    predictions.append(pred)
                except Exception as exc:
                    fold_record.update(status='fit_failed', exception_type=type(exc).__name__)
            fold_record['elapsed_seconds'] = round(time.monotonic() - begun, 2)
            record['folds'].append(fold_record)
            print(json.dumps({'model': candidate['name'], 'fold': fold, 'status': fold_record['status'], 'seconds': fold_record['elapsed_seconds']}), flush=True)
            timings.append({'model': candidate['name'], 'fold': fold, 'seconds': fold_record['elapsed_seconds']})
            report['models'] = records + [record]
            write_json(output / 'development_report.json', report)
        if len(record['fold_aurocs']) == len(outer):
            record.update(status='ok', mean_auroc=float(np.mean(record['fold_aurocs'])),
                          se_auroc=float(np.std(record['fold_aurocs'], ddof=1) / np.sqrt(len(outer))))
        else:
            record['status'] = 'incomplete_forward_cv'
        records.append(record)
    if not predictions:
        raise ValueError('No valid chronological predictions; inspect safe status report.')
    oof = pd.concat(predictions, ignore_index=True)
    write_parquet(output / 'oof_predictions.parquet', oof)
    report['models'] = records
    for record in records:
        if record['status'] == 'ok':
            pred = oof[oof.model.eq(record['name'])]
            record['pooled_oof_metrics'] = binary_metrics(pred.target, pred.probability, pred.subject_id, pred.threshold)
    # Model-family selection for each outer prediction uses INNER scores only.
    for scenario in config['feature_sets']:
        allowed = [r for r in records if r['scenario'] == scenario or r['name'] == 'demographic'
                   or (scenario.endswith('history') and r['name'] == 'history_only')]
        chosen_parts, chosen_names = [], []
        for fold in range(len(outer)):
            inner_records = [{**r, **r['folds'][fold]['tuning']} for r in allowed]
            choice = one_se(inner_records)
            chosen_names.append(choice)
            if choice['status'] != 'ok':
                continue
            selected_record = next(r for r in allowed if r['name'] == choice['selected_name'])
            if selected_record['folds'][fold]['status'] != 'ok':
                continue
            chosen_parts.append(oof[oof.model.eq(choice['selected_name']) & oof.outer_fold.eq(fold)])
        if len(chosen_parts) == len(outer):
            selected = pd.concat(chosen_parts)
            report['nested_selectors'][scenario] = {'fold_selections': chosen_names,
                'metrics': binary_metrics(selected.target, selected.probability, selected.subject_id, selected.threshold)}
        else:
            report['nested_selectors'][scenario] = {'status': 'incomplete_selection_procedure', 'fold_selections': chosen_names}
    # Fit an explicit developer-selected pipeline; do not reopen an old test.
    all_cal = np.flatnonzero(final_period & is_reserved)
    cal = all_cal[eligible[all_cal]]
    fit = np.flatnonzero((frame.index_date < final_cal_start - pd.Timedelta(days=config['gap_days'])) & ~is_reserved & eligible)
    if not _support(frame, fit) or not _support(frame, cal):
        raise ValueError('Final chronological fit/calibration class support insufficient.')
    roles = frame.iloc[np.r_[fit, cal]][['sample_id', 'subject_id', 'index_date']].copy()
    roles['role'] = ['fit'] * len(fit) + ['calibration'] * len(cal)
    write_parquet(output / 'frozen_fit_manifest.parquet', roles)
    for scenario in config['feature_sets']:
        allowed = [r for r in records if r['scenario'] == scenario or r['name'] == 'demographic'
                   or (scenario.endswith('history') and r['name'] == 'history_only')]
        selection = one_se(allowed)
        report['final_models'][scenario] = selection
        if selection['status'] != 'ok':
            continue
        candidate = next(c for c in specs if c['name'] == selection['selected_name'])
        tuning = tune(data, candidate, fit, config, config['seed'] + 10000)
        if tuning['status'] != 'ok':
            selection.update(status='final_tuning_failed', tuning=tuning)
            continue
        fitted, calibrator, operating = _fit_calibrate(data, candidate, tuning['parameters'], fit, cal, config, config['seed'] + 10000)
        artifact = {'model': fitted, 'calibrator': calibrator, 'threshold': operating['threshold'],
                    'candidate': candidate, 'provenance': data.provenance, 'parameters': tuning['parameters']}
        digest = _checkpoint(output / ('frozen_' + scenario.replace('+', '_') + '.joblib'), artifact)
        selection.update(checkpoint_sha256=digest, tuning=tuning, operating_point=operating,
                         fit_participants=int(frame.iloc[fit].subject_id.nunique()), fit_visits=len(fit),
                         calibration_participants=int(frame.iloc[cal].subject_id.nunique()), calibration_visits=len(cal))
    report['timings'] = timings
    if config.get('order_control'):
        try:
            _controls(data, specs, records, oof, outer, config, output, report)
        except Exception as exc:
            report['controls']['status'] = 'failed'
            report['controls']['exception_type'] = type(exc).__name__
    report['failed_models'] = [r['name'] for r in records if r['status'] != 'ok']
    all_ok = not report['failed_models'] and all(v['status'] == 'ok' for v in report['final_models'].values())
    all_ok = all_ok and report['controls'].get('status') != 'failed'
    report['status'] = 'completed_development_only' if all_ok else 'completed_with_failures_development_only'
    write_json(output / 'development_report.json', report)
    write_json(output / 'selection.json', {'analysis_type': report['analysis_type'], 'status': report['status'],
                                         'final_models': report['final_models'], 'test_evaluated': False,
                                         'reuse_old_test_for_new_selection': False})
    return report


def _controls(data, specs, records, oof, outer, config, output, report):
    eligible = [r for r in records if r['model'] in {'grud', 'tcn', 'mtm'} and r['status'] == 'ok' and r['scenario'] == 'sensor+ema']
    if not eligible:
        report['controls'] = {'status': 'no_completed_sensor_ema_sequence_candidate'}
        return
    record = max(eligible, key=lambda r: r['mean_auroc'])
    candidate = next(c for c in specs if c['name'] == record['name'])
    controls = []
    for control in ('shuffled_order', 'stability_seed'):
        parts = []
        for fold, split in enumerate(outer):
            matrix = design(data, candidate)
            seed = config['seed'] + fold
            if control == 'shuffled_order':
                matrix = shuffled_sequence(matrix, data.designs[candidate['scenario']]['sequence_columns'], config['window'], seed)
            else:
                seed = config.get('stability_seed', 2026) + fold
            fit, cal, valid = (split[r] for r in ('fit', 'calibration', 'validation'))
            fitted, calibrator, operating = _fit_calibrate(data, candidate, record['folds'][fold]['tuning']['parameters'], fit, cal, config, seed, matrix)
            p = _calibrated(calibrator, _probability(fitted, matrix[valid]))
            all_valid = split['all_validation']
            full_p = pd.Series(p, index=valid).reindex(all_valid).to_numpy()
            pred = data.frame.iloc[all_valid][['sample_id', 'subject_id', 'index_date']].copy()
            pred['target'], pred['probability'], pred['threshold'] = data.frame.iloc[all_valid].y_phq9.to_numpy(), full_p, operating['threshold']
            pred['control'], pred['outer_fold'] = control, fold
            parts.append(pred)
            print(json.dumps({'control': control, 'model': candidate['name'], 'fold': fold, 'status': 'ok'}), flush=True)
        table = pd.concat(parts, ignore_index=True)
        controls.append(table)
        original = oof[oof.model.eq(candidate['name'])]
        paired = original.merge(table, on=['sample_id', 'outer_fold'], validate='one_to_one', suffixes=('_original', '_control'))
        report['controls'][control] = {
            'candidate': candidate['name'], 'post_selection_exploratory': True,
            'metrics': binary_metrics(table.target, table.probability, table.subject_id, table.threshold),
            'original_minus_control': paired_cluster_bootstrap(paired.target_original, paired.probability_original,
                paired.probability_control, paired.subject_id_original, n_bootstrap=config.get('bootstrap_replicates', 2000), seed=config['seed'])}
        write_json(output / 'development_report.json', report)
    write_parquet(output / 'control_oof_predictions.parquet', pd.concat(controls, ignore_index=True))


def run_study(dataset, output, config_path):
    os.umask(0o077)
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise FileExistsError('Preserve old runs; choose a new output directory.')
    config = yaml.safe_load(Path(config_path).read_text())
    if config['window'] != 14:
        raise ValueError('This amendment fixes the primary window at 14 days.')
    if config.get('mtm_source_dir'):
        config['mtm_source_dir'] = str(Path(config['mtm_source_dir']).resolve())
    try:
        import torch
    except ModuleNotFoundError:
        pass  # Linear-only runs do not require an optional neural runtime.
    else:
        torch.set_num_threads(config.get('torch_threads', 2))
    before = _prior_fingerprints(dataset.parent)
    secure_dir(output)
    write_json(output / 'execution_lock.json', {'config': config, 'config_sha256': sha256(Path(config_path)),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'prior_artifacts': before,
        'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'old_test_must_not_be_evaluated': True})
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(limits=config.get('torch_threads', 2)):
            report = run_comparison(load_temporal_data(dataset, config['window'], development_end=config['development_end']), config, output)
    finally:
        after = _prior_fingerprints(dataset.parent)
        write_json(output / 'preservation_audit.json', {'all_prior_artifacts_unchanged': before == after, 'checked_files': len(before),
                                                      'old_test_loaded_or_evaluated': False})
    if before != after:
        raise ValueError('An existing artifact changed during this execution.')
    return {'status': report['status'], 'models': len(report['models']), 'test_evaluated': False,
            'final_models': {key: value.get('selected_name') for key, value in report['final_models'].items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Confidential development-only chronological model comparison')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/temporal_v5.yaml'))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_study(args.dataset, args.output, args.config)), flush=True)
    except Exception as exc:
        # Exception strings/tracebacks from external libraries can echo records.
        print(json.dumps({'status': 'failed', 'exception_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
