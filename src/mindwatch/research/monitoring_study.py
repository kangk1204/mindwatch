"""Versioned, development-only symptomatic-user monitoring study.

No source values or participant identifiers are printed. Earlier experiments and
the previously evaluated temporal-test participants are never used as new tests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import yaml

from .evaluation import (binary_metrics, holm_adjust, paired_cluster_bootstrap,
                         participant_weights, select_sensitivity_threshold)
from .privacy import secure_dir, sha256, write_json, write_parquet


def _print(**record):
    print(json.dumps(record), flush=True)


def _checkpoint(path, value):
    from mindwatch.pipeline import private_output
    path = private_output(path)
    temporary = Path(str(path) + '.tmp')
    try:
        joblib.dump(value, temporary)
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sources(root):
    paths = sorted((Path(__file__).parent).glob('*.py'))
    paths += sorted((root / 'tests').glob('*.py'))
    paths += sorted((root / 'configs').glob('*.yaml'))
    paths += [root / 'pyproject.toml', root / 'requirements.lock.txt']
    paths += [root / 'docs/AMENDMENT_20261022_v6_MONITORING.md']
    return {(str(p.relative_to(root)) if p.is_relative_to(root) else 'package/'+p.name): sha256(p)
            for p in paths if p.is_file()}


def _preserved(root, dataset):
    paths = list(Path(dataset).glob('*'))
    for pattern in ('artifacts/private/temporal_v5*/*',
                    'artifacts/private/monitoring_v6/*',
                    'artifacts/private/primary_*_age_corrected_v3/*',
                    'artifacts/private/supplemental_*_phq_v3/*',
                    'artifacts/private/dataset_primary_age_corrected_v3/split*',
                    'artifacts/reports/temporal_v5*/*',
                    'artifacts/reports/nonvoice_final_v4/*'):
        paths += list(root.glob(pattern))
    paths += [root / 'FOLLOWUP_STATUS_v4.md', root / 'TEMPORAL_STATUS_v5_1.md']
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def _verify_preserved(before):
    return {'checked_files': len(before),
            'all_prior_artifacts_unchanged': all(Path(p).is_file() and sha256(Path(p)) == h for p, h in before.items()),
            'original_test_loaded_or_evaluated': False}


def _support(data, rows, branch, target, task, config, neural=False):
    f = data.frame.iloc[np.asarray(rows, dtype=int)]
    y = pd.to_numeric(f[target], errors='coerce').to_numpy(dtype=float)
    available = f.history_available.to_numpy(dtype=bool,copy=True)
    if branch in ('B', 'D'):
        available &= f.sensor_available.to_numpy(dtype=bool)
    if branch in ('C', 'D'):
        available &= f.ema_available.to_numpy(dtype=bool)
    valid = available & np.isfinite(y)
    selected = f.loc[valid]
    yy = y[valid]
    minimum = config.get('min_neural_people', 100) if neural else config.get('min_fit_people', 50)
    per_class = config.get('min_neural_per_class', 20) if neural else config.get('min_fit_per_class', 10)
    good = selected.subject_id.nunique() >= minimum
    if task == 'classification':
        good &= all(selected.loc[yy == label, 'subject_id'].nunique() >= per_class for label in (0, 1))
    return np.asarray(rows, dtype=int)[valid], bool(good)


def regression_metrics(y, score, subjects):
    y, p, g = np.asarray(y, float), np.asarray(score, float), np.asarray(subjects)
    known = np.isfinite(y)
    keep = known & np.isfinite(p)
    all_w = participant_weights(g[known]) if known.any() else np.array([])
    result = {'n_labelled_visits': int(known.sum()), 'n_predicted_visits': int(keep.sum()),
              'n_labelled_participants': int(pd.Series(g[known]).nunique()),
              'n_predicted_participants': int(pd.Series(g[keep]).nunique()),
              'coverage': float(all_w[np.isfinite(p[known])].sum()/all_w.sum()) if len(all_w) else None,
              'mae': None, 'rmse': None, 'calibration_intercept': None, 'calibration_slope': None}
    if keep.any():
        w = participant_weights(g[keep])
        result.update(mae=float(np.average(abs(y[keep]-p[keep]), weights=w)),
                      rmse=float(np.sqrt(np.average((y[keep]-p[keep])**2, weights=w))))
        design = np.column_stack([np.ones(keep.sum()), p[keep]])
        if np.linalg.matrix_rank(design) == 2:
            beta = np.linalg.lstsq(design*np.sqrt(w[:, None]), y[keep]*np.sqrt(w), rcond=None)[0]
            result.update(calibration_intercept=float(beta[0]), calibration_slope=float(beta[1]))
    return result


def _metrics(table, task):
    if task == 'regression':
        return regression_metrics(table.target, table.score, table.subject_id)
    threshold = table.threshold.to_numpy(dtype=float)
    return binary_metrics(table.target, table.probability, table.subject_id,
                          threshold if np.isfinite(threshold).all() else None)


def _specs(config, task='classification', auxiliary=False):
    output = []
    for branch in config['feature_sets']:
        models = ['ridge', 'catboost'] if task == 'regression' else ['elasticnet', 'catboost']
        if auxiliary:
            models = ['elasticnet']
        elif task == 'classification' and branch in ('B', 'D'):
            models += ['grud', 'tcn']
        for model in models:
            output.append({'feature_set': branch, 'model': model, 'name': f'{branch}_{model}'})
    return output


def _parameter_complexity(model, parameters):
    """Explicit deterministic preference inside the one-standard-error band."""
    if model == 'ridge':
        key = (-float(parameters.get('alpha', 10.)),)
    elif model == 'elasticnet':
        key = (float(parameters.get('C', .01)), -float(parameters.get('l1_ratio', 0.)))
    elif model == 'catboost':
        key = (int(parameters.get('depth', 3)), int(parameters.get('iterations', 300)),
               -float(parameters.get('l2_leaf_reg', 10.)))
    elif model in ('grud', 'tcn'):
        key = (int(parameters.get('hidden_dim', 64)), int(parameters.get('epochs', 40)))
    else:
        raise ValueError('Unknown model complexity ordering')
    return (*key, json.dumps(parameters, sort_keys=True))


def tune_candidate(data, spec, fit_rows, config, target='y_phq9', task='classification', mode='known_user'):
    from .monitoring_models import fit_monitoring_bundle
    from .monitoring_splits import monitoring_forward_splits
    rows = np.asarray(fit_rows, dtype=int)
    frame = data.frame.iloc[rows].reset_index(drop=True)
    default = config['warmup_regression' if task == 'regression' else 'warmup_classification']
    grid = config['parameter_grids'][spec['model']]
    fallback = grid[0] if spec['model'] != default['model'] else default['parameters']
    result = {'parameters': fallback, 'selection_status': 'warmup_fixed_model',
              'valid_inner_folds': 0, 'mean_score': None, 'se_score': None, 'trials': []}
    try:
        splits = monitoring_forward_splits(frame, n_splits=config['inner_folds'],
            gap_days=config['gap_days'], min_train_fraction=config.get('inner_min_train_fraction', .4), mode=mode)
    except ValueError:
        return result
    valid = []
    for fit, validation in splits:
        tr, good = _support(data, rows[fit], spec['feature_set'], target, task, config,
                            neural=spec['model'] in ('grud', 'tcn'))
        yy = pd.to_numeric(data.frame.iloc[rows[validation]][target], errors='coerce')
        # Score the complete routed input pathway. The full branch must have
        # honest training support, and all labelled history-available validation
        # rows remain eligible even when their optional modalities are absent.
        va, va_good = _support(data, rows[validation], 'A', target, task,
                              {**config, 'min_fit_people': 2, 'min_fit_per_class': 1})
        if good and va_good and yy.notna().any():
            valid.append((rows[fit], va))
    result['valid_inner_folds'] = len(valid)
    if len(valid) < config['inner_folds']:
        return result
    for parameters in grid:
        scores, failures = [], []
        for train, validation in valid:
            try:
                fit_config = {**config, 'fit_end_exclusive':
                    str(data.frame.iloc[validation].index_date.min()-pd.Timedelta(days=config['gap_days']))}
                fitted = fit_monitoring_bundle(data,train,np.array([],dtype=int),
                    {**spec,'parameters':parameters},fit_config,target_column=target,task=task)
                prediction = fitted.predict(data,validation)
                pp = prediction['score' if task=='regression' else 'probability'].to_numpy(dtype=float)
                if not np.isfinite(pp).all():
                    raise ValueError('Inner routed model must predict all history-available evaluation rows.')
                if any(v.get('status')=='fit_failed' for v in fitted.metadata.get('branches',{}).values()):
                    raise ValueError('Inner branch fitting failed.')
                yy = data.frame.iloc[validation][target].to_numpy(dtype=float)
                weights = participant_weights(data.frame.iloc[validation].subject_id)
                score = -float(np.average(abs(yy-pp), weights=weights)) if task == 'regression' else float(roc_auc_score(yy, pp, sample_weight=weights))
                scores.append(score)
            except Exception as exc:
                failures.append(type(exc).__name__)
        trial = {'parameters': parameters, 'scores': scores, 'failures': failures}
        if len(scores) == config['inner_folds']:
            trial.update(mean_score=float(np.mean(scores)), se_score=float(np.std(scores, ddof=1)/np.sqrt(len(scores))))
        result['trials'].append(trial)
    good = [x for x in result['trials'] if 'mean_score' in x]
    if good:
        best = min(good, key=lambda r: (-r['mean_score'], _parameter_complexity(spec['model'], r['parameters'])))
        chosen = min((x for x in good if x['mean_score'] >= best['mean_score']-best['se_score']),
                     key=lambda x: _parameter_complexity(spec['model'], x['parameters']))
        result.update(parameters=chosen['parameters'], selection_status='nested_inner_selected',
                      mean_score=chosen['mean_score'], se_score=chosen['se_score'])
    return result


def choose_candidate(records, branch, config, task):
    candidates = [r for r in records if r['feature_set'] == branch]
    tuned = [r for r in candidates if r['tuning']['selection_status'] == 'nested_inner_selected']
    ranking = {'ridge': 0, 'elasticnet': 0, 'catboost': 1, 'grud': 2, 'tcn': 3}
    if not tuned:
        default = config['warmup_regression' if task == 'regression' else 'warmup_classification']
        chosen = next(r for r in candidates if r['model'] == default['model'])
        return {**chosen, 'selection_status': 'warmup_fixed_model'}
    best = max(tuned, key=lambda r: r['tuning']['mean_score'])
    boundary = best['tuning']['mean_score']-best['tuning']['se_score']
    chosen = min((r for r in tuned if r['tuning']['mean_score'] >= boundary), key=lambda r: ranking[r['model']])
    return {**chosen, 'selection_status': 'nested_inner_selected', 'one_se_boundary': boundary}


def _prediction_table(data, rows, predictions, mode, fold, name, branch, target, task, status):
    out = data.frame.iloc[rows][['sample_id', 'subject_id', 'index_date', 'baseline_phq9',
        'sensor_available', 'ema_available', 'quality__sensor_observed_days']].reset_index(drop=True).copy()
    for field in ('age_group', 'sex', 'region_group'):
        if field in data.frame:
            out[field] = data.frame.iloc[rows][field].to_numpy()
    if len(predictions) != len(out):
        raise ValueError('Prediction output must preserve every evaluation row.')
    for col in predictions:
        out[col] = predictions[col].to_numpy()
    out['target'] = data.frame.iloc[rows][target].to_numpy(dtype=float)
    out['mode'], out['outer_fold'], out['model'], out['feature_set'] = mode, fold, name, branch
    out['endpoint'], out['task'], out['selection_status'] = target, task, status
    out['partition'] = 'development'
    out['data_version'] = data.provenance.get('data_manifest_sha256','synthetic_or_manifest_unavailable')
    if 'threshold' not in out:
        out['threshold'] = np.nan
    return out


def predict_monitoring(classifier_bundle, regressor_bundle, data, rows):
    """Local joint inference with separate classifier/regressor provenance.

    Data must obey the prepared past-only extraction contract. Current labels are
    never requested or read, and every requested input row remains in the output.
    """
    if classifier_bundle.task != 'classification' or regressor_bundle.task != 'regression':
        raise ValueError('Separate classification and regression bundles required.')
    probability=classifier_bundle.predict(data,rows).add_prefix('classification_')
    score=regressor_bundle.predict(data,rows).add_prefix('regression_')
    if len(probability)!=len(rows) or len(score)!=len(rows):
        raise ValueError('Inference must retain every input row.')
    out=data.frame.iloc[rows][['sample_id','index_date']].reset_index(drop=True).copy()
    out=pd.concat([out,probability,score],axis=1)
    out['classification_target']=getattr(classifier_bundle,'metadata',{}).get('target_column','unspecified')
    out['regression_target']=getattr(regressor_bundle,'metadata',{}).get('target_column','unspecified')
    out['data_version']=data.provenance.get('data_manifest_sha256','manifest_unavailable')
    return out


def prepare(dataset, output, config):
    from .monitoring_data import load_monitoring_data
    from .monitoring_splits import make_monitoring_fold
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh versioned output; no completed experiment is overwritten.')
    secure_dir(output)
    root = Path(config.get('private_workspace', Path(dataset).resolve().parent)).resolve()
    before = _preserved(root, dataset)
    data = load_monitoring_data(dataset, window=config['window'], entry_threshold=config['entry_threshold'], development_end=config['development_end'])
    manifests, folds = [], []
    for mode in config['validation_modes']:
        for index, (start, end) in enumerate(config['outer_windows']):
            split = make_monitoring_fold(data.frame, start, end, mode=mode,
                gap_days=config['gap_days'], calibration_days=config['calibration_days'])
            folds.append({'mode': mode, 'fold': index, 'metadata': split['metadata']})
            for role in ('fit', 'calibration', 'validation'):
                rows = data.frame.iloc[split[role]][['sample_id', 'subject_id', 'index_date', 'label_available_at']].copy()
                rows['mode'], rows['outer_fold'], rows['role'] = mode, index, role
                manifests.append(rows)
    write_parquet(output/'cohort_manifest.parquet', data.frame[['sample_id','subject_id','index_date','baseline_phq9','label_available_at']])
    write_parquet(output/'development_cv_manifest.parquet', pd.concat(manifests, ignore_index=True))
    lock = {'config': config, 'private_workspace': str(root), 'source_hashes': _sources(root), 'preserved_hashes': before,
            'dataset': str(Path(dataset).resolve()), 'provenance': data.provenance,
            'folds': folds, 'analysis': 'exploratory_development_reanalysis',
            'original_test_loaded_or_evaluated': False}
    write_json(output/'execution_lock.json', lock)
    write_json(output/'development_report.json', {'status': 'prepared', 'provenance': data.provenance, 'folds': folds})
    _print(stage='prepared', folds=len(folds), old_test_evaluated=False)
    return data


def _load_locked(output):
    output = Path(output)
    lock = json.loads((output/'execution_lock.json').read_text())
    root = Path(lock.get('private_workspace', Path(lock.get('dataset', output)).resolve().parent))
    if _sources(root) != lock['source_hashes']:
        raise ValueError('Execution sources/configs changed after preparation; preserve this run and prepare a new version.')
    audit = _verify_preserved(lock['preserved_hashes'])
    if not audit['all_prior_artifacts_unchanged']:
        raise ValueError('Prior input/artifact hash changed.')
    return lock


def _persist(data, cal, valid, target, task):
    scale = {'y_phq9':'phq9','y_gad7':'gad7','y_isi':'isi','phq9':'phq9'}[target]
    key = f'history__previous_{scale}'
    design = data.designs['A']
    pos = design['tabular_names'].index(key)
    value = design['tabular'][:,pos]
    if task == 'regression':
        return pd.DataFrame({'score':value[valid], 'used_modalities':'A', 'fallback_reason':'persistence', 'model_version':'persistence_v6'})
    cal, valid = np.asarray(cal, dtype=int), np.asarray(valid, dtype=int)
    # Calibration alone may supply outcomes. Evaluation labels are never read,
    # including when the caller supplies unlabelled prediction queries.
    y = pd.to_numeric(data.frame.iloc[cal][target], errors='coerce').to_numpy(dtype=float)
    if np.isinf(y).any() or np.any(np.isfinite(y) & ~np.isin(y, [0, 1])):
        raise ValueError('Persistence calibration needs binary labels')
    keep = np.isfinite(y) & np.isfinite(value[cal])
    ci, cy = cal[keep], y[keep]
    groups = data.frame.iloc[ci].subject_id.to_numpy()
    threshold = None
    if all(pd.Series(groups[cy == c]).nunique() >= 10 for c in (0,1)):
        model = LogisticRegression(C=1e6, max_iter=1000).fit(value[ci,None], cy, sample_weight=participant_weights(groups))
        pp = np.full(len(valid), np.nan)
        present = np.isfinite(value[valid])
        if present.any():
            pp[present] = model.predict_proba(value[valid[present],None])[:,1]
        cp = model.predict_proba(value[ci,None])[:,1]
        threshold = select_sensitivity_threshold(cy, cp, groups)['threshold']
        state = 'calibrated_persistence'
    else:
        pp = np.full(len(valid), np.nan)
        state = 'insufficient_calibration'
    return pd.DataFrame({'probability':pp, 'used_modalities':'A', 'fallback_reason':'persistence',
        'model_version':'persistence_v6','threshold':threshold, 'calibration_status':state})


def develop(output):
    from .monitoring_data import load_monitoring_data
    from .monitoring_models import fit_monitoring_bundle
    from .monitoring_splits import make_monitoring_fold
    output = Path(output)
    lock = _load_locked(output)
    report_path = output/'development_report.json'
    if json.loads(report_path.read_text())['status'] != 'prepared':
        raise FileExistsError('Preserve running/completed/failed attempts; use a new version for reruns.')
    config = lock['config']
    try:
        import torch
    except ModuleNotFoundError:
        pass  # Linear-only runs do not require an optional neural runtime.
    else:
        torch.set_num_threads(config.get('torch_threads', 2))
    data = load_monitoring_data(lock['dataset'], window=config['window'], entry_threshold=config['entry_threshold'], development_end=config['development_end'])
    report = {'status':'running', 'provenance':data.provenance, 'config':config,
              'folds':lock['folds'], 'records':[], 'selectors':[], 'final_models':[],
              'failures':[], 'test_evaluated':False, 'clinical_validation':False}
    tables, selected_tables = [], []
    begun = time.monotonic()
    try:
        for mode in config['validation_modes']:
            for target, task, auxiliary in [('y_phq9','classification',False),('phq9','regression',False),
                                            ('y_gad7','classification',True),('y_isi','classification',True)]:
                specs = _specs(config, task, auxiliary)
                for fold, (start, end) in enumerate(config['outer_windows']):
                    split = make_monitoring_fold(data.frame,start,end,mode=mode,gap_days=config['gap_days'],calibration_days=config['calibration_days'])
                    fit, cal, valid = (split[r] for r in ('fit','calibration','validation'))
                    tuned = [{**s, 'tuning':tune_candidate(data,s,fit,config,target,task,mode)} for s in specs]
                    # Freeze choices BEFORE any outer model is fitted or scored.
                    selected = {b:choose_candidate(tuned,b,config,task) for b in config['feature_sets']}
                    write_json(output/f'selection_{mode}_{target}_{fold}.json', selected)
                    predicted = {}
                    for record in tuned:
                        tag = f'{mode}_{target}_{fold}_{record["name"]}'
                        try:
                            selection = {**record, 'parameters':record['tuning']['parameters']}
                            fit_config = {**config,'fit_end_exclusive':str(pd.Timestamp(start)-pd.Timedelta(days=56))}
                            bundle = fit_monitoring_bundle(data,fit,cal,selection,fit_config,target_column=target,task=task)
                            predictions = bundle.predict(data,valid)
                            table = _prediction_table(data,valid,predictions,mode,fold,record['name'],record['feature_set'],target,task,record['tuning']['selection_status'])
                            metrics = _metrics(table,task)
                            # No metrics/OOF success is committed until checkpoint succeeds.
                            _checkpoint(output/f'{tag}.joblib',bundle)
                            details = {**record,'mode':mode,'endpoint':target,'task':task,'fold':fold,
                                'status':'ok','metrics':metrics,'bundle_metadata':getattr(bundle,'metadata',{})}
                            report['records'].append(details)
                            branch_errors=[b for b,v in details['bundle_metadata'].get('branches',{}).items()
                                           if v.get('status')=='fit_failed']
                            prediction_errors=table.fallback_reason.astype(str).str.contains('prediction_failed').any() if 'fallback_reason' in table else False
                            if branch_errors or prediction_errors:
                                details['status']='routed_with_failures'
                                report['failures'].append({'stage':'routed_candidate','tag':tag,'failed_branches':branch_errors,
                                                           'prediction_failed':bool(prediction_errors)})
                            tables.append(table)
                            predicted[record['name']] = table
                        except Exception as exc:
                            details = {**record,'mode':mode,'endpoint':target,'task':task,'fold':fold,
                                       'status':'failed','exception_type':type(exc).__name__}
                            report['records'].append(details)
                            report['failures'].append({'stage':'candidate','tag':tag,'exception_type':type(exc).__name__})
                        write_json(report_path,report)
                        _print(stage='candidate',mode=mode,endpoint=target,fold=fold,model=record['name'],status=details['status'])
                    for branch, choice in selected.items():
                        record = {'mode':mode,'endpoint':target,'task':task,'fold':fold,'feature_set':branch,
                                  'chosen_name':choice['name'],'selection_status':choice['selection_status']}
                        if choice['name'] in predicted:
                            table = predicted[choice['name']].copy()
                            table['candidate_model'] = table.model
                            table['model'] = f'selected_{branch}'
                            table['selection_status'] = choice['selection_status']
                            selected_tables.append(table)
                            record.update(status='ok',metrics=_metrics(table,task))
                        else:
                            record['status']='selected_candidate_failed'
                            report['failures'].append({'stage':'selector',**record})
                        report['selectors'].append(record)
                    persistence = _persist(data,cal,valid,target,task)
                    tables.append(_prediction_table(data,valid,persistence,mode,fold,'persistence','A',target,task,'prespecified_persistence'))
        write_parquet(output/'candidate_oof.parquet',pd.concat(tables,ignore_index=True))
        write_parquet(output/'selected_oof.parquet',pd.concat(selected_tables,ignore_index=True))
        _final_snapshots(data,output,config,report)
        _controls(data,output,config,report)
        _prognosis(data,output,config,report)
        report['status']='completed_development_only' if not report['failures'] else 'completed_with_failures'
    except Exception as exc:
        report['status']='failed'
        report['failures'].append({'stage':'runner','exception_type':type(exc).__name__})
        raise
    finally:
        report['elapsed_seconds']=round(time.monotonic()-begun,2)
        audit=_verify_preserved(lock['preserved_hashes'])
        write_json(output/'preservation_audit.json',audit)
        if not audit['all_prior_artifacts_unchanged']:
            report['status']='failed_preservation'
        if _sources(Path(lock.get('private_workspace', Path(lock.get('dataset', output)).resolve().parent))) != lock['source_hashes']:
            report['status']='failed_source_drift'
        write_json(report_path,report)
    _print(stage='complete',status=report['status'],elapsed_seconds=report['elapsed_seconds'])
    return report


def _final_snapshots(data,output,config,report):
    from .monitoring_models import fit_monitoring_bundle
    from .monitoring_splits import make_monitoring_fold
    start=pd.Timestamp(config['snapshot_date'])
    split=make_monitoring_fold(data.frame,start,start+pd.offsets.MonthBegin(1),mode='known_user',gap_days=14,calibration_days=28)
    fit,cal=split['fit'],split['calibration']
    for target,task in [('y_phq9','classification'),('phq9','regression'),('y_gad7','classification'),('y_isi','classification')]:
        specs=_specs(config,task,target in ('y_gad7','y_isi'))
        tuned=[{**s,'tuning':tune_candidate(data,s,fit,config,target,task)} for s in specs]
        for branch in config['feature_sets']:
            choice=choose_candidate(tuned,branch,config,task)
            selection={**choice,'parameters':choice['tuning']['parameters']}
            bundle=fit_monitoring_bundle(data,fit,cal,selection,{**config,'fit_end_exclusive':str(start-pd.Timedelta(days=56))},target_column=target,task=task)
            name=f'final_{target}_{branch}'
            _checkpoint(output/f'{name}.joblib',bundle)
            report['final_models'].append({'name':name,'selection':selection,'snapshot_date':str(start),
                'fit_visits':len(fit),'fit_people':data.frame.iloc[fit].subject_id.nunique(),
                'calibration_visits':len(cal),'calibration_people':data.frame.iloc[cal].subject_id.nunique(),
                'new_independent_test_performance':None,'metadata':getattr(bundle,'metadata',{})})
            failed=[b for b,v in getattr(bundle,'metadata',{}).get('branches',{}).items() if v.get('status')=='fit_failed']
            if failed:
                report['failures'].append({'stage':'snapshot','snapshot':name,'failed_branches':failed})
            write_json(output/'development_report.json',report)
            _print(stage='snapshot',endpoint=target,branch=branch)


def _controls(data,output,config,report):
    from .monitoring_models import fit_monitoring_bundle
    from .monitoring_splits import make_monitoring_fold
    from .sequence_arrays import past_only_delta,flatten_sequence
    if not config.get('controls'):
        return
    rng=np.random.default_rng(config['seed'])
    designs=deepcopy(data.designs)
    branch=config['controls']['feature_set']
    xx=designs[branch]['sequence']
    n_channels=len(data.provenance.get('channels',list(range(11))))
    width=config['window']*n_channels
    values=xx[:,:width].reshape(-1,config['window'],n_channels).copy()
    mask=xx[:,width:2*width].reshape(values.shape).copy()
    for i in range(len(xx)):
        order=rng.permutation(config['window'])
        values[i],mask[i]=values[i,order],mask[i,order]
    xx[:,:3*width]=flatten_sequence(values,mask,past_only_delta(mask))
    shuffled=replace(data,designs=designs)
    tables=[]
    for fold,(start,end) in enumerate(config['outer_windows']):
        split=make_monitoring_fold(data.frame,start,end,mode='known_user')
        fit,cal,valid=(split[r] for r in ('fit','calibration','validation'))
        spec={'feature_set':branch,'model':'tcn','name':f'{branch}_tcn'}
        tuning=tune_candidate(data,spec,fit,config)
        selection={**spec,'parameters':tuning['parameters']}
        for kind,source,seeds in [('shuffled_order',shuffled,config['seeds'])]+[(f'seed_{s}',data,[s]) for s in config['seeds']]:
            try:
                bundle=fit_monitoring_bundle(source,fit,cal,selection,{**config,'seeds':seeds},target_column='y_phq9')
                pred=bundle.predict(source,valid)
                table=_prediction_table(data,valid,pred,'known_user',fold,kind,branch,'y_phq9','classification','prespecified_exploratory_control')
                _checkpoint(output/f'control_{kind}_{fold}.joblib',bundle)
                tables.append(table)
                failed=[b for b,v in getattr(bundle,'metadata',{}).get('branches',{}).items() if v.get('status')=='fit_failed']
                prediction_failed=table.fallback_reason.astype(str).str.contains('prediction_failed').any()
                if failed or prediction_failed:
                    report['failures'].append({'stage':'control','control':kind,'fold':fold,'failed_branches':failed,
                                               'prediction_failed':bool(prediction_failed)})
            except Exception as exc:
                report['failures'].append({'stage':'control','control':kind,'fold':fold,'exception_type':type(exc).__name__})
            _print(stage='control',control=kind,fold=fold)
    if tables:
        write_parquet(output/'control_oof.parquet',pd.concat(tables,ignore_index=True))
    report['controls']={'prespecified_candidate':f'{branch}_tcn','status':'completed' if not any(f['stage']=='control' for f in report['failures']) else 'failed',
                        'scope':'sensor value/mask order jointly shuffled; delta recomputed; static H/EMA retained; seed analyses exploratory'}


def _prognosis(data,output,config,report):
    from .monitoring_models import fit_base_estimator,predict_base
    pf=getattr(data,'prognosis_frame',None)
    if pf is None or pf.empty:
        report['prognosis']={'status':'no_available_pairs'}
        return
    pf=pf.copy()
    pf['assessment_date']=pf.index_date
    # Forecasting begins only after the index questionnaire is actually known.
    # Date-only exports conservatively release it on the next calendar day.
    pf['index_date']=pf['index_label_available_at']
    pf['prediction_time']=pf.index_date
    current='history__forecast_current_phq9'
    forecast_designs={}
    for branch in ('A','B'):
        original=data.prognosis_designs[branch]
        columns=[i for i,n in enumerate(original['tabular_names'])
                 if n==current or (branch=='B' and n.startswith(('sensor__','quality__')))]
        forecast_designs[branch]={'tabular':original['tabular'][:,columns],
                                  'tabular_names':[original['tabular_names'][i] for i in columns]}
    source=replace(data,frame=pf,designs=forecast_designs)
    # Current PHQ is allowed ONLY in this separately typed future task.
    predictions,records=[],[]
    for fold,(start,end) in enumerate(config['outer_windows']):
        split=forecast_fold(pf,start,end)
        fit,cal,valid=(split[r] for r in ('fit','calibration','validation'))
        if not len(valid):
            records.append({'fold':fold,'status':'no_mature_validation_pairs',
                            'fit_pairs_with_released_labels':len(fit),
                            'validation_pairs':0})
            continue
        for target,task in [('future_phq9','regression'),('future_ge10','classification'),('worsening_ge5','classification')]:
            for branch,model,parameters in [('A','ridge',{'alpha':10.}),('B','elasticnet',{'alpha':.1,'l1_ratio':.5})] if task=='regression' else [('A','elasticnet',{'C':.01,'l1_ratio':0.}),('B','elasticnet',{'C':.01,'l1_ratio':0.})]:
                rows=np.asarray(fit)
                if branch=='B': rows=rows[pf.iloc[rows].sensor_available.to_numpy(bool)]
                y=pf.iloc[rows][target].to_numpy(float)
                ok=len(rows)>=config['prognosis']['min_fit_pairs']
                if task=='classification': ok &= all(np.sum(y==v)>=20 for v in (0,1))
                record={'fold':fold,'target':target,'branch':branch,'fit_pairs':len(rows),
                        'status':'eligible' if ok else 'insufficient_fit_support'}
                if task=='classification': record['fit_events']=int(np.sum(y==1))
                if not ok:
                    records.append(record);continue
                try:
                    # EN regression is the fixed sparse sensor prognosis comparator.
                    if task=='regression' and branch=='B':
                        from sklearn.linear_model import ElasticNet
                        from sklearn.impute import SimpleImputer
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.pipeline import make_pipeline
                        estimator=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),ElasticNet(alpha=.1,l1_ratio=.5,max_iter=5000))
                        x=source.designs[branch]['tabular']
                        weights=participant_weights(pf.iloc[rows].subject_id)
                        estimator.fit(x[rows],y,standardscaler__sample_weight=weights,elasticnet__sample_weight=weights)
                        eligible=valid[pf.iloc[valid].sensor_available.to_numpy(bool)]
                        score=np.full(len(valid),np.nan)
                        pos={r:i for i,r in enumerate(valid)}
                        if len(eligible):
                            for r,p in zip(eligible,estimator.predict(x[eligible])):score[pos[r]]=p
                        _checkpoint(output/f'prognosis_{fold}_{target}_{branch}.joblib',estimator)
                    else:
                        base=fit_base_estimator(source,branch,model,parameters,rows,
                            {**config,'allow_current_state_for_forecast':True,
                             'fit_end_exclusive':str(pd.Timestamp(start)-pd.Timedelta(days=56))},
                            target_column=target,task=task)
                        xvalid=valid if branch=='A' else valid[pf.iloc[valid].sensor_available.to_numpy(bool)]
                        if not len(xvalid):
                            record['status']='no_available_validation_inputs';records.append(record);continue
                        raw=predict_base(base,source,branch,xvalid,task=task)
                        score=np.full(len(valid),np.nan)
                        positions={r:i for i,r in enumerate(valid)}
                        threshold=None
                        if task=='classification':
                            cr=cal if branch=='A' else cal[pf.iloc[cal].sensor_available.to_numpy(bool)]
                            cy=pf.iloc[cr][target].to_numpy(float)
                            cal_groups=pf.iloc[cr].subject_id.to_numpy()
                            cal_people={str(v):int(pd.Series(cal_groups[cy==v]).nunique()) for v in (0,1)}
                            record['calibration_class_participants']=cal_people
                            if not all(cal_people[str(v)]>=10 for v in (0,1)):
                                record['status']='insufficient_calibration_support';records.append(record);continue
                            cp=predict_base(base,source,branch,cr,task=task)
                            calibrator=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(cp,1e-7,1-1e-7))[:,None],cy,sample_weight=participant_weights(pf.iloc[cr].subject_id))
                            raw=calibrator.predict_proba(logit(np.clip(raw,1e-7,1-1e-7))[:,None])[:,1]
                            cp=calibrator.predict_proba(logit(np.clip(cp,1e-7,1-1e-7))[:,None])[:,1]
                            threshold=select_sensitivity_threshold(cy,cp,pf.iloc[cr].subject_id)['threshold']
                            _checkpoint(output/f'prognosis_{fold}_{target}_{branch}_calibrator.joblib',calibrator)
                        for r,p in zip(xvalid,raw):score[positions[r]]=p
                        _checkpoint(output/f'prognosis_{fold}_{target}_{branch}.joblib',base)
                    table=pf.iloc[valid][['sample_id','subject_id','index_date','target_date']].copy()
                    table['target']=pf.iloc[valid][target].to_numpy(float)
                    table['probability' if task=='classification' else 'score']=score
                    table['threshold']=threshold if task=='classification' else np.nan
                    table['model']=f'{branch}_{model}';table['endpoint']=target;table['task']=task;table['outer_fold']=fold
                    predictions.append(table)
                    record.update(status='ok',metrics=_metrics(table,task))
                except Exception as exc:
                    record.update(status='failed',exception_type=type(exc).__name__)
                    report['failures'].append({'stage':'prognosis',**record})
                records.append(record)
            if task=='regression':
                table=pf.iloc[valid][['sample_id','subject_id','index_date','target_date']].copy()
                table['target']=pf.iloc[valid].future_phq9.to_numpy(float)
                table['score']=pf.iloc[valid].phq9.to_numpy(float)
                table['model']='persistence';table['endpoint']=target;table['task']=task;table['outer_fold']=fold
                predictions.append(table)
    if predictions:write_parquet(output/'prognosis_oof.parquet',pd.concat(predictions,ignore_index=True))
    report['prognosis']={'status':'completed_with_documented_support_gates','records':records,
                          'label_availability':'target_date+1day; future outcomes purged before fit/cal',
                          'calibration_policy':'future_label_release in [T-42d,T-14d); fit release<T-56d; validation by prediction date',
                          'new_independent_test':False}


def forecast_fold(frame,start,end):
    """Horizon-aware split: calibrate on recently RELEASED future outcomes.

    Requiring long-horizon calibration input dates in the same recent interval
    would necessarily produce an empty calibration set. Release times define
    fit/cal roles; prediction times define future evaluation membership.
    """
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if not frame.partition.eq('development').all():
        raise ValueError('Forecast development membership is required.')
    if frame.sample_id.duplicated().any() or frame.sample_id.isna().any():
        raise ValueError('Forecast samples must be unique.')
    dates=pd.to_datetime(frame.index_date)
    labels=pd.to_datetime(frame.label_available_at)
    targets=pd.to_datetime(frame.target_date)
    if dates.isna().any() or labels.isna().any() or (labels<dates).any() or (labels<targets).any():
        raise ValueError('Future outcome availability cannot precede prediction or outcome.')
    fit=np.flatnonzero(((dates<start-pd.Timedelta(days=56)) & (labels<start-pd.Timedelta(days=56))).to_numpy())
    cal=np.flatnonzero(((dates<start-pd.Timedelta(days=14)) &
        (labels>=start-pd.Timedelta(days=42)) & (labels<start-pd.Timedelta(days=14))).to_numpy())
    valid=np.flatnonzero(((dates>=start)&(dates<end)).to_numpy())
    if np.intersect1d(fit,cal).size or np.intersect1d(fit,valid).size or np.intersect1d(cal,valid).size:
        raise ValueError('Forecast roles overlap.')
    return {'fit':fit,'calibration':cal,'validation':valid}


def report_results(output,report_dir):
    from .monitoring_report import write_monitoring_report
    return write_monitoring_report(Path(output),Path(report_dir))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','develop','report'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--config',type=Path,default=Path('configs/monitoring_v6.yaml'))
    parser.add_argument('--report-dir',type=Path)
    args=parser.parse_args(argv)
    os.umask(0o077)
    if args.stage=='prepare':
        if args.dataset is None:parser.error('--dataset required for prepare')
        prepare(args.dataset,args.output,yaml.safe_load(args.config.read_text()))
    elif args.stage=='develop':develop(args.output)
    else:
        if args.report_dir is None:parser.error('--report-dir required for report')
        report_results(args.output,args.report_dir)


if __name__=='__main__':
    main()
