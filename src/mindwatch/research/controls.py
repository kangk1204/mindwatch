"""Prespecified software/research controls using development data only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .pipeline import load_model_frame
from .sequence_run import sequence_frame
from .benchmarks import BenchmarkConfig, run_development
from .neural import past_only_delta, flatten_sequence
from .privacy import write_json


def permute_trajectories(frame, target, seed=42):
    out = frame.copy()
    if not out.partition.eq('development').all():
        raise ValueError('controls_development_only')
    groups, strata = {}, {}
    for pid, group in out.groupby('subject_id'):
        order = group.sort_values('index_date').index.to_numpy()
        groups[pid] = order
        pattern = tuple(group.loc[order, 'wave']) if 'wave' in group else (len(order),)
        strata.setdefault(pattern, []).append(pid)
    rng, permutable = np.random.default_rng(seed), 0
    for ids in strata.values():
        if len(ids) < 2:
            continue
        shuffled = rng.permutation(ids)
        original = {pid: out.loc[groups[pid], target].to_numpy(copy=True) for pid in ids}
        for dst, src in zip(ids, shuffled):
            out.loc[groups[dst], target] = original[src]
        permutable += len(ids)
    return out, {'permutable_subjects': permutable, 'method': 'whole_label_trajectory_permutation_within_visit_pattern', 'seed': seed}


def shuffle_time_order(frame, columns, length=14, n_demographics=0, seed=42):
    out = frame.copy()
    names = columns[:-n_demographics] if n_demographics else columns
    arr = out[names].to_numpy(dtype=np.float32, copy=True)
    n_features = len(names) // (3 * length)
    width = length * n_features
    values = arr[:, :width].reshape(-1, length, n_features)
    mask = arr[:, width:2*width].reshape(-1, length, n_features)
    rng = np.random.default_rng(seed)
    for i in range(len(values)):
        order = rng.permutation(length)
        values[i], mask[i] = values[i, order].copy(), mask[i, order].copy()
    delta = past_only_delta(mask)
    out.loc[:, names] = flatten_sequence(values, mask, delta)
    return out


def run_controls(dataset, output, seed=42):
    output = Path(output)
    frame, features = load_model_frame(dataset, feature_set='sensor')
    permuted, audit = permute_trajectories(frame, 'y_phq9', seed)
    cfg = BenchmarkConfig(models=('catboost',), parameter_grids={'catboost':[{'depth':3,'l2_leaf_reg':10.}]},
                          protocol_verified=False, protocol_unresolved=('Negative control only; metadata unresolved.',))
    perm_result = run_development(permuted, features, 'y_phq9', output/'permuted_labels', cfg)
    seq, columns, n_demo = sequence_frame(dataset)
    ordered = shuffle_time_order(seq, columns, n_demographics=n_demo, seed=seed)
    neural = BenchmarkConfig(models=('grud',), neural_epochs=20, n_demographics=n_demo,
                              protocol_verified=False, protocol_unresolved=('Temporal-order destruction control only.',))
    time_result = run_development(ordered, columns, 'y_phq9', output/'shuffled_time', neural)
    summary = {'status':'development_controls_completed','label_permutation':audit,
               'permutation_fit_status':perm_result['status'],'time_shuffle_fit_status':time_result['status'],
               'interpretation':'chance controls do not prove absence of every leakage path', 'temporal_test_evaluated':False}
    write_json(output/'summary.json', summary)
    return summary
