"""Research-only local inference from a trusted frozen development model."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .benchmarks import BenchmarkConfig, _eligible, _probability, _calibrated
from .privacy import sha256


def predict_research(frame, model_dir):
    model_dir = Path(model_dir)
    spec = json.loads((model_dir/'selection.json').read_text())
    checkpoint = model_dir/'frozen_development_model.joblib'
    if not checkpoint.exists() or sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('trusted_frozen_checkpoint_hash_mismatch')
    cfg = BenchmarkConfig(**spec['config'])
    frozen = joblib.load(checkpoint)
    features = spec['feature_columns']
    if not set(features).issubset(frame):
        raise ValueError('inference_feature_schema_mismatch')
    available = _eligible(frame, features, cfg)
    probability = np.full(len(frame), np.nan)
    if available.any():
        X = frame.loc[available, frozen['feature_columns']].to_numpy(dtype=float, copy=True)
        X[~np.isfinite(X)] = np.nan
        probability[available] = _calibrated(frozen['calibrator'], _probability(frozen['model'], X))
    return pd.DataFrame({'sample_id': frame['sample_id'].to_numpy(), 'probability': probability,
                         'flag': np.where(np.isfinite(probability), probability >= frozen['threshold'], np.nan),
                         'insufficient_data': ~available, 'target': spec['target_column'],
                         'model_version': spec['checkpoint_sha256'],
                         'use_status': 'research_only_not_clinically_validated'})
