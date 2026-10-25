"""Optional frozen survey-history fallback when all recent EMA is absent.

The caller selects and fits this policy using development data. No chosen study
policy, fitted coefficients, membership or results are distributed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import features as survey_features
from . import lightweight as core

H_COLUMNS = survey_features.H_COLUMNS
FEATURE_COLUMNS = survey_features.FEATURE_COLUMNS


@dataclass(frozen=True)
class NumericHistoryHead:
    """Frozen median imputer, missing indicators, scaler and linear head.

    Unlike the compact transform, this survey-history transform is not clipped.
    Indicator order must match the fitted imputer's indicator feature indices.
    """
    features: tuple[str, ...]
    seen: np.ndarray
    median: np.ndarray
    indicator_indices: tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def __post_init__(self):
        names = tuple(self.features)
        indices = tuple(self.indicator_indices)
        if not names or len(set(names)) != len(names) or not set(names) <= set(H_COLUMNS):
            raise ValueError('Unique survey-history feature subset required')
        if (len(set(indices)) != len(indices) or any(isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer)) or not 0 <= i < len(names) for i in indices)):
            raise ValueError('Unique fitted missing-indicator indices required')
        object.__setattr__(self, 'features', names)
        object.__setattr__(self, 'indicator_indices', indices)
        for name in ('seen', 'median', 'mean', 'scale', 'coef'):
            array = np.asarray(getattr(self, name))
            expected = len(names) if name in ('seen', 'median') else len(names)+len(indices)
            if array.shape != (expected,):
                raise ValueError('History parameter dimensions disagree')
            if name == 'seen':
                if array.dtype.kind != 'b': raise ValueError('Boolean observation mask required')
                array = array.astype(bool, copy=True)
            else:
                if array.dtype.kind == 'c': raise ValueError('Real numeric parameters required')
                array = array.astype(float, copy=True)
                if not np.isfinite(array).all() or (name == 'scale' and (array <= 0).any()):
                    raise ValueError('Finite parameters and positive scales required')
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not np.isscalar(self.intercept) or not np.isfinite(self.intercept):
            raise ValueError('Finite intercept required')
        object.__setattr__(self, 'intercept', float(self.intercept))

    def linear(self, frame):
        if not frame.columns.is_unique or not set(self.features) <= set(frame):
            raise ValueError('Required unique history features missing')
        array = frame.loc[:, list(self.features)].to_numpy(float, copy=True)
        if np.isinf(array).any(): raise ValueError('Infinite history feature')
        array[:, ~self.seen] = np.nan
        missing = ~np.isfinite(array)
        imputed = np.where(missing, self.median, array)
        if self.indicator_indices:
            imputed = np.column_stack([imputed, missing[:, self.indicator_indices].astype(float)])
        return ((imputed-self.mean)/self.scale) @ self.coef+self.intercept


def features_from_history(queries, surveys, ema):
    """Reuse canonical PHQ/EMA and survey-history builders on prior-day data.

    GAD-7 and ISI fields may be absent; they remain missing for frozen training
    imputation. Their absence never changes the current outcome contract. EMA
    response-age features are intentionally unused by this policy and left NaN.
    """
    meta, compact, provenance = core.features_from_history(queries, surveys, ema)
    history = surveys.copy()
    history['index_date'] = core._local_dates(history.index_date)
    for scale in survey_features.SCALES:
        if scale not in history: history[scale] = np.nan
    rows, scale_dates = [], []
    for row in meta.itertuples():
        anchor = row.index_date.normalize()
        past = history.loc[history.subject_id.eq(row.subject_id) & history.index_date.lt(anchor),
                           ['subject_id', 'index_date', *survey_features.SCALES]].copy()
        past['index_date'] = past.index_date.dt.normalize()
        query = pd.DataFrame({'subject_id': [row.subject_id], 'index_date': [anchor],
                              **{name: [np.nan] for name in survey_features.SCALES}})
        inputs = pd.concat([past, query], ignore_index=True)
        # One person and a strictly later empty query: the final row is the query.
        values = survey_features.history_features(inputs).iloc[-1]
        rows.append(values.to_dict())
        scale_dates.append({name: past.loc[past[name].notna(),'index_date'].max()
                            for name in survey_features.SCALES})
    frame = pd.DataFrame(rows, index=meta.index, columns=FEATURE_COLUMNS, dtype=float)
    for name in compact: frame[name] = compact[name]
    dates = pd.DataFrame(scale_dates,index=meta.index,columns=tuple(survey_features.SCALES))
    for name in dates: dates[name] = pd.to_datetime(dates[name])
    for name in survey_features.SCALES:
        provenance[f'latest_{name}_input_date'] = dates[name]
    fallback = compact.h_previous_phq9.notna() & compact.iloc[:,1:].isna().all(axis=1)
    provenance.loc[fallback,'latest_survey_input_date'] = dates.loc[fallback].max(axis=1)
    return meta, frame, provenance


@dataclass(frozen=True)
class FrozenHistoryPolicy:
    """Compact when any recent EMA exists; survey-history model otherwise.

    The primary model's calibration applies once to the whole routed policy.
    Its calibration must therefore be fitted to routed calibration predictions,
    not copied from a differently routed model without development validation.
    """
    primary: core.FrozenLightweight
    fallback_classifier: NumericHistoryHead
    fallback_change_regressor: NumericHistoryHead
    fallback_name: str = 'survey_history'

    def __post_init__(self):
        if not isinstance(self.primary, core.FrozenLightweight): raise ValueError('Frozen primary required')
        if not isinstance(self.fallback_classifier, NumericHistoryHead) or not isinstance(self.fallback_change_regressor, NumericHistoryHead):
            raise ValueError('Numeric survey-history fallback heads required')
        if not isinstance(self.fallback_name, str) or not self.fallback_name.strip():
            raise ValueError('Fallback name required')

    def predict(self, queries, features, policy='auto'):
        core._target_free(features)
        if tuple(features.columns) != FEATURE_COLUMNS:
            raise ValueError('Full ordered survey and EMA feature schema required for history policy')
        if np.isinf(features.to_numpy(float)).any(): raise ValueError('Infinite prepared feature')
        compact = features.loc[:, list(core.FEATURES)]
        out = self.primary.predict(queries, compact, policy)
        prior = compact.h_previous_phq9.to_numpy(float)
        valid = np.isfinite(prior)
        zero_ema = ~np.isfinite(compact.iloc[:, 1:].to_numpy(float)).any(axis=1)
        raw = out.raw_prediction.to_numpy(copy=True)
        total = out.predicted_phq9.to_numpy(copy=True)
        indices = np.flatnonzero(zero_ema & valid)
        if len(indices):
            raw[indices] = expit(self.fallback_classifier.linear(features.iloc[indices]))
            total[indices] = np.clip(prior[indices]+self.fallback_change_regressor.linear(features.iloc[indices]), 0, 27)
        calibration = self.primary
        prediction = raw.copy() if calibration.calibration_slope == 1 and calibration.calibration_intercept == 0 else expit(
            calibration.calibration_slope*logit(np.clip(raw, 1e-7, 1-1e-7))+calibration.calibration_intercept)
        out['raw_prediction'] = raw
        out['prediction'] = prediction
        out['predicted_phq9'] = total
        out['predicted_phq9_change'] = total-prior
        used_fallback = zero_ema & valid
        out['inference_route'] = np.where(~valid, 'none', np.where(used_fallback, self.fallback_name, 'compact'))
        out['history_fallback_used'] = used_fallback
        out.loc[used_fallback, 'used_modalities'] = 'past_surveys'
        out.loc[used_fallback, 'status'] = 'ok_history_fallback'
        out.loc[used_fallback, 'fallback_reason'] = 'no_recent_ema_use_past_survey_model'
        out.loc[used_fallback, 'ema_retained_features'] = 0
        out.loc[used_fallback, 'ema_used_features'] = 0
        out.loc[used_fallback, 'ema_used'] = False
        for domain in ('gad7', 'isi'):
            out[f'past_{domain}_available'] = features[f'h_previous_{domain}'].notna().to_numpy()
        out['fallback_auxiliary_imputation'] = used_fallback & ~(out.past_gad7_available & out.past_isi_available)
        for scale in survey_features.SCALES:
            out[f'latest_{scale}_input_date'] = pd.NaT
        return out

    def predict_from_history(self, queries, surveys, ema, policy='auto'):
        meta, features, provenance = features_from_history(queries, surveys, ema)
        out = self.predict(meta, features, policy)
        for column in provenance: out[column] = provenance[column]
        return out


def save_release(model, directory):
    from .pipeline import private_output
    if not isinstance(model, FrozenHistoryPolicy): raise TypeError('FrozenHistoryPolicy required')
    folder = private_output(directory)
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    folder.chmod(0o700)
    core.save_release(model.primary, folder/'primary')
    arrays, heads = {}, {}
    for name, head in (('classifier', model.fallback_classifier), ('change_regressor', model.fallback_change_regressor)):
        heads[name] = {'features': list(head.features), 'indicator_indices': list(head.indicator_indices), 'intercept': head.intercept}
        for field in ('seen', 'median', 'mean', 'scale', 'coef'):
            arrays[name+'_'+field] = getattr(head, field)
    fd = os.open(folder/'history_weights.npz', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream: np.savez_compressed(stream, **arrays)
    core._write_json(folder/'policy.json', {'format_version': 1, 'feature_schema': list(FEATURE_COLUMNS),
        'route': 'all_recent_ema_missing', 'fallback_name': model.fallback_name, 'heads': heads})
    names = ['history_weights.npz', 'policy.json', 'primary/model.json', 'primary/weights.npz', 'primary/manifest.json']
    core._write_json(folder/'manifest.json', {'source_sha256': core._digest(__file__),
        'core_source_sha256': core._digest(core.__file__), 'feature_source_sha256': core._digest(survey_features.__file__),
        'files': {name: core._digest(folder/name) for name in names}})


def load_release(directory):
    from .pipeline import private_output
    folder = private_output(directory)
    expected = {'history_weights.npz', 'policy.json', 'primary/model.json', 'primary/weights.npz', 'primary/manifest.json'}
    if (folder/'manifest.json').is_symlink(): raise ValueError('Policy manifest symlink forbidden')
    manifest = json.loads((folder/'manifest.json').read_text())
    if (set(manifest.get('files', {})) != expected or manifest.get('source_sha256') != core._digest(__file__)
            or manifest.get('core_source_sha256') != core._digest(core.__file__)
            or manifest.get('feature_source_sha256') != core._digest(survey_features.__file__)):
        raise ValueError('Frozen policy source or member schema changed')
    for name, digest in manifest['files'].items():
        path = folder/name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != folder and folder in parent.parents):
            raise ValueError('Policy member symlinks forbidden')
        if core._digest(path) != digest: raise ValueError('Policy member checksum changed')
    spec = json.loads((folder/'policy.json').read_text())
    if (spec.get('format_version') != 1 or spec.get('feature_schema') != list(FEATURE_COLUMNS)
            or spec.get('route') != 'all_recent_ema_missing'):
        raise ValueError('Unexpected frozen policy contract')
    expected_arrays = {prefix+'_'+field for prefix in ('classifier', 'change_regressor') for field in ('seen','median','mean','scale','coef')}
    with np.load(folder/'history_weights.npz', allow_pickle=False) as stored:
        if set(stored.files) != expected_arrays: raise ValueError('Unexpected history numeric member')
        heads = {prefix: NumericHistoryHead(**spec['heads'][prefix], **{
            field: stored[prefix+'_'+field] for field in ('seen','median','mean','scale','coef')})
            for prefix in ('classifier','change_regressor')}
    return FrozenHistoryPolicy(core.load_release(folder/'primary'), heads['classifier'], heads['change_regressor'], spec['fallback_name'])
