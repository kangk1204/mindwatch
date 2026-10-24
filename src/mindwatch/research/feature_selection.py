"""Refitted feature ablations and participant-disjoint temporal selection.

Only a supplied development-training partition is accepted. This module has no
study paths, fitted models, cohort manifests, stored predictions, or file IO.
Callers must freeze their protocol and retain test people outside this API.
"""
from dataclasses import dataclass
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from mindwatch.features import E_COLUMNS, H_COLUMNS
from mindwatch.models import calibrate, fit_head, person_weights, validate_x
from .contracts import calendar_table


@dataclass(frozen=True)
class FeatureRecipe:
    name: str
    family: str
    columns: tuple[str, ...]

    def __post_init__(self):
        allowed = H_COLUMNS if self.family == 'H' else ('h_previous_phq9', *E_COLUMNS)
        if (self.family not in ('H', 'compact') or not isinstance(self.name, str)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', self.name)
                or not isinstance(self.columns, tuple)
                or any(not isinstance(c, str) for c in self.columns)
                or len(set(self.columns)) != len(self.columns)
                or not set(self.columns) <= set(allowed)):
            raise ValueError('Valid named recipe and unique family columns required')

    def fit(self, x, y, people):
        return fit_head(x, y, people, self.family, columns=self.columns)


def collection_pattern_recipes():
    """Four fixed recipes; this function does not designate an active model."""
    compact = ('h_previous_phq9', *E_COLUMNS)
    return (
        FeatureRecipe('H_full', 'H', tuple(H_COLUMNS)),
        FeatureRecipe('H_no_collection', 'H', tuple(c for c in H_COLUMNS
                      if 'days' not in c and 'count' not in c)),
        FeatureRecipe('compact_full', 'compact', compact),
        FeatureRecipe('compact_no_days', 'compact', tuple(c for c in compact
                      if not c.endswith('_days'))),
    )


def ablation_recipes(recipe, groups=None):
    """Generate complete refit recipes, never just zero fitted coefficients.

    None means one deletion per input; explicit groups can include correlated
    derivatives that must be removed together. Overlapping groups are allowed.
    """
    groups = {c: (c,) for c in recipe.columns} if groups is None else groups
    result = [recipe]
    for name, removed in groups.items():
        if (not isinstance(removed, (list, tuple)) or not removed
                or any(not isinstance(c, str) for c in removed)
                or len(set(removed)) != len(removed)
                or not set(removed) <= set(recipe.columns)):
            raise ValueError('A nonempty group of unique recipe inputs is required')
        result.append(FeatureRecipe(recipe.name+'_drop_'+name, recipe.family,
                                    tuple(c for c in recipe.columns if c not in removed)))
    return tuple(result)


@dataclass(frozen=True)
class TemporalSelectionConfig:
    window_days: int = 28
    gap_days: int = 14
    min_train_people: int = 50
    min_class_people: int = 10
    calibration_modulus: int = 5
    calibration_salt: str = 'mindwatch-temporal-calibration'

    def __post_init__(self):
        for value in (self.window_days, self.gap_days, self.min_train_people,
                      self.min_class_people, self.calibration_modulus):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Positive integer temporal selection settings required')
        if self.calibration_modulus < 2 or not isinstance(self.calibration_salt, str):
            raise ValueError('A calibration subset and string salt are required')


def choose_one_se(trials):
    """AUROC band, then Brier band, then fewer inputs; two-fold SE is heuristic."""
    if not trials or len({r['name'] for r in trials}) != len(trials):
        raise ValueError('Nonempty uniquely named trials required')
    for r in trials:
        values = [r[k] for k in ('mean_auc', 'se_auc', 'mean_brier', 'se_brier')]
        if (not np.isfinite(values).all() or not 0 <= r['mean_auc'] <= 1
                or not 0 <= r['mean_brier'] <= 1 or r['se_auc'] < 0
                or r['se_brier'] < 0 or isinstance(r['n_features'], bool)
                or not isinstance(r['n_features'], (int, np.integer))
                or r['n_features'] < 0):
            raise ValueError('Finite scores, nonnegative uncertainty and feature count required')
    trials = sorted(trials, key=lambda r: r['name'])
    best = min(trials, key=lambda r: (-r['mean_auc'], r['name']))
    auc_cut = best['mean_auc']-best['se_auc']
    auc_set = [r for r in trials if r['mean_auc'] >= auc_cut-1e-12]
    calibrated = min(auc_set, key=lambda r: (r['mean_brier'], r['name']))
    brier_cut = calibrated['mean_brier']+calibrated['se_brier']
    eligible = [r for r in auc_set if r['mean_brier'] <= brier_cut+1e-12]
    selected = min(eligible, key=lambda r: (r['n_features'], r['name']))
    return {'selected': selected['name'], 'auc_cut': auc_cut, 'brier_cut': brier_cut,
            'auc_eligible': [r['name'] for r in auc_set],
            'brier_eligible': [r['name'] for r in eligible]}


def _class_people(frame, y, rows):
    return [frame.iloc[rows].loc[y[rows] == k, 'subject_id'].nunique() for k in (0, 1)]


def _training_metadata(metadata, y, boundary, excluded_people):
    frame = calendar_table(metadata, 'index_date', date_only=True)
    boundary = pd.Timestamp(boundary)
    if pd.isna(boundary) or boundary.tz is not None or boundary != boundary.normalize():
        raise ValueError('A date-only local training boundary is required')
    if 'label_available_at' not in frame:
        raise ValueError('Explicit label availability is required')
    available = pd.to_datetime(frame.label_available_at, errors='raise')
    if (available.isna().any() or available.dt.tz is not None
            or not available.eq(available.dt.normalize()).all()
            or (available < frame.index_date+pd.Timedelta(days=1)).any()
            or (available >= boundary).any()):
        raise ValueError('Only date-only labels available before the training boundary are allowed')
    frame['label_available_at'] = available
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != len(frame) or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned nonempty binary training outcomes required')
    if set(frame.subject_id) & {str(p) for p in excluded_people}:
        raise ValueError('Excluded participant present in training input')
    return frame, y, boundary


def temporal_folds(metadata, y, boundary, *, config=None, excluded_people=()):
    config = TemporalSelectionConfig() if config is None else config
    frame, y, boundary = _training_metadata(metadata, y, boundary, excluded_people)
    folds = []
    for j in range(2):
        start = boundary-pd.Timedelta(days=config.window_days*(2-j))
        end = start+pd.Timedelta(days=config.window_days)
        validation = np.flatnonzero(frame.index_date.ge(start) & frame.index_date.lt(end))
        validation_people = set(frame.iloc[validation].subject_id)
        eligible = np.flatnonzero(frame.label_available_at.lt(start-pd.Timedelta(days=config.gap_days))
                                  & ~frame.subject_id.isin(validation_people))
        calibration_people = {p for p in frame.iloc[eligible].subject_id
                              if int(hashlib.sha256((config.calibration_salt+'|'+p).encode()).hexdigest()[:8], 16)
                              % config.calibration_modulus == 0}
        calibration = np.array([i for i in eligible if frame.subject_id.iloc[i] in calibration_people], int)
        train = np.array([i for i in eligible if frame.subject_id.iloc[i] not in calibration_people], int)
        for a, b in ((train, calibration), (train, validation), (calibration, validation)):
            if set(frame.iloc[a].subject_id) & set(frame.iloc[b].subject_id):
                raise ValueError('Participant overlap between inner roles')
        train_people = frame.iloc[train].subject_id.nunique()
        train_classes = _class_people(frame, y, train)
        validation_classes = _class_people(frame, y, validation)
        valid = (train_people >= config.min_train_people
                 and min(train_classes) >= config.min_class_people
                 and min(validation_classes) >= config.min_class_people)
        folds.append({'fold': j, 'start': str(start), 'end': str(end),
                      'train': train.tolist(), 'calibration': calibration.tolist(),
                      'validation': validation.tolist(), 'valid': bool(valid),
                      'train_people': int(train_people),
                      'train_class_people': train_classes,
                      'validation_class_people': validation_classes})
    return folds


def select_on_training(x, y, metadata, boundary, recipes, *, fallback,
                       config=None, excluded_people=()):
    """Choose on inner folds of an already isolated outer-training partition.

    Return local fold rows and selection diagnostics, not outer performance.
    Callers must keep these diagnostics private and evaluate the saved choice on
    their later untouched partition without changing it from that result.
    """
    validate_x(x)
    if not x.index.equals(metadata.index):
        raise ValueError('Feature and metadata row indices must align')
    config = TemporalSelectionConfig() if config is None else config
    frame, y, boundary = _training_metadata(metadata, y, boundary, excluded_people)
    if len(x) != len(y):
        raise ValueError('Aligned training features and outcomes required')
    recipes = tuple(recipes)
    if (not recipes or any(not isinstance(r, FeatureRecipe) for r in recipes)
            or len({r.name for r in recipes}) != len(recipes)
            or fallback not in {r.name for r in recipes}):
        raise ValueError('Unique candidate recipes and an explicit fallback required')
    folds = temporal_folds(frame, y, boundary, config=config)
    valid = [f for f in folds if f['valid']]
    if len(valid) < 2:
        return {'selected': fallback, 'status': 'warmup_fallback',
                'valid_folds': len(valid), 'folds': folds, 'trials': [], 'fit_count': 0}
    trials = []
    for recipe in recipes:
        scores = []
        for fold in valid:
            train, calibration, validation = [fold[k] for k in ('train', 'calibration', 'validation')]
            model = recipe.fit(x.iloc[train], y[train], frame.subject_id.iloc[train])
            cal = calibrate(y[calibration], model.predict(x.iloc[calibration]),
                            frame.subject_id.iloc[calibration], config.min_class_people)
            p = cal.predict(model.predict(x.iloc[validation]))
            weight = person_weights(frame.subject_id.iloc[validation])
            scores.append({'auc': float(roc_auc_score(y[validation], p, sample_weight=weight)),
                           'brier': float(brier_score_loss(y[validation], p, sample_weight=weight)),
                           'calibration_status': cal.status})
        auc = np.array([r['auc'] for r in scores]); brier = np.array([r['brier'] for r in scores])
        trials.append({'name': recipe.name, 'n_features': len(recipe.columns),
                       'mean_auc': float(auc.mean()), 'se_auc': float(auc.std(ddof=1)/np.sqrt(len(auc))),
                       'mean_brier': float(brier.mean()), 'se_brier': float(brier.std(ddof=1)/np.sqrt(len(brier))),
                       'scores': scores})
    return {**choose_one_se(trials), 'status': 'inner_selected', 'valid_folds': len(valid),
            'folds': folds, 'trials': trials, 'fit_count': len(valid)*len(recipes)}
