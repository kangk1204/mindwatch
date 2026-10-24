"""Causal long-history EMA summaries and development-only selection helpers.

No files, population data, fitted parameters, or study-specific decisions are
included. Callers isolate training, selection, calibration and test partitions.
The one-standard-error gates are a heuristic, not an uncertainty guarantee.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mindwatch.features import DOMAINS
from . import state_heads
from .state_model import _days, _identity, _scoped_events


HISTORY = tuple(f'ss_history_{d}_{s}' for d in DOMAINS for s in ('mean', 'ewm14'))


def history_features(meta, events):
    """Summarize actual prior-day EMA; age attenuates toward the personal mean.

    For each domain, retain its all-history mean and a 14-day exponentially
    weighted mean. The latter shrinks toward the all-history mean as the latest
    response ages. No interpolation, outcome access, or population fitting is
    performed. Date-only information excludes an anchor's entire calendar day.
    """
    _identity(meta)
    if 'index_date' not in meta:
        raise ValueError('Prediction metadata requires index_date')
    anchors = _days(meta.index_date)
    rows = []
    for subject, anchor in zip(meta.subject_id, anchors):
        # Scope before checking numeric values or duplicates: editing a future
        # response must not change a previous prediction's inputs.
        group = _scoped_events(events, {subject}, anchor, DOMAINS)
        features = {}
        for domain in DOMAINS:
            observed = group.loc[group[domain].notna(), ['date', domain]]
            values = observed[domain].to_numpy(float)
            if not len(values):
                mean = decayed = np.nan
            else:
                age = (anchor-observed.date).dt.total_seconds().to_numpy()/86400
                weight = np.exp2(-(age-age.min())/14.)
                mean = float(values.mean())
                decayed = mean + np.exp2(-age.min()/14.)*(np.average(values, weights=weight)-mean)
            features[f'ss_history_{domain}_mean'] = mean
            features[f'ss_history_{domain}_ewm14'] = decayed
        rows.append(features)
    return pd.DataFrame(rows, index=meta.index, columns=HISTORY, dtype=float)


def available(head, x):
    """Whether any retained history input is observed, row by row."""
    columns = [c for c in head.columns if c in HISTORY]
    if x.columns.duplicated().any() or not set(columns) <= set(x.columns):
        raise ValueError('Unique columns and fitted history inputs required')
    return (x[columns].notna().any(axis=1).to_numpy() if columns
            else np.zeros(len(x), dtype=bool))


def _feature_matrix(x, columns):
    if (x.columns.duplicated().any() or len(set(columns)) != len(columns)
            or any(not isinstance(c, str) for c in columns)
            or not set(columns) <= set(x.columns)):
        raise ValueError('Unique named feature columns required')
    a = x.loc[:, list(columns)].to_numpy(float, copy=True)
    if np.isinf(a).any():
        raise ValueError('Infinite feature')
    return a


def _previous(x):
    if 'h_previous_phq9' not in x:
        raise ValueError('Previous PHQ required')
    previous = x.h_previous_phq9.to_numpy(float)
    if not np.isfinite(previous).all() or ((previous < 0) | (previous > 27)).any():
        raise ValueError('Valid previous PHQ required')
    return previous


@dataclass
class HAbsolute:
    """Absolute-loss change head with train-only imputation and scaling."""
    columns: tuple
    seen: np.ndarray
    preprocessor: object
    theta: np.ndarray
    diagnostic: dict

    def predict(self, x):
        a = _feature_matrix(x, self.columns)
        previous = _previous(x)
        if not len(x):
            return np.empty(0, dtype=float)
        a[:, ~self.seen] = np.nan
        z = self.preprocessor.transform(a)
        return np.clip(previous+self.theta[0]+z@self.theta[1:], 0, 27)


def fit_h_absolute(x, total, ids, columns):
    """Fit the absolute-loss counterpart of a survey-history change model.

    Median imputation (with missingness indicators) and unweighted scaling match
    the survey-history preprocessing family; participant-equal weights apply to
    the prediction loss. All preprocessing is fitted on the supplied rows only.
    """
    columns = tuple(columns)
    a = _feature_matrix(x, columns)
    previous = _previous(x)
    total = np.asarray(total, dtype=float)
    if (not len(x) or not len(columns) or len(ids) != len(x)
            or total.shape != (len(x),) or not np.isfinite(total).all()
            or ((total < 0) | (total > 27)).any()):
        raise ValueError('Aligned nonempty training inputs and PHQ totals required')
    seen = np.isfinite(a).any(axis=0)
    a[:, ~seen] = np.nan
    pre = make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                     keep_empty_features=True), StandardScaler())
    z = pre.fit_transform(a)
    target = total-previous
    weight = state_heads.weights(ids)
    design = np.column_stack([np.ones(len(z)), z])
    result = minimize(state_heads.absolute_objective,
                      np.r_[np.median(target), np.zeros(z.shape[1])],
                      args=(design, target, weight/weight.sum()),
                      method='L-BFGS-B', jac=True,
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('H absolute optimization failed')
    return HAbsolute(columns, seen, pre, result.x,
                     {'converged': True, 'iterations': int(result.nit)})


def _evidence(rows, metrics, name_key):
    rows = list(rows)
    if (not rows or any(not isinstance(r.get(name_key), str) or not r[name_key] for r in rows)
            or len({r[name_key] for r in rows}) != len(rows)):
        raise ValueError('Nonempty, uniquely named candidate evidence required')
    for row in rows:
        for metric in metrics:
            mean, se = row['mean_'+metric], row['se_'+metric]
            if (not np.isfinite([mean, se]).all() or se < 0 or mean < 0
                    or (metric in ('auc', 'brier') and mean > 1)):
                raise ValueError('Finite scores and nonnegative uncertainty required')
    return sorted(rows, key=lambda row: row[name_key])


def decision(trials):
    """AUROC, Brier and MAE bands, then fewer inputs; development evidence only."""
    trials = _evidence(trials, ('auc', 'brier', 'mae', 'rmse'), 'id')
    for row in trials:
        if (isinstance(row.get('n_features'), bool)
                or not isinstance(row.get('n_features'), (int, np.integer))
                or row['n_features'] < 0 or not isinstance(row.get('regression_head'), str)
                or not row['regression_head']):
            raise ValueError('Nonnegative feature count and regression head required')
    best = min(trials, key=lambda row: (-row['mean_auc'], row['id']))
    auc_cut = best['mean_auc']-best['se_auc']
    auc_set = [row for row in trials if row['mean_auc'] >= auc_cut-1e-12]
    best = min(auc_set, key=lambda row: (row['mean_brier'], row['id']))
    brier_cut = best['mean_brier']+best['se_brier']
    brier_set = [row for row in auc_set if row['mean_brier'] <= brier_cut+1e-12]
    best = min(brier_set, key=lambda row: (row['mean_mae'], row['id']))
    mae_cut = best['mean_mae']+best['se_mae']
    eligible = [row for row in brier_set if row['mean_mae'] <= mae_cut+1e-12]
    selected = min(eligible, key=lambda row: (row['n_features'], row['id']))
    return {'selected': selected['id'], 'selected_head': selected['regression_head'],
            'auc_cut': auc_cut, 'brier_cut': brier_cut, 'mae_cut': mae_cut,
            'auc_eligible': [row['id'] for row in auc_set],
            'brier_eligible': [row['id'] for row in brier_set],
            'mae_eligible': [row['id'] for row in eligible]}


def select_head(rows):
    """MAE band, then RMSE band, then Ridge for equivalent regression heads."""
    rows = _evidence(rows, ('mae', 'rmse'), 'head')
    best = min(rows, key=lambda row: (row['mean_mae'], row['head']))
    cut = best['mean_mae']+best['se_mae']
    eligible = [row for row in rows if row['mean_mae'] <= cut+1e-12]
    best = min(eligible, key=lambda row: (row['mean_rmse'], row['head']))
    cut = best['mean_rmse']+best['se_rmse']
    eligible = [row for row in eligible if row['mean_rmse'] <= cut+1e-12]
    return min(eligible, key=lambda row: (row['head'] != 'ridge', row['head']))
