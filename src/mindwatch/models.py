"""Small PHQ classifiers with fit-only preprocessing and equal person weights."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import E_COLUMNS, FEATURE_COLUMNS, H_COLUMNS


def person_weights(ids) -> np.ndarray:
    keys = pd.Series(np.asarray(ids))
    if keys.isna().any() or not len(keys):
        raise ValueError("Nonempty nonmissing participant keys required")
    weights = 1.0/keys.map(keys.value_counts()).to_numpy(float)
    return weights/weights.mean()


def _binary_vectors(y, p, ids, *, allow_empty=False):
    """Validate labels before conversion; fractional scores are not classes."""
    y, p, ids = np.asarray(y, float), np.asarray(p, float), np.asarray(ids)
    if (y.ndim != 1 or p.ndim != 1 or ids.ndim != 1 or
            not (len(y) == len(p) == len(ids)) or (not allow_empty and not len(y)) or
            not np.isin(y, [0., 1.]).all() or not np.isfinite(p).all() or
            not ((p >= 0) & (p <= 1)).all() or pd.isna(ids).any()):
        raise ValueError("Aligned binary outcomes, probabilities and participant keys required")
    return y.astype(int), p, ids


def validate_x(x: pd.DataFrame):
    if tuple(x.columns) != FEATURE_COLUMNS:
        raise ValueError("Exact feature schema required; outcomes/identifiers are forbidden")
    a = x.to_numpy(float, copy=True)
    if np.isinf(a).any():
        raise ValueError("Infinite features are not missing observations")


@dataclass
class Head:
    recipe: str
    columns: tuple
    seen: np.ndarray
    estimator: object
    constant: float
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        validate_x(x)
        if not len(x):
            return np.empty(0)
        if self.estimator is None:
            return np.full(len(x), self.constant)
        a = x.loc[:, list(self.columns)].to_numpy(float, copy=True)
        a[:, ~self.seen] = np.nan
        if self.recipe == "compact":
            a = np.where(np.isfinite(a), a, self.median)
            a = np.clip((a-self.mean)/self.scale, -12, 12)
        return self.estimator.predict_proba(a)[:, 1]


def fit_head(x: pd.DataFrame, y, ids, recipe: str) -> Head:
    validate_x(x)
    y = np.asarray(y, float)
    if len(x) != len(y) or len(y) != len(ids) or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned nonempty binary training outcomes required")
    if recipe not in ("H", "compact"):
        raise ValueError("Unknown model recipe")
    weights = person_weights(ids)
    columns = H_COLUMNS if recipe == "H" else ("h_previous_phq9", *E_COLUMNS)
    if recipe == "compact":
        columns = tuple(c for c in columns if x[c].notna().mean() >= .3 and x[c].nunique() > 1)
    a = x.loc[:, list(columns)].to_numpy(float, copy=True)
    seen = np.isfinite(a).any(axis=0)
    constant = float(np.average(y, weights=weights))
    result = Head(recipe, tuple(columns), seen, None, constant)
    if not seen.any() or np.unique(y).size < 2:
        return result
    if recipe == "H":
        result.estimator = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=.01, solver="lbfgs", max_iter=4000, random_state=42),
        )
        result.estimator.fit(a, y, logisticregression__sample_weight=weights)
    else:
        result.median = np.nanmedian(a, axis=0)
        a = np.where(np.isfinite(a), a, result.median)
        result.mean = np.average(a, axis=0, weights=weights)
        result.scale = np.sqrt(np.average((a-result.mean)**2, axis=0, weights=weights))
        result.scale[result.scale < 1e-6] = 1
        a = np.clip((a-result.mean)/result.scale, -12, 12)
        result.estimator = LogisticRegression(C=1., solver="lbfgs", max_iter=1500)
        result.estimator.fit(a, y, sample_weight=weights)
    return result


@dataclass
class Calibration:
    intercept: float = 0.
    threshold: float = .5
    status: str = "insufficient_calibration_support"

    def predict(self, p):
        if self.intercept == 0.:
            return np.asarray(p, float).copy()
        return expit(logit(np.clip(np.asarray(p, float), 1e-7, 1-1e-7))+self.intercept)


def calibrate(y, p, ids, min_people_per_class: int = 10) -> Calibration:
    """Offset-only calibration; choose a weighted sensitivity>=0.8 threshold."""
    y, p, ids = _binary_vectors(y, p, ids, allow_empty=True)
    if (isinstance(min_people_per_class, (bool, np.bool_)) or
            not isinstance(min_people_per_class, (int, np.integer)) or min_people_per_class < 1):
        raise ValueError("Positive integer participant support required")
    if not len(y) or min(len(np.unique(ids[y == k])) for k in (0, 1)) < min_people_per_class:
        return Calibration()
    w, z = person_weights(ids), logit(np.clip(p, 1e-7, 1-1e-7))
    fit = minimize_scalar(lambda a: np.average(np.logaddexp(0, z+a)-y*(z+a), weights=w),
                          bounds=(-12., 12.), method="bounded")
    if not fit.success:
        raise RuntimeError("Calibration did not converge")
    result = Calibration(float(fit.x), status="intercept_calibrated")
    positive = result.predict(p)[y == 1]
    wp = w[y == 1]
    order = np.argsort(-positive, kind="stable")
    index = np.searchsorted(np.cumsum(wp[order]), .8*wp.sum(), side="left")
    result.threshold = float(positive[order[min(index, len(order)-1)]])
    return result
