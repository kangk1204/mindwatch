"""Small train-only supervised heads for causal state representations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge


def weights(ids):
    """Equal total contribution per participant, normalized to mean one."""
    ids = np.asarray(ids, dtype=object)
    if ids.ndim != 1 or not len(ids) or pd.isna(ids).any():
        raise ValueError('Nonempty, nonmissing participant keys required')
    inv, _ = pd.factorize(ids, sort=False)
    counts = np.bincount(inv)
    w = 1. / counts[inv]
    return w / w.mean()


def absolute_objective(theta, design, target, w, alpha=.01, epsilon=.05):
    error = design @ theta - target
    norm = np.sqrt(error * error + epsilon * epsilon)
    penalty = np.r_[0., theta[1:]]
    value = np.sum(w * norm) + .5 * alpha * np.dot(theta[1:], theta[1:])
    gradient = design.T @ (w * error / norm) + alpha * penalty
    return value, gradient


@dataclass
class SmallHead:
    task: str
    columns: tuple
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    estimator: object
    constant: float
    diagnostic: dict

    def matrix(self, x):
        if x.columns.duplicated().any() or not set(self.columns) <= set(x.columns):
            raise ValueError('Unique input columns and fitted features required')
        a = x.loc[:, list(self.columns)].to_numpy(float, copy=True)
        if np.isinf(a).any():
            raise ValueError('Infinite feature')
        a = np.where(np.isfinite(a), a, self.median)
        return np.clip((a - self.mean) / self.scale, -12, 12)

    def predict(self, x, previous=None):
        if not len(x):return np.empty(0,dtype=float)
        a = self.matrix(x)
        if self.estimator is None:
            raw = np.full(len(x), self.constant)
        elif self.task == 'class':
            raw = self.estimator.predict_proba(a)[:, 1]
        elif self.task == 'absolute':
            raw = self.estimator[0] + a @ self.estimator[1:]
        else:
            raw = self.estimator.predict(a)
        if self.task == 'class':
            return raw
        previous = np.asarray(previous, float)
        if (previous.shape != raw.shape or not np.isfinite(previous).all()
                or ((previous < 0) | (previous > 27)).any()):
            raise ValueError('Previous PHQ required for change head')
        return np.clip(previous + raw, 0, 27)


def fit(x, y, ids, columns, task, previous=None):
    """Fit only on a caller-isolated training partition.

    Classification uses binary labels. Regression fits current minus previous
    PHQ, then reconstructs a bounded total at inference. This helper does not
    split data or select hyperparameters; callers must exclude validation and
    test people before invoking it.
    """
    columns = tuple(columns)
    if (x.columns.duplicated().any() or len(set(columns)) != len(columns)
            or not set(columns) <= set(x.columns)
            or any(not isinstance(c, str) for c in columns)):
        raise ValueError('Unique named feature columns required')
    y = np.asarray(y, float)
    if task not in ('class', 'ridge', 'absolute') or y.shape != (len(x),) or not len(x) or len(ids)!=len(x) or not np.isfinite(y).all():
        raise ValueError('Invalid head contract')
    if task=='class' and not np.isin(y,[0,1]).all():raise ValueError('Binary labels required')
    if np.isinf(x.loc[:,list(columns)].to_numpy(float)).any():raise ValueError('Infinite feature')
    if task!='class':
        previous=np.asarray(previous,float)
        if previous.shape!=y.shape or not np.isfinite(previous).all() or ((previous<0)|(previous>27)).any():
            raise ValueError('Invalid past PHQ')
        if ((y<0)|(y>27)).any():raise ValueError('Invalid PHQ target')
    cols = tuple(c for c in columns if x[c].notna().mean() >= .3 and x[c].nunique() > 1)
    a = x.loc[:, list(cols)].to_numpy(float, copy=True)
    w = weights(ids)
    median = np.nanmedian(a, axis=0) if len(cols) else np.empty(0)
    a = np.where(np.isfinite(a), a, median)
    mean = np.average(a, axis=0, weights=w)
    scale = np.sqrt(np.average((a - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-6] = 1
    a = np.clip((a - mean) / scale, -12, 12)
    # A pooled OU level and deviation differ only by a fitted constant.
    # Remove affine duplicates of added state columns without removing duplicate baseline inputs.
    keep = []
    dropped = []
    for j, name in enumerate(cols):
        duplicate = name.startswith('ss_') and any(
            np.allclose(a[:, j], a[:, k], atol=1e-10, rtol=0) or
            np.allclose(a[:, j], -a[:, k], atol=1e-10, rtol=0) for k in keep)
        if duplicate:
            dropped.append(name)
        else:
            keep.append(j)
    cols = tuple(cols[j] for j in keep)
    a, median, mean, scale = a[:, keep], median[keep], mean[keep], scale[keep]
    target = y if task == 'class' else y - np.asarray(previous, float)
    constant = float(np.average(target, weights=w))
    if task=='absolute' and not len(cols):
        opt=minimize(absolute_objective,np.array([np.median(target)]),
                     args=(np.ones((len(target),1)),target,w/w.sum()),method='L-BFGS-B',jac=True)
        if not opt.success:raise RuntimeError('Constant absolute head failed')
        constant=float(opt.x[0])
    estimator = None
    diagnostic = {'converged': True, 'n_features': len(cols), 'affine_duplicates_removed': dropped}
    if len(cols):
        if task == 'class' and len(np.unique(y)) > 1:
            estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=1500).fit(a, y, sample_weight=w)
            diagnostic['converged'] = bool(estimator.n_iter_[0] < 1500)
        elif task == 'ridge':
            estimator = Ridge(alpha=10., solver='svd').fit(a, target, sample_weight=w)
        elif task == 'absolute':
            design = np.column_stack([np.ones(len(a)), a])
            start = np.r_[np.median(target), np.zeros(a.shape[1])]
            opt = minimize(absolute_objective, start, args=(design, target, w / w.sum()),
                           method='L-BFGS-B', jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-7})
            if not opt.success or not np.isfinite(opt.x).all():
                raise RuntimeError('Absolute-loss optimization failed: ' + str(opt.message))
            estimator = opt.x
            diagnostic.update(iterations=int(opt.nit), gradient_max=float(np.max(abs(opt.jac))))
    return SmallHead(task, cols, median, mean, scale, estimator, constant, diagnostic)
