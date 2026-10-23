"""Participant-weighted evaluation; no source records are logged.

These routines validate software/statistical calculations, not clinical validity.
Undefined estimates are returned as ``None`` with an explicit status.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def participant_weights(subject_ids: Sequence[Any]) -> np.ndarray:
    """Each participant's weights sum to one, regardless of visit count."""
    subjects = pd.Series(subject_ids, copy=False)
    if subjects.isna().any():
        raise ValueError("Participant identifiers must be present.")
    counts = subjects.groupby(subjects, sort=False).transform("size").to_numpy()
    return np.divide(1.0, counts, out=np.empty(len(counts), dtype=float))


def _arrays(y: Sequence[float], p: Sequence[float], subjects: Sequence[Any]):
    y_arr, p_arr, group_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(subjects)
    if y_arr.ndim != 1 or p_arr.ndim != 1 or group_arr.ndim != 1 or not (len(y_arr) == len(p_arr) == len(group_arr)):
        raise ValueError("Outcomes, predictions and participants must be aligned vectors.")
    if pd.isna(group_arr).any():
        raise ValueError("Participant identifiers must be present.")
    if np.any(np.isfinite(y_arr) & ~np.isin(y_arr, [0, 1])):
        raise ValueError("Binary outcomes must be zero, one or missing.")
    if np.any(np.isinf(p_arr)) or np.any(np.isfinite(p_arr) & ((p_arr < 0) | (p_arr > 1))):
        raise ValueError("Probabilities must lie between zero and one, or be missing.")
    return y_arr, p_arr, group_arr


def calibration_parameters(y, p, weights) -> dict[str, Any]:
    """Joint intercept/slope of logit outcome versus logit prediction.

    Unpenalized maximum likelihood is undefined under complete separation;
    those fits are explicitly unavailable, rather than regularized silently.
    """
    y, p, weights = np.asarray(y), np.asarray(p), np.asarray(weights, dtype=float)
    unavailable = {"calibration_intercept": None, "calibration_slope": None}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {**unavailable, "calibration_status": "single_class_or_empty"}
    x = logit(np.clip(p, 1e-7, 1 - 1e-7))
    if np.ptp(x) < 1e-9:
        return {**unavailable, "calibration_status": "constant_predictions"}
    design = np.column_stack([np.ones(len(x)), x])
    w = weights / weights.sum()

    def objective(beta):
        eta = design @ beta
        loss = np.sum(w * (np.logaddexp(0, eta) - y * eta))
        grad = design.T @ (w * (expit(eta) - y))
        return loss, grad

    fit = minimize(objective, np.array([0.0, 1.0]), jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 300})
    fitted = expit(design @ fit.x)
    hessian = design.T @ ((w * fitted * (1 - fitted))[:, None] * design)
    eigen = np.linalg.eigvalsh(hessian)
    if not fit.success or not np.isfinite(fit.x).all() or eigen.min() < 1e-8 or np.max(np.abs(fit.x)) > 30:
        return {**unavailable, "calibration_status": "nonidentifiable_or_not_converged"}
    return {"calibration_intercept": float(fit.x[0]), "calibration_slope": float(fit.x[1]), "calibration_status": "ok"}


def weighted_calibration_curve(y, p, subject_ids, n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width reliability bins using global equal-participant weights."""
    if n_bins < 1:
        raise ValueError("At least one calibration bin is required.")
    y, p, subjects = _arrays(y, p, subject_ids)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p, subjects = y[keep], p[keep], subjects[keep]
    weights = participant_weights(subjects)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    result = []
    for index in range(n_bins):
        selected = bins == index
        if not selected.any():
            continue
        result.append({"lower": index / n_bins, "upper": (index + 1) / n_bins, "mean_predicted": float(np.average(p[selected], weights=weights[selected])), "observed_frequency": float(np.average(y[selected], weights=weights[selected])), "weight": float(weights[selected].sum()), "n_visits": int(selected.sum()), "n_participants": int(pd.Series(subjects[selected]).nunique())})
    return result


def binary_metrics(y, p, subject_ids, threshold: float | Sequence[float] | None = 0.5) -> dict[str, Any]:
    """Weighted discrimination, calibration and confusion metrics with abstention.

    ``p=NaN`` means abstain. Coverage weights are computed among all labelled
    visits; evaluation weights are recomputed among predicted visits so every
    represented participant has equal total evaluation weight.
    """
    y, p, subjects = _arrays(y, p, subject_ids)
    labelled = np.isfinite(y)
    valid = labelled & np.isfinite(p)
    all_weights = participant_weights(subjects[labelled]) if labelled.any() else np.array([])
    coverage = float(all_weights[np.isfinite(p[labelled])].sum() / all_weights.sum()) if len(all_weights) else None
    result: dict[str, Any] = {
        "n_labelled_visits": int(labelled.sum()), "n_predicted_visits": int(valid.sum()),
        "n_labelled_participants": int(pd.Series(subjects[labelled]).nunique()),
        "n_predicted_participants": int(pd.Series(subjects[valid]).nunique()),
        "coverage": coverage, "auroc": None, "auprc": None, "brier": None,
        "sensitivity": None, "specificity": None, "ppv": None, "npv": None,
        "calibration_intercept": None, "calibration_slope": None,
        "calibration_curve": [],
    }
    if not valid.any():
        return {**result, "status": "no_predictions", "calibration_status": "empty"}
    yy, pp = y[valid].astype(int), p[valid]
    weights = participant_weights(subjects[valid])
    both = len(np.unique(yy)) == 2
    result.update(status="ok" if both else "single_class", weighted_prevalence=float(np.average(yy, weights=weights)), brier=float(brier_score_loss(yy, pp, sample_weight=weights)))
    if both:
        result.update(auroc=float(roc_auc_score(yy, pp, sample_weight=weights)), auprc=float(average_precision_score(yy, pp, sample_weight=weights)))
    result.update(calibration_parameters(yy, pp, weights))
    result["calibration_curve"] = weighted_calibration_curve(yy, pp, subjects[valid])
    if threshold is not None:
        thresholds = np.asarray(threshold, dtype=float)
        if thresholds.ndim == 0:
            thresholds = np.repeat(thresholds, len(y))
        if thresholds.shape != y.shape:
            raise ValueError("Thresholds must be scalar or match the outcome vector.")
        tt = thresholds[valid]
        if not np.isfinite(tt).all() or np.any((tt < 0) | (tt > 1)):
            raise ValueError("Thresholds must be finite probabilities.")
        predicted = pp >= tt
        tp = float(weights[(yy == 1) & predicted].sum())
        fn = float(weights[(yy == 1) & ~predicted].sum())
        fp = float(weights[(yy == 0) & predicted].sum())
        tn = float(weights[(yy == 0) & ~predicted].sum())
        ratio = lambda a, b: float(a / b) if b > 0 else None
        result.update(sensitivity=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp), ppv=ratio(tp, tp + fp), npv=ratio(tn, tn + fn), weighted_tp=tp, weighted_fn=fn, weighted_fp=fp, weighted_tn=tn)
    return result


def select_sensitivity_threshold(y, p, subject_ids, target: float = 0.8) -> dict[str, Any]:
    """Highest calibration-set threshold attaining the prespecified sensitivity.

    Caller must supply calibration data only. Both classes are required so a
    single-class calibration subset cannot silently define an operating point.
    """
    if not 0 < target <= 1:
        raise ValueError("Target sensitivity must be in (0, 1].")
    y, p, subjects = _arrays(y, p, subject_ids)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p, subjects = y[keep], p[keep], subjects[keep]
    if len(np.unique(y)) < 2:
        return {"threshold": None, "status": "single_class_or_empty", "target_sensitivity": target}
    weights = participant_weights(subjects)
    positive = y == 1
    order = np.argsort(-p[positive], kind="stable")
    positive_p, positive_w = p[positive][order], weights[positive][order]
    position = min(int(np.searchsorted(np.cumsum(positive_w) / positive_w.sum(), target - 1e-12)), len(positive_p) - 1)
    threshold = float(positive_p[position])
    achieved = binary_metrics(y, p, subjects, threshold)
    return {"threshold": threshold, "status": "ok", "target_sensitivity": target, "calibration_sensitivity": achieved["sensitivity"], "calibration_specificity": achieved["specificity"]}


def _metric(y, p, weights, name):
    if name in ("auroc", "auprc") and len(np.unique(y)) < 2:
        return None
    if name == "auroc":
        return float(roc_auc_score(y, p, sample_weight=weights))
    if name == "auprc":
        return float(average_precision_score(y, p, sample_weight=weights))
    if name == "brier":
        return float(np.average((y - p) ** 2, weights=weights))
    raise ValueError("Bootstrap metric must be auroc, auprc or brier.")


def paired_cluster_bootstrap(y, p_a, p_b, subject_ids, *, metric: str = "auroc", n_bootstrap: int = 2000, seed: int = 42, confidence: float = 0.95) -> dict[str, Any]:
    """Paired participant-cluster bootstrap; delta is metric(a) - metric(b).

    Only the common labelled/predicted visits are compared. Resampled copies
    retain equal participant weight, including when the same cluster is drawn
    repeatedly. The centred-bootstrap two-sided p value is approximate.
    """
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Bootstrap count and confidence must be valid.")
    y, p_a, subjects = _arrays(y, p_a, subject_ids)
    _, p_b, _ = _arrays(y, p_b, subject_ids)
    keep = np.isfinite(y) & np.isfinite(p_a) & np.isfinite(p_b)
    y, p_a, p_b, subjects = y[keep].astype(int), p_a[keep], p_b[keep], subjects[keep]
    codes, unique = pd.factorize(subjects, sort=False)
    base_weights = participant_weights(subjects)
    empty = {"metric": metric, "delta_a_minus_b": None, "ci_lower": None, "ci_upper": None, "p_value": None, "n_bootstrap": n_bootstrap, "n_valid": 0, "n_participants": len(unique)}
    if len(unique) < 2:
        return {**empty, "status": "insufficient_participants"}
    estimate_a, estimate_b = _metric(y, p_a, base_weights, metric), _metric(y, p_b, base_weights, metric)
    if estimate_a is None or estimate_b is None:
        return {**empty, "status": "single_class"}
    delta = estimate_a - estimate_b
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_bootstrap):
        multiplicity = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        weights = base_weights * multiplicity[codes]
        included = weights > 0
        aa, bb = _metric(y[included], p_a[included], weights[included], metric), _metric(y[included], p_b[included], weights[included], metric)
        if aa is not None and bb is not None:
            draws.append(aa - bb)
    if not draws:
        return {**empty, "delta_a_minus_b": delta, "status": "no_valid_resamples"}
    draws = np.asarray(draws)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(draws, [alpha, 1 - alpha])
    pvalue = (1 + np.count_nonzero(np.abs(draws - delta) >= abs(delta) - 1e-12)) / (len(draws) + 1)
    return {**empty, "status": "ok", "estimate_a": estimate_a, "estimate_b": estimate_b, "delta_a_minus_b": float(delta), "ci_lower": float(lower), "ci_upper": float(upper), "p_value": float(pvalue), "p_value_method": "approximate_centred_participant_bootstrap", "n_valid": len(draws), "n_invalid": n_bootstrap - len(draws), "confidence": confidence, "seed": seed}


def holm_adjust(pvalues: Mapping[str, float | None] | Sequence[float | None]):
    """Holm familywise adjustment; missing tests remain missing in the family."""
    is_mapping = isinstance(pvalues, Mapping)
    keys = list(pvalues) if is_mapping else list(range(len(pvalues)))
    values = [pvalues[key] for key in keys]
    for value in values:
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("P values must be probabilities or None.")
    order = sorted([i for i, value in enumerate(values) if value is not None], key=lambda i: values[i])
    adjusted: list[float | None] = [None] * len(values)
    running = 0.0
    # The prespecified family size includes unavailable tests conservatively.
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[i]))
        adjusted[i] = running
    return dict(zip(keys, adjusted)) if is_mapping else adjusted
