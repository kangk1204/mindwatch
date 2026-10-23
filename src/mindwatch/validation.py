"""Disjoint participant partitions and weighted descriptive metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .models import _binary_vectors, person_weights
from .research.evaluation import calibration_parameters


def assert_disjoint(*groups):
    sets = [set(map(str, group)) for group in groups]
    for i, first in enumerate(sets):
        if any(first & other for other in sets[i+1:]):
            raise ValueError("Participant overlap across population fitting/evaluation roles")


def assign_roles(baselines: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign baseline-only rows to train/selection/calibration/test (60/10/10/20).

    Callers must supply exactly one first-observation row per person. No future
    outcome, maximum across visits, or follow-up availability is used here.
    """
    if set(baselines.columns) != {"subject_id", "baseline_phq9"}:
        raise ValueError("Only participant key and baseline PHQ are permitted")
    if baselines.subject_id.isna().any() or baselines.subject_id.astype(str).duplicated().any():
        raise ValueError("One baseline row per participant required")
    scores = pd.to_numeric(baselines.baseline_phq9, errors="raise")
    if not (np.isfinite(scores) & scores.between(5, 27)).all():
        raise ValueError("Valid baseline PHQ>=5 required")
    result = baselines[["subject_id"]].copy()
    result["subject_id"] = result.subject_id.astype(str)
    result["role"] = ""
    rng = np.random.default_rng(seed)
    for severe in (False, True):
        keys = sorted(result.loc[(scores >= 10) == severe, "subject_id"])
        keys = rng.permutation(keys)
        a, b, c = int(.6*len(keys)), int(.7*len(keys)), int(.8*len(keys))
        for role, group in zip(("train", "selection", "calibration", "test"),
                               (keys[:a], keys[a:b], keys[b:c], keys[c:])):
            result.loc[result.subject_id.isin(group), "role"] = role
    return result


def classification_metrics(y, p, ids, threshold=.5):
    y, p, ids = _binary_vectors(y, p, ids)
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A finite probability threshold is required")
    w = person_weights(ids)
    both = np.unique(y).size == 2
    result = {"AUROC": float(roc_auc_score(y, p, sample_weight=w)) if both else None,
              "AUPRC": float(average_precision_score(y, p, sample_weight=w)) if both else None,
              "Brier": float(np.average((p-y)**2, weights=w)),
              "calibration_intercept": None, "calibration_slope": None}
    # Diagnostic fit on evaluation labels; never used to change predictions.
    result.update(calibration_parameters(y, p, w))
    positive = p >= threshold
    for name, mask, correct in (("sensitivity", y == 1, positive), ("specificity", y == 0, ~positive),
                                ("PPV", positive, y == 1), ("NPV", ~positive, y == 0)):
        result[name] = float(np.average(correct[mask], weights=w[mask])) if mask.any() else None
    return result
