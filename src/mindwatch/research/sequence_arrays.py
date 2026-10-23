"""Causal NumPy sequence packing; usable without neural runtimes."""
from __future__ import annotations
import numpy as np

def past_only_delta(mask: np.ndarray, step_days: float = 1.0) -> np.ndarray:
    """Elapsed days at each step, independent of future masks/values.

    At the first position delta is zero. At an observed position delta is time
    since the preceding observation, not zero; the current observation resets the
    next step. This is the convention in the GRU-D recurrence.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or step_days <= 0:
        raise ValueError("mask must be binary [N,T,F], step_days must be positive")
    delta = np.zeros(mask.shape, dtype=np.float32)
    for t in range(1, mask.shape[1]):
        delta[:, t] = step_days + (1 - mask[:, t - 1]) * delta[:, t - 1]
    return delta

def flatten_sequence(values: np.ndarray, mask: np.ndarray, delta: np.ndarray,
                     demographics: np.ndarray | None = None) -> np.ndarray:
    if values.shape != mask.shape or values.shape != delta.shape or values.ndim != 3:
        raise ValueError("Sequences must have equal [N,T,F] shapes")
    arrays = [a.reshape(len(values), -1) for a in (values, mask, delta)]
    if demographics is not None:
        if demographics.ndim != 2 or len(demographics) != len(values) or not np.isfinite(demographics).all():
            raise ValueError("Demographics must be finite encoded [N,D] covariates")
        arrays.append(demographics)
    return np.concatenate(arrays, axis=1)
