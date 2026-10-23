"""Pure calendar splits for known-user monitoring and new-user comparison.

All arrays are positional iloc indices into the unchanged development frame.
No outcome values, features, class support or model scores determine boundaries.
Only fit/calibration labels need availability before their frozen role cutoff;
validation labels available after month end remain in the denominator.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .temporal_splits import _boundary, _date_series, _days, _positions, _summary, _validate


def _inputs(frame, mode, label_time_column):
    if mode not in {"known_user", "new_user"}:
        raise ValueError("Monitoring mode must be known_user or new_user.")
    dates, labels, subjects = _validate(frame, label_time_column)
    if "prediction_time" in frame:
        prediction = _date_series(frame["prediction_time"])
        if not prediction.equals(dates):
            raise ValueError("Prediction provenance must equal the index-date anchor; earlier prediction requires a separately anchored forecast protocol.")
        source = "provided_prediction_time"
    else:
        prediction, source = dates, "index_date_anchor_prediction_time_not_provided"
    return dates, labels, subjects, prediction, source


def assert_monitoring_safety(frame: pd.DataFrame, fit: Sequence[int], validation: Sequence[int], *,
                             calibration: Sequence[int] | None = None, mode: str = "known_user",
                             gap_days: int = 14, calibration_days: int = 28,
                             label_time_column: str | None = "label_available_at",
                             validation_start=None, validation_end=None) -> dict[str, Any]:
    """Assert row/time/availability safety and the declared participant policy.

    Supplying both validation boundaries additionally checks exact frozen role
    windows. Without them only actual selected-date gaps are asserted. Empty
    roles are reported explicitly; outcome class support is never inspected.
    """
    gap = _days(gap_days, "gap_days")
    duration = _days(calibration_days, "calibration_days", positive=True)
    dates, labels, subjects, prediction, prediction_source = _inputs(frame, mode, label_time_column)
    roles = {"fit": _positions(fit, len(frame)), "validation": _positions(validation, len(frame))}
    if calibration is not None:
        roles["calibration"] = _positions(calibration, len(frame))
    checks: dict[str, bool | None] = {"development_only": True, "unique_samples": True,
                                    "prediction_time_at_or_before_index": True,
                                    "label_availability_at_or_after_index_and_target": True}
    overlaps = {}
    names = list(roles)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = roles[left], roles[right]
            row_disjoint = not np.intersect1d(a, b).size
            common_people = set(subjects.iloc[a]) & set(subjects.iloc[b])
            participant_policy = mode == "known_user" or not common_people
            checks[f"disjoint_rows_{left}_{right}"] = bool(row_disjoint)
            checks[f"participant_policy_{left}_{right}"] = bool(participant_policy)
            overlaps[f"{left}_{right}_participants"] = len(common_people)
            if not row_disjoint or not participant_policy:
                raise ValueError("Monitoring roles violate row separation or their participant policy.")
    comparisons = [("fit", "validation")]
    if calibration is not None:
        comparisons += [("fit", "calibration"), ("calibration", "validation")]
    for earlier, later in comparisons:
        a, b = roles[earlier], roles[later]
        index_ok = bool(dates.iloc[a].max() + gap < dates.iloc[b].min()) if len(a) and len(b) else None
        label_ok = bool(labels.iloc[a].max() + gap < dates.iloc[b].min()) if len(a) and len(b) else None
        checks[f"strict_index_gap_{earlier}_{later}"] = index_ok
        checks[f"strict_label_gap_{earlier}_{later}"] = label_ok
        if index_ok is False or label_ok is False:
            raise ValueError("Monitoring dates or label availability violate the strict chronological gap.")
    if (validation_start is None) != (validation_end is None):
        raise ValueError("Both validation boundaries are required together.")
    calendar = None
    late_validation = None
    if validation_start is not None:
        start, end = _boundary(validation_start), _boundary(validation_end)
        if start >= end:
            raise ValueError("Validation end must be later than validation start.")
        cal_end, cal_start = start-gap, start-gap-duration
        fit_end = cal_start-gap if calibration is not None else start-gap
        validation_rows, fit_rows = roles["validation"], roles["fit"]
        checks["validation_in_fixed_interval"] = bool(((dates.iloc[validation_rows] >= start) & (dates.iloc[validation_rows] < end)).all())
        checks["fit_index_and_labels_before_fixed_cutoff"] = bool(((dates.iloc[fit_rows] < fit_end) & (labels.iloc[fit_rows] < fit_end)).all())
        if calibration is not None:
            cal = roles["calibration"]
            checks["calibration_in_fixed_interval_with_available_labels"] = bool(((dates.iloc[cal] >= cal_start) & (dates.iloc[cal] < cal_end) & (labels.iloc[cal] < cal_end)).all())
        if any(value is False for value in checks.values()):
            raise ValueError("Monitoring roles violate their fixed calendar boundaries.")
        calendar = {"validation_start_inclusive": start.isoformat(), "validation_end_exclusive": end.isoformat(),
                    "fit_end_exclusive": fit_end.isoformat(),
                    "calibration_start_inclusive": cal_start.isoformat() if calibration is not None else None,
                    "calibration_end_exclusive": cal_end.isoformat() if calibration is not None else None}
        late_validation = int((labels.iloc[validation_rows] >= end).sum())
    validation_people = set(subjects.iloc[roles["validation"]])
    prior_people = set(subjects.iloc[roles["fit"]])
    if "calibration" in roles:
        prior_people |= set(subjects.iloc[roles["calibration"]])
    return {"status": "ok" if all(len(rows) for rows in roles.values()) else "empty_partition",
            "mode": mode, "gap_days": int(gap_days), "calibration_days": int(calibration_days),
            "prediction_time_source": prediction_source,
            "label_timing": "explicit_label_availability" if label_time_column is not None else "index_date_current_outcome",
            "partitions": {name: _summary(rows, dates, labels, subjects) for name, rows in roles.items()},
            "validation_participants_seen_in_fit_or_calibration": len(validation_people & prior_people),
            "validation_participants_not_seen_in_fit_or_calibration": len(validation_people - prior_people),
            "participant_overlap_counts": overlaps, "checks": checks, "calendar_boundaries": calendar,
            "validation_labels_available_at_or_after_window_end": late_validation,
            "validation_denominator_policy": "all_index_rows_in_window; late_observed_outcomes_retained",
            "within_month_updates": False, "class_support": "not_inspected_runner_must_check"}


def make_monitoring_fold(frame: pd.DataFrame, validation_start, validation_end, mode: str = "known_user",
                         gap_days: int = 14, calibration_days: int = 28,
                         label_time_column: str | None = "label_available_at") -> dict[str, Any]:
    """Create one frozen month fold, allowing genuine past known-user history.

    For default settings and validation start T: fit index AND label <T-56d;
    calibration index in [T-42d,T-14d) AND label <T-14d. Validation is every
    index date in [start,end), including outcomes observed after end. No model or
    calibration update uses any outcome from inside that validation month.
    """
    gap = _days(gap_days, "gap_days")
    duration = _days(calibration_days, "calibration_days", positive=True)
    start, end = _boundary(validation_start), _boundary(validation_end)
    if start >= end:
        raise ValueError("Validation end must be later than validation start.")
    dates, labels, subjects, _, _ = _inputs(frame, mode, label_time_column)
    cal_end, cal_start = start-gap, start-gap-duration
    fit_end = cal_start-gap
    validation_mask = (dates >= start) & (dates < end)
    fit_candidates = dates < fit_end
    calibration_candidates = (dates >= cal_start) & (dates < cal_end)
    fit_labels = fit_candidates & (labels < fit_end)
    calibration_labels = calibration_candidates & (labels < cal_end)
    fit_mask, calibration_mask = fit_labels.copy(), calibration_labels.copy()
    if mode == "new_user":
        validation_people = set(subjects.loc[validation_mask])
        calibration_mask &= ~subjects.isin(validation_people)
        calibration_people = set(subjects.loc[calibration_mask])
        fit_mask &= ~subjects.isin(validation_people | calibration_people)
    fit, calibration, validation = [np.flatnonzero(mask.to_numpy()).astype(np.int64) for mask in (fit_mask, calibration_mask, validation_mask)]
    metadata = assert_monitoring_safety(frame, fit, validation, calibration=calibration, mode=mode,
        gap_days=gap_days, calibration_days=calibration_days, label_time_column=label_time_column,
        validation_start=start, validation_end=end)
    metadata["purge_counts"] = {
        "fit_candidate_rows": int(fit_candidates.sum()), "calibration_candidate_rows": int(calibration_candidates.sum()),
        "fit_rows_removed_for_label_availability": int((fit_candidates & ~fit_labels).sum()),
        "calibration_rows_removed_for_label_availability": int((calibration_candidates & ~calibration_labels).sum()),
        "fit_rows_removed_for_participant_policy": int((fit_labels & ~fit_mask).sum()),
        "calibration_rows_removed_for_participant_policy": int((calibration_labels & ~calibration_mask).sum()),
    }
    return {"fit": fit, "calibration": calibration, "validation": validation, "metadata": metadata}


def monitoring_forward_splits(frame: pd.DataFrame, n_splits: int = 2, gap_days: int = 14,
                              min_train_fraction: float = .4, mode: str = "known_user",
                              label_time_column: str | None = "label_available_at") -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic unique-calendar-day inner folds with frozen label timing.

    Fit index/label must be strictly before validation_start-gap. known_user
    permits earlier visits by a validation participant; new_user purges them.
    Date boundaries never adapt to outcome values or class support.
    """
    gap = _days(gap_days, "gap_days")
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, (int, np.integer)) or n_splits < 1:
        raise ValueError("Monitoring split count must be a positive integer.")
    if not isinstance(min_train_fraction, (int, float, np.integer, np.floating)) or not np.isfinite(min_train_fraction) or not 0 < min_train_fraction < 1:
        raise ValueError("Minimum training fraction must lie strictly between zero and one.")
    dates, labels, subjects, _, _ = _inputs(frame, mode, label_time_column)
    unique = pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()
    if len(unique) < 2:
        raise ValueError("At least two unique calendar days are required.")
    first = min(len(unique)-1, max(1, int(np.ceil(len(unique)*min_train_fraction))))
    blocks = np.array_split(np.arange(first, len(unique)), min(int(n_splits), len(unique)-first))
    output = []
    for block in blocks:
        start = unique[int(block[0])]
        after = int(block[-1])+1
        end = unique[after] if after < len(unique) else unique[-1]+pd.Timedelta(days=1)
        validation_mask = (dates >= start) & (dates < end)
        fit_mask = (dates < start-gap) & (labels < start-gap)
        if mode == "new_user":
            fit_mask &= ~subjects.isin(set(subjects.loc[validation_mask]))
        fit, validation = [np.flatnonzero(mask.to_numpy()).astype(np.int64) for mask in (fit_mask, validation_mask)]
        assert_monitoring_safety(frame, fit, validation, mode=mode, gap_days=gap_days,
            label_time_column=label_time_column, validation_start=start, validation_end=end)
        output.append((fit, validation))
    return output
