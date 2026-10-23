"""Label-agnostic, participant-purged calendar-forward validation.

Only development frames are accepted. Returned arrays are positional ``iloc``
indices into the unchanged input frame, never DataFrame index labels. Outcome
values are neither accessed nor used to choose calendar boundaries. Metadata
contains aggregate counts and date bounds only.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _date_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError("Numeric dates require explicit upstream unit conversion.")
    try:
        dates = pd.to_datetime(values, errors="coerce", format="mixed")
        if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
            raise ValueError
    except (TypeError, ValueError, AttributeError, OverflowError):
        raise ValueError("Dates must be complete, valid, timezone-naive local calendar values.") from None
    return dates.reset_index(drop=True)


def _boundary(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, float, bool, np.number)):
        raise ValueError("Numeric calendar boundaries require explicit upstream unit conversion.")
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is not None:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Calendar boundaries must be valid timezone-naive timestamps.") from None
    return result


def _validate(frame: pd.DataFrame, label_time_column: str | None = None):
    required = {"sample_id", "subject_id", "index_date", "partition"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("A nonempty development frame with sample, participant and date columns is required.")
    if frame["partition"].isna().any() or not frame["partition"].eq("development").all():
        raise ValueError("Forward validation accepts development rows only; mixed or test partitions are forbidden.")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Sample identifiers must be present and unique.")
    if frame["subject_id"].isna().any() or frame["subject_id"].astype(str).str.strip().eq("").any() or frame["sample_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Sample and participant identifiers cannot be missing or empty.")
    dates = _date_series(frame["index_date"])
    target_dates = _date_series(frame["target_date"]) if "target_date" in frame else None
    if target_dates is not None and (target_dates < dates).any():
        raise ValueError("Outcome dates cannot precede their index dates.")
    if label_time_column is None:
        if target_dates is not None and (target_dates > dates).any():
            raise ValueError("Future outcomes require an explicit label_time_column and availability purge.")
        labels = dates
    else:
        if label_time_column not in frame:
            raise ValueError("The explicit label availability column is absent.")
        labels = _date_series(frame[label_time_column])
        if (labels < dates).any() or (target_dates is not None and (labels < target_dates).any()):
            raise ValueError("Label availability cannot precede the index or target date.")
    return dates, labels, frame["subject_id"].reset_index(drop=True)


def _days(value: int, name: str, *, positive=False) -> pd.Timedelta:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (1 if positive else 0):
        raise ValueError("Calendar durations must be integer days in the permitted range.")
    return pd.Timedelta(days=int(value))


def _positions(values: Sequence[int], length: int) -> np.ndarray:
    positions = np.asarray(values)
    if positions.ndim != 1:
        raise ValueError("Fold positions must be one-dimensional integer arrays.")
    if positions.size == 0:
        return np.empty(0, dtype=np.int64)
    if positions.dtype.kind not in "iu" or len(np.unique(positions)) != len(positions) or positions.min() < 0 or positions.max() >= length:
        raise ValueError("Fold positions must be unique, in-range integer iloc positions.")
    return positions.astype(np.int64)


def _summary(positions, dates, labels, subjects):
    return {"n_rows": len(positions), "n_participants": int(subjects.iloc[positions].nunique()),
            "index_date_min": dates.iloc[positions].min().isoformat() if len(positions) else None,
            "index_date_max": dates.iloc[positions].max().isoformat() if len(positions) else None,
            "label_available_min": labels.iloc[positions].min().isoformat() if len(positions) else None,
            "label_available_max": labels.iloc[positions].max().isoformat() if len(positions) else None}


def assert_forward_safety(frame: pd.DataFrame, fit: Sequence[int], validation: Sequence[int], *,
                          calibration: Sequence[int] | None = None, gap_days: int = 14,
                          label_time_column: str | None = None) -> dict[str, Any]:
    """Raise on participant/time/label leakage; otherwise return aggregate checks.

    Empty partitions are retained and reported explicitly. Class support is a
    runner responsibility; this helper never inspects outcome values.
    """
    gap = _days(gap_days, "gap_days")
    dates, labels, subjects = _validate(frame, label_time_column)
    fit, validation = _positions(fit, len(frame)), _positions(validation, len(frame))
    cal = _positions(calibration, len(frame)) if calibration is not None else None
    roles = {"fit": fit, "validation": validation}
    if cal is not None:
        roles["calibration"] = cal
    checks: dict[str, bool | None] = {"development_only": True, "unique_samples": True}
    names = list(roles)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = roles[left], roles[right]
            no_row_overlap = not np.intersect1d(a, b).size
            no_person_overlap = not set(subjects.iloc[a]) & set(subjects.iloc[b])
            checks[f"disjoint_rows_{left}_{right}"] = bool(no_row_overlap)
            checks[f"disjoint_participants_{left}_{right}"] = bool(no_person_overlap)
            if not no_row_overlap or not no_person_overlap:
                raise ValueError("Forward fold roles overlap by sample or participant.")
    comparisons = [("fit", "validation")]
    if cal is not None:
        comparisons.extend([("fit", "calibration"), ("calibration", "validation")])
    for earlier, later in comparisons:
        a, b = roles[earlier], roles[later]
        index_ok = bool(dates.iloc[a].max() + gap < dates.iloc[b].min()) if len(a) and len(b) else None
        label_ok = bool(labels.iloc[a].max() + gap < dates.iloc[b].min()) if len(a) and len(b) else None
        checks[f"strict_index_gap_{earlier}_{later}"] = index_ok
        checks[f"strict_label_availability_gap_{earlier}_{later}"] = label_ok
        if index_ok is False or label_ok is False:
            raise ValueError("Forward fold dates or label availability violate the strict calendar gap.")
    return {"status": "ok" if all(len(rows) for rows in roles.values()) else "empty_partition",
            "gap_days": int(gap_days), "label_timing": "index_date_current_outcome" if label_time_column is None else "explicit_label_availability",
            "partitions": {name: _summary(rows, dates, labels, subjects) for name, rows in roles.items()},
            "checks": checks, "class_support": "not_inspected_runner_must_check"}


def make_forward_fold(frame: pd.DataFrame, validation_start, validation_end, gap_days: int = 14,
                      calibration_days: int = 28, *, label_time_column: str | None = None) -> dict[str, Any]:
    """One calendar outer fold with chronological, participant-disjoint roles.

    Validation is [start,end). Calibration is [start-gap-calibration_days,
    start-gap). Fit index/label availability must be < calibration_start-gap.
    Every validation participant is purged from fit/calibration, and every
    retained calibration participant is purged from fit. The explicit future
    outcome mode also purges labels unavailable at the corresponding boundary.
    """
    gap = _days(gap_days, "gap_days")
    cal_duration = _days(calibration_days, "calibration_days", positive=True)
    start, end = _boundary(validation_start), _boundary(validation_end)
    if start >= end:
        raise ValueError("Validation end must be later than validation start.")
    dates, labels, subjects = _validate(frame, label_time_column)
    cal_end, cal_start = start - gap, start - gap - cal_duration
    fit_end = cal_start - gap
    validation_mask = (dates >= start) & (dates < end)
    validation_people = set(subjects.loc[validation_mask])
    cal_candidates = (dates >= cal_start) & (dates < cal_end)
    cal_after_people = cal_candidates & ~subjects.isin(validation_people)
    cal_mask = cal_after_people & (labels < cal_end)
    calibration_people = set(subjects.loc[cal_mask])
    fit_candidates = dates < fit_end
    fit_after_people = fit_candidates & ~subjects.isin(validation_people | calibration_people)
    fit_mask = fit_after_people & (labels < fit_end)
    fit, calibration, validation = [np.flatnonzero(mask.to_numpy()).astype(np.int64) for mask in (fit_mask, cal_mask, validation_mask)]
    metadata = assert_forward_safety(frame, fit, validation, calibration=calibration, gap_days=gap_days,
                                     label_time_column=label_time_column)
    metadata.update(validation_interval={"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat()},
                    calibration_interval={"start_inclusive": cal_start.isoformat(), "end_exclusive": cal_end.isoformat()},
                    fit_end_exclusive=fit_end.isoformat(), calibration_days=int(calibration_days),
                    purge_counts={"fit_candidate_rows": int(fit_candidates.sum()), "calibration_candidate_rows": int(cal_candidates.sum()),
                                  "fit_rows_removed_for_participant_overlap": int((fit_candidates & ~fit_after_people).sum()),
                                  "calibration_rows_removed_for_validation_participants": int((cal_candidates & ~cal_after_people).sum()),
                                  "fit_rows_removed_for_label_availability": int((fit_after_people & ~fit_mask).sum()),
                                  "calibration_rows_removed_for_label_availability": int((cal_after_people & ~cal_mask).sum())})
    return {"fit": fit, "calibration": calibration, "validation": validation, "metadata": metadata}


def forward_splits(frame: pd.DataFrame, n_splits: int = 2, gap_days: int = 14,
                   min_train_fraction: float = .4, *, label_time_column: str | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding inner folds using unique local calendar days, never labels.

    Starting after ceil(unique_days * min_train_fraction), split remaining days
    into deterministic contiguous validation blocks. Each fit uses all strictly
    earlier days/available labels outside the gap and purges every validation
    participant. Return sklearn-compatible positional (fit,validation) pairs.
    Empty fit arrays are returned rather than hidden or repaired with random CV.
    Fewer than requested folds are returned only when fewer validation days exist.
    """
    gap = _days(gap_days, "gap_days")
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, (int, np.integer)) or n_splits < 1:
        raise ValueError("Inner split count must be a positive integer.")
    if not np.isfinite(min_train_fraction) or not 0 < min_train_fraction < 1:
        raise ValueError("Minimum training fraction must lie strictly between zero and one.")
    dates, labels, subjects = _validate(frame, label_time_column)
    unique = pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()
    if len(unique) < 2:
        raise ValueError("At least two unique calendar days are required for forward validation.")
    first = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * min_train_fraction))))
    chunks = np.array_split(np.arange(first, len(unique)), min(int(n_splits), len(unique)-first))
    output = []
    for chunk in chunks:
        start = unique[int(chunk[0])]
        next_index = int(chunk[-1]) + 1
        end = unique[next_index] if next_index < len(unique) else unique[-1] + pd.Timedelta(days=1)
        validation_mask = (dates >= start) & (dates < end)
        people = set(subjects.loc[validation_mask])
        fit_mask = (dates < start - gap) & (labels < start - gap) & ~subjects.isin(people)
        fit = np.flatnonzero(fit_mask.to_numpy()).astype(np.int64)
        validation = np.flatnonzero(validation_mask.to_numpy()).astype(np.int64)
        assert_forward_safety(frame, fit, validation, gap_days=gap_days, label_time_column=label_time_column)
        output.append((fit, validation))
    return output
