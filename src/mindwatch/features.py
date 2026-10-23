"""Causal, date-only survey history and actually observed EMA features.

Calendar dates must already be resolved to the study's local calendar. This
module never guesses a timezone, interpolates EMA, or infers device units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCALES = {"phq9": 27, "gad7": 21, "isi": 28}
DOMAINS = ("ema_depression", "ema_anxiety", "ema_sleep", "ema_stress")
H_COLUMNS = (
    "h_prior_visit_count", "h_days_since_previous", "h_elapsed_days_from_first",
    *(f"h_{stem}_{name}" for name in SCALES
      for stem in ("baseline", "previous", "past_mean", "past_change", "past_count")),
    *(f"h_previous_phq9_hinge{k}" for k in (5, 10, 15)),
    "h_log_days_since_previous",
)
E_COLUMNS = tuple(f"e_{d}_{s}" for d in DOMAINS for s in ("last", "median", "days"))
FEATURE_COLUMNS = (*H_COLUMNS, *E_COLUMNS)


def _calendar(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    if not {"subject_id", date}.issubset(frame) or frame.columns.duplicated().any():
        raise ValueError("Unique columns and subject/date fields are required")
    result = frame.copy().reset_index(drop=True)
    if result.subject_id.isna().any():
        raise ValueError("Missing subject key")
    # Conversion is explicit and checked for collisions, e.g. integer/string IDs.
    if result.subject_id.nunique() != result.subject_id.astype(str).nunique():
        raise ValueError("Subject key string conversion would merge people")
    result["subject_id"] = result.subject_id.astype(str)
    parsed = pd.to_datetime(result[date], errors="raise")
    if parsed.isna().any() or parsed.dt.tz is not None:
        raise ValueError("Use nonmissing, timezone-resolved local calendar dates")
    if not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError("Date-only input required; resolve intraday timing upstream")
    result[date] = parsed
    if result.duplicated(["subject_id", date]).any():
        raise ValueError("Duplicate subject/date observations require upstream resolution")
    return result.sort_values(["subject_id", date], kind="stable").reset_index(drop=True)


def validate_visits(visits: pd.DataFrame) -> pd.DataFrame:
    result = _calendar(visits, "index_date")
    if not set(SCALES).issubset(result):
        raise ValueError("All three score fields are required; use NaN if unavailable")
    for name, maximum in SCALES.items():
        score = pd.to_numeric(result[name], errors="raise").astype(float)
        if (score.notna() & (~np.isfinite(score) | (score < 0) | (score > maximum) | score.ne(score.round()))).any():
            raise ValueError("Score outside declared scale")
        result[name] = score
    if "sample_id" not in result:
        result["sample_id"] = result.subject_id + "@" + result.index_date.dt.strftime("%Y-%m-%d")
    if result.sample_id.isna().any() or result.sample_id.astype(str).duplicated().any():
        raise ValueError("Sample keys must be nonmissing and unique")
    result["sample_id"] = result.sample_id.astype(str)
    return result


def validate_ema(ema: pd.DataFrame) -> pd.DataFrame:
    result = _calendar(ema, "date")
    if not set(DOMAINS).issubset(result):
        raise ValueError("All four EMA fields are required; use NaN if unavailable")
    for name in DOMAINS:
        value = pd.to_numeric(result[name], errors="raise").astype(float)
        if (value.notna() & (~np.isfinite(value) | (value < 0) | (value > 10))).any():
            raise ValueError("EMA requires an explicitly harmonized 0 to 10 scale")
        result[name] = value
    return result


def history_features(visits: pd.DataFrame) -> pd.DataFrame:
    """Include only earlier scores, available from the following calendar day.

    Baseline/previous are the first/last available valid score for each scale.
    A query visit may have all current score fields missing.
    """
    data = validate_visits(visits)
    records = []
    for _, group in data.groupby("subject_id", sort=False):
        for pos in range(len(group)):
            row = group.iloc[pos]
            prior = group.iloc[:pos]
            prior = prior[prior.index_date + pd.Timedelta(days=1) <= row.index_date]
            z = {"h_prior_visit_count": float(len(prior)),
                 "h_days_since_previous": float((row.index_date-prior.index_date.iloc[-1]).days) if len(prior) else np.nan,
                 "h_elapsed_days_from_first": float((row.index_date-group.index_date.iloc[0]).days)}
            for scale in SCALES:
                known = prior[scale].dropna().to_numpy(float)
                z.update({f"h_baseline_{scale}": known[0] if len(known) else np.nan,
                          f"h_previous_{scale}": known[-1] if len(known) else np.nan,
                          f"h_past_mean_{scale}": known.mean() if len(known) else np.nan,
                          f"h_past_change_{scale}": known[-1]-known[0] if len(known) else np.nan,
                          f"h_past_count_{scale}": float(len(known))})
            for knot in (5, 10, 15):
                z[f"h_previous_phq9_hinge{knot}"] = np.maximum(0, z["h_previous_phq9"]-knot)
            z["h_log_days_since_previous"] = np.log1p(z["h_days_since_previous"])
            records.append(z)
    return pd.DataFrame(records, columns=H_COLUMNS)


def build_features(visits: pd.DataFrame, ema: pd.DataFrame, window_days: int = 14):
    """Return aligned metadata and features, including unlabeled query visits.

    EMA uses [prediction date - window, prediction date). Metadata/outcomes are
    never predictors. This operation does not fit population statistics.
    """
    if not isinstance(window_days, int) or window_days < 1:
        raise ValueError("Positive integer EMA lookback required")
    data, events = validate_visits(visits), validate_ema(ema)
    h = history_features(data)
    first = data.groupby("subject_id", sort=False).head(1)
    baseline = first.set_index("subject_id").phq9
    eligible = data.subject_id.map(baseline).ge(5) & ~data.sample_id.isin(first.sample_id)
    meta = data.loc[eligible, ["sample_id", "subject_id", "index_date"]].copy()
    meta["baseline_phq9"] = meta.subject_id.map(baseline)
    meta["latest_survey_input_date"] = data.groupby("subject_id").index_date.shift().loc[eligible]
    meta["EMA_window_days"] = window_days
    groups = {key: g for key, g in events.groupby("subject_id", sort=False)}
    records, latest = [], []
    for row in meta.itertuples():
        group = groups.get(row.subject_id, events.iloc[:0])
        past = group[(group.date >= row.index_date-pd.Timedelta(days=window_days)) & (group.date < row.index_date)]
        z = {}
        for domain in DOMAINS:
            observed = past.loc[past[domain].notna(), ["date", domain]]
            values = observed[domain].to_numpy(float)
            z[f"e_{domain}_last"] = values[-1] if len(values) else np.nan
            z[f"e_{domain}_median"] = np.median(values) if len(values) else np.nan
            z[f"e_{domain}_days"] = (row.index_date-observed.date.iloc[-1]).days if len(values) else np.nan
        observed = past[past[list(DOMAINS)].notna().any(axis=1)]
        latest.append(observed.date.max() if len(observed) else pd.NaT)
        records.append(z)
    meta["latest_EMA_input_date"] = pd.to_datetime(latest)
    x = pd.concat([h.loc[eligible].reset_index(drop=True), pd.DataFrame(records, columns=E_COLUMNS, dtype=float)], axis=1)
    return meta.reset_index(drop=True), x


def build_dataset(visits: pd.DataFrame, ema: pd.DataFrame, window_days: int = 14):
    """Attach current PHQ>=10 labels only after causal feature construction."""
    data = validate_visits(visits)
    meta, x = build_features(data, ema, window_days)
    scores = meta.sample_id.map(data.set_index("sample_id").phq9)
    keep = scores.notna()
    meta = meta.loc[keep].copy()
    meta["target"] = scores.loc[keep].ge(10).astype(int)
    meta["label_available_at"] = meta.index_date + pd.Timedelta(days=1)
    return meta.reset_index(drop=True), x.loc[keep].reset_index(drop=True)


def ema_forecasts(events: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Previous value and fixed P0 predict the next actually observed response.

    P0 = half the last value plus half the median of up to three past responses.
    First responses have no forecast. No population fitting or daily imputation.
    """
    if domain not in DOMAINS:
        raise ValueError("Unknown EMA domain")
    data = validate_ema(events)
    rows = []
    for sid, group in data.groupby("subject_id", sort=False):
        observed = group[group[domain].notna()]
        values = observed[domain].to_numpy(float)
        for pos in range(1, len(observed)):
            rows.append({"subject_id": sid, "date": observed.date.iloc[pos],
                         "target": values[pos], "last": values[pos-1],
                         "P0": .5*values[pos-1]+.5*np.median(values[max(0, pos-3):pos])})
    return pd.DataFrame(rows, columns=["subject_id", "date", "target", "last", "P0"])
