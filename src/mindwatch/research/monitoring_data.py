"""Development-only data for follow-up monitoring with strictly past-only inputs.

No participant records, source labels or filenames are printed. Current-state
training/evaluation excludes the first observation of an entry-qualified subject.
Forecast targets are kept in a separate table and never enter monitoring designs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .privacy import sha256


@dataclass
class MonitoringData:
    frame: pd.DataFrame
    designs: dict
    provenance: dict
    prognosis_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    prognosis_designs: dict = field(default_factory=dict)
    source_frame: pd.DataFrame = field(default_factory=pd.DataFrame)


SCALES = {"phq9": (27, 10), "gad7": (21, 10), "isi": (28, 15)}


def _validated_scores(values: pd.Series, scale: str) -> pd.Series:
    upper = SCALES[scale][0]
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.between(0, upper) & numeric.eq(numeric.round()))


def _visit_keys(visits: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "subject_id", "index_date"}
    if not required.issubset(visits):
        raise ValueError("Unique sample/subject/date fields are required")
    result = visits.copy().reset_index(drop=True)
    result["index_date"] = pd.to_datetime(result.index_date, errors="coerce")
    if result.sample_id.isna().any() or result.sample_id.duplicated().any() or result.subject_id.isna().any() or result.index_date.isna().any():
        raise ValueError("Present unique samples, subjects and dates are required")
    if result.duplicated(["subject_id", "index_date"]).any():
        raise ValueError("Duplicate subject assessment times require resolution")
    return result


def build_past_history(visits: pd.DataFrame) -> pd.DataFrame:
    """Build sample-keyed history from earlier released labels only.

    A label becomes available one day after its index timestamp (date-only proxy).
    For each scale, baseline/previous/mean/change use available valid past scores;
    change is last available minus the first available score. It is zero when
    only that baseline has been observed, and missing if no score is available.
    No current score is included in its own feature row, even at the first visit.
    """
    data = _visit_keys(visits)
    if not set(SCALES).issubset(data):
        raise ValueError("PHQ, GAD and ISI score fields are required")
    scores = {name: _validated_scores(data[name], name) for name in SCALES}
    rows = []
    for _, group in data.sort_values(["subject_id", "index_date", "sample_id"]).groupby("subject_id", sort=False):
        prior = []
        first_date = group.iloc[0].index_date
        for position in group.index:
            visit = data.loc[position]
            now = visit.index_date
            released = [index for index in prior if data.loc[index, "index_date"]+pd.Timedelta(days=1) <= now]
            row = {"sample_id": visit.sample_id, "history__prior_visit_count": float(len(released)),
                   "history__days_since_previous": float((now-data.loc[released[-1], "index_date"]).total_seconds()/86400) if released else np.nan,
                   "history__elapsed_days_from_first": float((now-first_date).total_seconds()/86400)}
            for name in SCALES:
                known = scores[name].loc[released].dropna().to_numpy(dtype=float)
                row[f"history__baseline_{name}"] = float(known[0]) if len(known) else np.nan
                row[f"history__previous_{name}"] = float(known[-1]) if len(known) else np.nan
                row[f"history__past_mean_{name}"] = float(known.mean()) if len(known) else np.nan
                row[f"history__past_change_{name}"] = float(known[-1]-known[0]) if len(known) else np.nan
                row[f"history__past_count_{name}"] = float(len(known))
            row["history_available"] = bool(any(row[f"history__past_count_{name}"] > 0 for name in SCALES))
            rows.append(row)
            prior.append(position)
    columns = ["sample_id", "history__prior_visit_count", "history__days_since_previous", "history__elapsed_days_from_first"]
    columns += [f"history__{stem}_{name}" for name in SCALES for stem in ("baseline", "previous", "past_mean", "past_change", "past_count")]
    columns += ["history_available"]
    result = pd.DataFrame(rows, columns=columns)
    return data[["sample_id"]].merge(result, on="sample_id", how="left", validate="one_to_one")


def personal_sensor_features(visits: pd.DataFrame, daily: pd.DataFrame, channels: Sequence[str],
                             min_observed_days: int = 14) -> pd.DataFrame:
    """Robust individual baselines [t-42,t-14) versus recent [t-14,t).

    Each channel needs >=14 distinct actually observed baseline days. Insufficient
    history leaves baseline/difference/z missing and sets explicit flags; it never
    excludes a visit. Zero IQR leaves z undefined. Robust z uses IQR / 1.349.
    Future rows are sliced away before their values are inspected or converted.
    """
    data = _visit_keys(visits)
    if min_observed_days < 1 or min_observed_days > 28:
        raise ValueError("Baseline minimum must be between one and 28 observed days")
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Unique nonempty sensor channels are required")
    for name in channels:
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None or
                re.search(r"(?:^|_)(?:phq9|gad7|isi|subject|sample|target|label|id)(?:_|$)", name, re.I)):
            raise ValueError("Channel names must be sensor variables, not labels or identifiers")
    if not {"subject_id", "date", *channels}.issubset(daily):
        raise ValueError("Daily table lacks declared sensor channels")
    calendar = daily[["subject_id", "date", *channels]].copy()
    calendar["date"] = pd.to_datetime(calendar.date, errors="coerce")
    if calendar.subject_id.isna().any() or calendar.date.isna().any() or calendar.duplicated(["subject_id", "date"]).any():
        raise ValueError("Daily observations require unique subject/date keys")
    if not calendar.date.eq(calendar.date.dt.normalize()).all():
        raise ValueError("Personal baseline expects genuinely daily calendar rows")
    groups = {key: group for key, group in calendar.groupby("subject_id", sort=False)}
    rows = []
    for visit in data.itertuples(index=False):
        t = visit.index_date.normalize()
        group = groups.get(visit.subject_id)
        if group is None:
            baseline = recent = calendar.iloc[:0]
        else:
            baseline = group.loc[group.date.ge(t-pd.Timedelta(days=42)) & group.date.lt(t-pd.Timedelta(days=14))]
            recent = group.loc[group.date.ge(t-pd.Timedelta(days=14)) & group.date.lt(t)]
        row = {"sample_id": visit.sample_id}
        for name in channels:
            past = pd.to_numeric(baseline[name], errors="coerce").to_numpy(dtype=float)
            last = pd.to_numeric(recent[name], errors="coerce").to_numpy(dtype=float)
            past, last = past[np.isfinite(past)], last[np.isfinite(last)]
            enough = len(past) >= min_observed_days
            median = float(np.median(past)) if enough else np.nan
            iqr = float(np.percentile(past, 75)-np.percentile(past, 25)) if enough else np.nan
            recent_median = float(np.median(last)) if len(last) else np.nan
            difference = recent_median-median
            prefix, quality = f"sensor__personal_{name}__", f"quality__personal_{name}__"
            row.update({prefix+"baseline_median_28d": median, prefix+"baseline_iqr_28d": iqr,
                        prefix+"recent_median_14d": recent_median, prefix+"recent_minus_baseline": difference,
                        prefix+"robust_z": difference/(iqr/1.349) if enough and iqr > 0 else np.nan,
                        quality+"baseline_observed_days": float(len(past)), quality+"recent_observed_days": float(len(last)),
                        quality+"baseline_insufficient": float(not enough), quality+"baseline_zero_iqr": float(enough and iqr == 0)})
        rows.append(row)
    return pd.DataFrame(rows)


def _aligned_sequences(dataset: Path, frame: pd.DataFrame, window: int):
    from .sequence_arrays import flatten_sequence, past_only_delta
    with np.load(dataset/f"sequences_{window}d.npz", allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError("Sequence sample identifiers must be unique")
        positions = {value: index for index, value in enumerate(ids)}
        if any(str(value) not in positions for value in frame.sample_id):
            raise ValueError("A monitoring visit lacks its original sensor sequence")
        order = np.asarray([positions[str(value)] for value in frame.sample_id], dtype=int)
        values, mask = archive["values"][order], archive["mask"][order]
        channels = archive["feature_names"].astype(str).tolist()
    if values.ndim != 3 or values.shape != mask.shape or values.shape[1] != window or values.shape[2] != len(channels):
        raise ValueError("Original sequence dimensions are inconsistent")
    if not np.isin(mask, [0, 1]).all() or not np.isfinite(values[mask.astype(bool)]).all():
        raise ValueError("Observed sequence values and masks are invalid")
    observed_days = mask.astype(bool).any(axis=2).sum(axis=1)
    if "quality__sensor_observed_days" not in frame or not np.array_equal(observed_days, frame.quality__sensor_observed_days.to_numpy()):
        raise ValueError("Sensor observed-day counts differ from the frozen features")
    if "quality__sensor_eligible" in frame and not np.array_equal(observed_days >= 4, frame.quality__sensor_eligible.eq(1).to_numpy()):
        raise ValueError("Sensor eligibility differs from exact original sequence masks")
    delta = past_only_delta(mask)
    return values, mask, channels, observed_days, flatten_sequence(values, mask, delta)


def _designs(frame, tab, packed):
    h = [name for name in tab if name.startswith(("history__", "demo__"))]
    s = [name for name in tab if name.startswith(("sensor__", "quality__"))]
    e = [name for name in tab if name.startswith("ema__")]
    if not h:
        raise ValueError("A past-history/demographic design is required")
    result = {}
    for name, names in (("A", h), ("B", h+s), ("C", h+e), ("D", h+e+s)):
        if len(set(names)) != len(names):
            raise ValueError("Monitoring feature names collide")
        result[name] = {"tabular": tab[names].to_numpy(dtype=float), "tabular_names": names}
        if name in {"B", "D"}:
            static = h+e if name == "D" else h
            result[name].update(sequence=np.column_stack([packed, tab[static].to_numpy(dtype=float)]),
                                static_names=static, sequence_columns=packed.shape[1])
    return result


def load_monitoring_data(dataset: str | Path, window: int = 14, entry_threshold: float = 5,
                         development_end=None) -> MonitoringData:
    """Prepare one-time entry-qualified, follow-up-only development monitoring.

    The first observed survey determines entry once. Its PHQ must be >= threshold;
    every later development visit remains, including low scores and sparse/no
    sensor windows. Original reserved test participants never enter this loader.
    Label release times are explicit date-only proxies, not known upload times.
    """
    from .temporal_features import temporal_features
    dataset = Path(dataset)
    if window != 14:
        raise ValueError("Monitoring v6 prespecifies the genuine 14-day sequences")
    if not 0 <= entry_threshold <= 27:
        raise ValueError("Entry threshold must lie within the PHQ range")
    spec = json.loads((dataset/"split.json").read_text())
    end = pd.Timestamp(development_end or spec.get("cutoff"))
    if pd.isna(end):
        raise ValueError("Explicit development end date required")
    reserved = spec.get("original_test_subjects_reserved")
    if type(reserved) is not int or reserved < 1:
        raise ValueError("Use amended development data that exclude original reserved test people")
    split = pd.read_parquet(dataset/"split_manifest.parquet")
    if not {"sample_id", "subject_id", "partition", "index_date"}.issubset(split):
        raise ValueError("Locked split metadata are incomplete")
    if split.sample_id.duplicated().any():
        raise ValueError("Split sample identifiers must be unique")
    allowed = split.loc[split.partition.eq("development") & pd.to_datetime(split.index_date).lt(end)].copy()
    forbidden = set(split.loc[split.partition.eq("temporal_test"), "subject_id"])
    if set(allowed.subject_id) & forbidden:
        raise ValueError("Development and reserved test subjects overlap")
    ids = set(allowed.sample_id)
    # Filter membership before score conversion, summary or selection. The amended
    # parquet itself must already exclude original test subjects at preparation.
    visits = pd.read_parquet(dataset/"visits.parquet")
    visits = visits.loc[visits.sample_id.isin(ids)].copy()
    visits = _visit_keys(visits)
    if not set(SCALES).issubset(visits):
        raise ValueError("All PHQ/GAD/ISI scale fields must be present")
    if set(visits.subject_id) & forbidden or visits.index_date.ge(end).any():
        raise ValueError("Restricted development membership changed during join")
    for name, (_, threshold) in SCALES.items():
        visits[name] = _validated_scores(visits[name], name)
        visits[f"y_{name}"] = (visits[name] >= threshold).astype(float).where(visits[name].notna())
    visits = visits.merge(allowed[["sample_id", "subject_id", "partition"]], on=["sample_id", "subject_id"], validate="one_to_one")
    visits = visits.sort_values(["subject_id", "index_date", "sample_id"]).reset_index(drop=True)
    first = visits.groupby("subject_id", sort=False).head(1)
    entry_subjects = set(first.loc[first.phq9.ge(entry_threshold), "subject_id"])
    selected = visits.loc[visits.subject_id.isin(entry_subjects)].copy().reset_index(drop=True)
    selected["baseline_phq9"] = selected.subject_id.map(first.set_index("subject_id").phq9)
    selected["label_available_at"] = selected.index_date+pd.Timedelta(days=1)
    selected["_first_observation"] = ~selected.subject_id.duplicated()
    if selected.empty:
        return MonitoringData(selected.drop(columns="_first_observation"), {}, {
            "status": "no_entry_qualified_development_subjects", "old_test_loaded": False,
            "entry_threshold": entry_threshold, "development_end": str(end),
            "enrollment_basis": "first_observed_survey_proxy", "as_of_upload_time": "unknown"})
    features = pd.read_parquet(dataset/f"features_{window}d.parquet")
    features = features.loc[features.sample_id.isin(set(selected.sample_id))].copy()
    if features.sample_id.duplicated().any() or set(features.sample_id) != set(selected.sample_id):
        raise ValueError("Frozen features must cover every entry-qualified visit exactly once")
    # Rebuild history from source visits; never reuse stale target-derived columns.
    feature_names = [name for name in features if name.startswith(("sensor__", "quality__", "ema__", "demo__"))]
    full = selected.merge(features[["sample_id", *feature_names]], on="sample_id", how="left", validate="one_to_one")
    history = build_past_history(selected)
    full = full.merge(history, on="sample_id", how="left", validate="one_to_one")
    values, mask, channels, days, packed = _aligned_sequences(dataset, full, window)
    full["sensor_available"] = days >= 4
    ema_coverage = [name for name in full if name.startswith("ema__") and name.endswith("__coverage")]
    if ema_coverage:
        coverage = full[ema_coverage].to_numpy(dtype=float)
        if np.any(np.isfinite(coverage) & ((coverage < 0) | (coverage > 1))):
            raise ValueError("EMA coverage must lie between zero and one")
        full["ema_available"] = np.any(np.isfinite(coverage) & (coverage > 0), axis=1)
    else:
        full["ema_available"] = False
    signal, quality = temporal_features(values, mask, channels)
    daily = pd.read_parquet(dataset/"daily.parquet")
    daily = daily.loc[daily.subject_id.isin(entry_subjects)].copy()
    personal = personal_sensor_features(full[["sample_id", "subject_id", "index_date"]], daily, channels).drop(columns="sample_id")
    base = full[[name for name in full if name.startswith(("sensor__", "quality__", "ema__", "demo__", "history__"))]].reset_index(drop=True)
    tab = pd.concat([base, signal, quality, personal], axis=1)
    if tab.columns.duplicated().any():
        raise ValueError("Monitoring feature names collide across sources")
    all_designs = _designs(full, tab, packed)
    monitoring_keep = ~full._first_observation.to_numpy(dtype=bool)
    frame = full.loc[monitoring_keep].drop(columns="_first_observation").reset_index(drop=True)
    designs = {}
    for name, design in all_designs.items():
        designs[name] = {key: value[monitoring_keep] if isinstance(value, np.ndarray) else value.copy() if isinstance(value, list) else value
                         for key, value in design.items()}
    prognosis = full.drop(columns="_first_observation").copy()
    grouped = full.groupby("subject_id", sort=False)
    prognosis["target_date"] = grouped.index_date.shift(-1)
    prognosis["future_phq9"] = grouped.phq9.shift(-1)
    prognosis["horizon_days"] = (prognosis.target_date-prognosis.index_date).dt.total_seconds()/86400
    prognosis_keep = (prognosis.horizon_days.between(90, 180) & prognosis.phq9.notna() & prognosis.future_phq9.notna()).to_numpy()
    prognosis["worsening_ge5"] = (prognosis.future_phq9-prognosis.phq9 >= 5).astype(float).where(prognosis_keep)
    prognosis["onset_ge10"] = (prognosis.future_phq9 >= 10).astype(float).where(prognosis_keep & prognosis.phq9.lt(10))
    prognosis["future_ge10"] = (prognosis.future_phq9 >= 10).astype(float).where(prognosis_keep)
    prognosis["index_label_available_at"] = prognosis.label_available_at
    prognosis["target_available_at"] = prognosis.target_date+pd.Timedelta(days=1)
    prognosis["label_available_at"] = prognosis.target_available_at
    # Forecasting starts only after the current index assessment has been released.
    # Current PHQ is then a known input even for the entry visit; this availability
    # gate is deliberately separate from the past-only current-state gate.
    prognosis["past_history_available"] = prognosis.history_available
    prognosis["history_available"] = prognosis.phq9.notna()
    prognosis["history__forecast_current_phq9"] = prognosis.phq9
    prognosis["history__forecast_current_gad7"] = prognosis.gad7
    prognosis["history__forecast_current_isi"] = prognosis.isi
    prognosis["sensor_window_end"] = prognosis.index_date.dt.normalize()-pd.Timedelta(days=1)
    # Forecast matrices deliberately separate current labels as allowed index-time
    # measurements; they are never appended to a monitoring A/B/C/D design.
    prognosis_designs = {}
    current_names = ["history__forecast_current_phq9", "history__forecast_current_gad7", "history__forecast_current_isi"]
    current = prognosis[current_names].to_numpy(dtype=float)
    for name, design in all_designs.items():
        prognosis_designs[name] = {
            "tabular": np.column_stack([design["tabular"], current])[prognosis_keep],
            "tabular_names": [*design["tabular_names"], *current_names],
        }
    prognosis_frame = prognosis.loc[prognosis_keep].reset_index(drop=True)
    # Keep tabular predictors in the forecast frame as well for simple regressors.
    extras = tab[[name for name in tab if name not in prognosis_frame]].loc[prognosis_keep].reset_index(drop=True)
    prognosis_frame = pd.concat([prognosis_frame, extras], axis=1)
    provenance = {
        "status": "prepared_development_monitoring", "window": window, "channels": channels,
        "entry_rule": "first_observed_survey_PHQ_ge_threshold_once", "entry_threshold": entry_threshold,
        "initial_visits_in_current_state": False, "all_later_visits_retained": True,
        "entry_participants": len(entry_subjects), "followup_participants": int(frame.subject_id.nunique()),
        "followup_visits": len(frame), "followup_phq_positive": int(frame.y_phq9.eq(1).sum()),
        "sensor_available_visits": int(frame.sensor_available.sum()), "ema_available_visits": int(frame.ema_available.sum()),
        "history_available_visits": int(frame.history_available.sum()), "prognosis_pairs": len(prognosis_frame),
        "development_end_exclusive": str(end), "old_test_loaded": False,
        "original_test_people_excluded": reserved, "enrollment_basis": "first_observed_survey_proxy",
        "label_release_proxy": "index_date_plus_1_day", "as_of_upload_time": "unknown",
        "forecast_history_availability": "valid_current_index_PHQ_at_index_label_available_at; includes_entry_visit",
        "sensor_eligibility": "at_least_4_distinct_observed_days_in_exact_14d_mask",
        "individual_sensor_baseline": {"interval": "[t-42,t-14)", "recent_interval": "[t-14,t)", "minimum_observed_days_per_channel": 14,
                                       "robust_z": "(recent_median-baseline_median)/(IQR/1.349); undefined_if_IQR_zero"},
        "delta_initialization": "zero; past masks only", "feature_names": {name: value["tabular_names"] for name, value in designs.items()},
        "sources_sha256": {name: sha256(dataset/name) for name in ("split.json", "split_manifest.parquet", "visits.parquet", f"features_{window}d.parquet", f"sequences_{window}d.npz", "daily.parquet")},
    }
    manifest = dataset/"data_manifest.json"
    if manifest.exists():
        provenance["data_manifest_sha256"] = sha256(manifest)
    source_frame = full.drop(columns="_first_observation").copy()
    source_extras = tab[[name for name in tab if name not in source_frame]].reset_index(drop=True)
    source_frame = pd.concat([source_frame.reset_index(drop=True), source_extras], axis=1)
    return MonitoringData(frame, designs, provenance, prognosis_frame, prognosis_designs, source_frame)
