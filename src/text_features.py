"""Build text-derived features from survey responses and cache the result.

We convert categorical answers into numerical representations (one-hot, frequency,
string length) so that the tabular models can exploit the survey context alongside
sensor and voice information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = Path("logs")
CACHE_PATH = CACHE_DIR / "text_features.csv"
META_PATH = CACHE_DIR / "text_features_meta.json"
FEATURE_VERSION = 2  # bump when feature selection rules change

LABEL_BLOCKLIST = (
    "phq9",
    "target_binary",
    "target_score",
)


def _label_latest_mtime(label_dir: Path = Path("00_label_data")) -> float:
    if not label_dir.exists():
        return 0.0
    latest = 0.0
    for entry in label_dir.glob("*.xlsx"):
        try:
            latest = max(latest, entry.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def _load_cached_features(label_mtime: float) -> pd.DataFrame | None:
    """Return cached feature table if source data and schema version match."""
    if not CACHE_PATH.exists() or not META_PATH.exists():
        return None
    try:
        with META_PATH.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception:
        return None
    if meta.get("label_mtime") != label_mtime or meta.get("feature_version") != FEATURE_VERSION:
        return None
    try:
        return pd.read_csv(CACHE_PATH)
    except Exception:
        return None


def _store_cached_features(df: pd.DataFrame, label_mtime: float) -> None:
    """Persist feature table to disk so subsequent runs skip recomputation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    META_PATH.write_text(
        json.dumps({"label_mtime": label_mtime, "feature_version": FEATURE_VERSION}),
        encoding="utf-8",
    )


def _parse_time_column(series: pd.Series) -> pd.Series:
    """Convert free-form time strings into minutes after midnight."""
    dt = pd.to_datetime(series, errors="coerce")
    minutes = (dt.dt.hour.fillna(0) * 60 + dt.dt.minute.fillna(0)).astype(np.float32)
    return minutes


def build_text_feature_table(label_df: pd.DataFrame) -> pd.DataFrame:
    if label_df.empty:
        return pd.DataFrame()

    label_mtime = _label_latest_mtime()
    cached = _load_cached_features(label_mtime)
    if cached is not None:
        return cached.set_index(["ID", "survey_wave"]).astype(np.float32)

    base = (
        label_df.drop_duplicates(subset=["ID", "survey_wave"])
        [["ID", "survey_wave"]]
        .copy()
    )
    base["survey_wave"] = base["survey_wave"].astype(int)

    full = (
        label_df.drop_duplicates(subset=["ID", "survey_wave"])
        .set_index(["ID", "survey_wave"])
    )

    feature_frames: List[pd.DataFrame] = []

    # Time-of-day features
    for col in ["bedtime", "usual_wake_time"]:
        if col in full.columns:
            minutes = _parse_time_column(full[col])
            feature_frames.append(
                minutes.to_frame(name=f"text_{col}_minutes")
            )

    # Mixed categorical features (low cardinality)
    candidate_cols: List[str] = []
    for col in full.columns:
        if col in {"survey_timestamp"}:
            continue
        series = full[col]
        if series.dtype == object:
            lower_name = col.lower()
            if any(keyword in lower_name for keyword in LABEL_BLOCKLIST):
                continue
            nunique = series.nunique(dropna=True)
            if 1 < nunique <= 12:
                candidate_cols.append(col)

    for col in candidate_cols:
        series = full[col].astype("string").fillna("미응답")
        dummies = pd.get_dummies(series, prefix=f"text_{col}", dtype=np.int8)
        feature_frames.append(dummies)
        lengths = series.astype(str).str.len().astype(np.float32)
        feature_frames.append(lengths.to_frame(name=f"text_{col}_strlen"))

    # Higher-cardinality columns -> frequency encoding + length
    for col in full.columns:
        if col in candidate_cols or full[col].dtype != object:
            continue
        series = full[col].astype("string").fillna("미응답")
        lower_name = col.lower()
        if any(keyword in lower_name for keyword in LABEL_BLOCKLIST):
            continue
        if series.nunique(dropna=False) <= 1:
            continue
        freq = series.value_counts(normalize=True)
        encoded = series.map(freq).fillna(0.0).astype(np.float32)
        feature_frames.append(encoded.to_frame(name=f"text_{col}_freq"))
        lengths = series.astype(str).str.len().astype(np.float32)
        feature_frames.append(lengths.to_frame(name=f"text_{col}_strlen"))

    if not feature_frames:
        return pd.DataFrame()

    features = pd.concat(feature_frames, axis=1).fillna(0.0)
    features = features.astype(np.float32)
    features = features.reset_index()
    merged = base.merge(features, on=["ID", "survey_wave"], how="left").fillna(0.0)
    _store_cached_features(merged, label_mtime)
    return merged.set_index(["ID", "survey_wave"]).astype(np.float32)
