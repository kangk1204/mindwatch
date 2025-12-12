import json
import math
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import load_config, get_data_paths

"""Utility helpers to extract voice-derived features from participant MP3 clips.

The module caches expensive audio processing results to speed up re-runs and
aligns clips to the closest survey wave before aggregating statistics.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG = load_config()
DATA_PATHS = get_data_paths(CONFIG)
VOICE_DIR = DATA_PATHS["voice"]

CACHE_DIR = Path("logs")
CACHE_PATH = CACHE_DIR / "voice_features.csv"
META_PATH = CACHE_DIR / "voice_features_meta.json"
MAX_CLIPS_PER_WAVE = 2
FEATURE_VERSION = 3
# Only attach voice clips that occur on/before the survey within this window
MAX_CLIP_LAG_DAYS = 7


@dataclass
class VoiceClip:
    participant_id: str
    clip_path: Path
    recorded_ts: pd.Timestamp


def _parse_clip_timestamp(filename: str) -> Optional[pd.Timestamp]:
    """Extract YYYYMMDD prefix from filenames like 20250428-....mp3."""
    prefix = filename.split("_", 1)[0]
    parts = prefix.split("-", 1)
    if not parts:
        return None
    date_str = parts[0]
    if len(date_str) != 8 or not date_str.isdigit():
        return None
    try:
        return pd.to_datetime(date_str, format="%Y%m%d")
    except ValueError:
        return None


def _collect_voice_clips() -> Tuple[List[VoiceClip], float]:
    """Gather all available MP3 clips and track the most recent modification time."""
    if not VOICE_DIR.exists():
        logger.warning(f"Voice directory not found: {VOICE_DIR}")
        return [], 0.0

    clips: List[VoiceClip] = []
    latest_mtime = 0.0
    for participant_dir in sorted(VOICE_DIR.iterdir()):
        if not participant_dir.is_dir():
            continue
        pid = participant_dir.name
        for entry in sorted(participant_dir.glob("*.mp3")):
            ts = _parse_clip_timestamp(entry.name)
            if ts is None:
                continue
            clips.append(VoiceClip(participant_id=pid, clip_path=entry, recorded_ts=ts))
            latest_mtime = max(latest_mtime, entry.stat().st_mtime)
    logger.info(f"Found {len(clips)} voice clips across {len(set(c.participant_id for c in clips))} participants.")
    return clips, latest_mtime


def _label_latest_mtime(label_dir: Path = DATA_PATHS["label"]) -> float:
    if not label_dir.exists():
        return 0.0
    latest = 0.0
    for entry in label_dir.glob("*.xlsx"):
        try:
            latest = max(latest, entry.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def _load_cached_features(voice_mtime: float, label_mtime: float) -> Optional[pd.DataFrame]:
    """Load cached feature table when the underlying data has not changed."""
    if not CACHE_PATH.exists() or not META_PATH.exists():
        return None
    try:
        with META_PATH.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception as e:
        logger.warning(f"Failed to load cache metadata: {e}")
        return None
    if (
        not math.isclose(meta.get("voice_mtime", -1.0), voice_mtime, rel_tol=0.0, abs_tol=1e-3)
        or not math.isclose(meta.get("label_mtime", -1.0), label_mtime, rel_tol=0.0, abs_tol=1e-3)
        or meta.get("feature_version") != FEATURE_VERSION
    ):
        logger.info("Cache stale or version mismatch. Recomputing features.")
        return None
    try:
        cached = pd.read_csv(CACHE_PATH)
    except Exception as e:
        logger.warning(f"Failed to read cache CSV: {e}")
        return None
    if {"ID", "survey_wave"}.issubset(cached.columns):
        cached = cached.set_index(["ID", "survey_wave"])
    logger.info("Loaded voice features from cache.")
    return cached.astype(np.float32)


def _store_cached_features(df: pd.DataFrame, voice_mtime: float, label_mtime: float) -> None:
    """Persist the aggregated feature table along with metadata for cache invalidation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(CACHE_PATH, index=False)
    meta = {
        "voice_mtime": voice_mtime,
        "label_mtime": label_mtime,
        "feature_version": FEATURE_VERSION,
    }
    META_PATH.write_text(json.dumps(meta), encoding="utf-8")
    logger.info(f"Cached voice features to {CACHE_PATH}")


def _import_librosa():
    """Lazy import librosa to avoid imposing audio dependencies on non-voice runs."""
    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "librosa is required for voice feature extraction. Install it with `pip install librosa`."
        ) from exc
    return librosa


def _extract_clip_features(
    clip: VoiceClip,
    target_sr: int = 8000,
    max_duration: float = 30.0,
) -> Dict[str, float]:
    """Extract per-clip statistics (spectral, MFCC, pitch, etc.) for a single audio file."""
    librosa = _import_librosa()
    try:
        y, sr = librosa.load(
            clip.clip_path.as_posix(),
            sr=target_sr,
            mono=True,
            duration=max_duration,
        )
    except Exception as e:
        logger.error(f"Failed to load audio file {clip.clip_path}: {e}")
        return {}

    if y.size == 0:
        logger.warning(f"Empty audio file: {clip.clip_path}")
        return {}

    features: Dict[str, float] = {}
    duration = float(len(y) / sr)
    features["duration_sec"] = duration

    try:
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spec_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spec_centroid_std"] = float(np.std(spectral_centroid))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spec_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spec_bandwidth_std"] = float(np.std(spectral_bandwidth))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spec_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spec_rolloff_std"] = float(np.std(spectral_rolloff))

        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features["spec_flatness_mean"] = float(np.mean(spectral_flatness))
        features["spec_flatness_std"] = float(np.std(spectral_flatness))

        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for idx, band in enumerate(spectral_contrast, start=1):
                features[f"spec_contrast{idx}_mean"] = float(np.mean(band))
                features[f"spec_contrast{idx}_std"] = float(np.std(band))
        except Exception:
            pass

        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for idx, chroma_band in enumerate(chroma, start=1):
                features[f"chroma{idx}_mean"] = float(np.mean(chroma_band))
                features[f"chroma{idx}_std"] = float(np.std(chroma_band))
        except Exception:
            pass

        try:
            y_harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
            for idx, component in enumerate(tonnetz, start=1):
                features[f"tonnetz{idx}_mean"] = float(np.mean(component))
                features[f"tonnetz{idx}_std"] = float(np.std(component))
        except Exception:
            pass

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for idx, coeff in enumerate(mfcc, start=1):
            features[f"mfcc{idx}_mean"] = float(np.mean(coeff))
            features[f"mfcc{idx}_std"] = float(np.std(coeff))

        try:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            for idx, coeff in enumerate(mfcc_delta, start=1):
                features[f"mfcc{idx}_delta_mean"] = float(np.mean(coeff))
                features[f"mfcc{idx}_delta_std"] = float(np.std(coeff))
            for idx, coeff in enumerate(mfcc_delta2, start=1):
                features[f"mfcc{idx}_delta2_mean"] = float(np.mean(coeff))
                features[f"mfcc{idx}_delta2_std"] = float(np.std(coeff))
        except Exception:
            pass

        try:
            pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            pitch = pitch[np.isfinite(pitch)]
            if pitch.size:
                features["pitch_mean"] = float(np.mean(pitch))
                features["pitch_std"] = float(np.std(pitch))
                features["pitch_median"] = float(np.median(pitch))
        except Exception:
            pass

        try:
            non_silent = librosa.effects.split(y, top_db=30)
            voiced_ratio = sum((end - start) for start, end in non_silent) / max(len(y), 1)
            features["voiced_ratio"] = float(voiced_ratio)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Error computing features for {clip.clip_path}: {e}")
        return {}

    return features


def _closest_survey_wave(
    clip_ts: pd.Timestamp,
    survey_times: Iterable[Tuple[int, pd.Timestamp]],
    max_gap: pd.Timedelta,
) -> Optional[int]:
    """
    Return the nearest survey wave where the survey occurs on/after the clip
    and within the allowed gap to avoid look-ahead leakage.
    """
    best_wave: Optional[int] = None
    best_diff: Optional[pd.Timedelta] = None
    for wave, ts in survey_times:
        if clip_ts > ts:
            continue
        diff = ts - clip_ts
        if diff > max_gap:
            continue
        if best_diff is None or diff < best_diff:
            best_wave = wave
            best_diff = diff
    return best_wave


def _process_single_clip(
    clip: VoiceClip,
    survey_map: Dict[str, List[Tuple[int, pd.Timestamp]]]
) -> Optional[Dict[str, object]]:
    """Helper to process a single clip in parallel."""
    if clip.participant_id not in survey_map:
        return None
    
    survey_times = survey_map[clip.participant_id]
    wave = _closest_survey_wave(
        clip.recorded_ts,
        survey_times,
        max_gap=pd.Timedelta(days=MAX_CLIP_LAG_DAYS),
    )
    
    if wave is None:
        return None
    
    try:
        features = _extract_clip_features(clip)
        if not features:
            return None
        
        row: Dict[str, object] = {
            "ID": clip.participant_id,
            "survey_wave": wave,
            "clip_timestamp": clip.recorded_ts,
            "clip_count": 1,
        }
        row.update(features)
        return row
    except Exception as exc:
        logger.error(f"Failed to process {clip.clip_path}: {exc}")
        return None


def build_voice_feature_table(label_df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated voice features indexed by (ID, survey_wave)."""
    clips, voice_mtime = _collect_voice_clips()
    if not clips:
        return pd.DataFrame()

    label_mtime = _label_latest_mtime()
    cached = _load_cached_features(voice_mtime, label_mtime)
    if cached is not None:
        return cached

    logger.info("Recomputing voice features (parallel)...")
    
    survey_times = (
        label_df[["ID", "survey_wave", "survey_timestamp"]]
        .dropna(subset=["survey_timestamp"])
        .drop_duplicates()
    )
    survey_times["survey_timestamp"] = pd.to_datetime(survey_times["survey_timestamp"])
    
    # Pre-build a map of ID -> List[(wave, timestamp)] for fast lookup in workers
    survey_map: Dict[str, List[Tuple[int, pd.Timestamp]]] = {}
    for pid, group in survey_times.groupby("ID"):
        survey_map[pid] = list(group[["survey_wave", "survey_timestamp"]].itertuples(index=False, name=None))

    # Parallel processing
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_process_single_clip)(clip, survey_map) for clip in clips
    )
    
    # Filter out None results and enforce clip limit per wave
    valid_records = [r for r in results if r is not None]
    
    # Sort by timestamp to prioritize processing order if needed, though we aggregate all.
    # We still need to respect MAX_CLIPS_PER_WAVE which is harder to do purely in parallel 
    # if we want to pick *specific* clips. 
    # Heuristic: Process all, then group and slice.
    
    if not valid_records:
        return pd.DataFrame()
        
    df_temp = pd.DataFrame(valid_records)
    
    # Apply MAX_CLIPS_PER_WAVE limit
    # Sort by clip_timestamp to keep earliest/latest or just consistent? 
    # Let's keep existing logic: we just took the first ones encountered. 
    # Better: sort by timestamp to be deterministic.
    df_temp = df_temp.sort_values("clip_timestamp")
    df_temp = df_temp.groupby(["ID", "survey_wave"]).head(MAX_CLIPS_PER_WAVE)

    # Aggregation
    feature_cols = [col for col in df_temp.columns if col not in {"ID", "survey_wave", "clip_timestamp"}]
    aggregations: Dict[str, List[str]] = {}
    for col in feature_cols:
        if col == "clip_count":
            aggregations[col] = ["sum"]
        else:
            aggregations[col] = ["mean", "std"]

    aggregated = (
        df_temp.groupby(["ID", "survey_wave"])
        .agg(aggregations)
        .sort_index()
    )
    aggregated.columns = ["_".join([col] + [stat]) for col, stat in aggregated.columns]
    aggregated = aggregated.rename(columns={"clip_count_sum": "voice_clip_count"})

    # Do not fill NaNs with 0.0 here; let downstream models/imputers handle missing data
    # aggregated = aggregated.fillna(0.0) 
    aggregated = aggregated.astype(np.float32)
    _store_cached_features(aggregated, voice_mtime, label_mtime)
    
    return aggregated
