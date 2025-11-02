import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def format_seconds(seconds: float) -> str:
    """Return human-readable formatting for elapsed seconds."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import xgboost as xgb

try:
    import lightgbm as lgb  # type: ignore
    warnings.filterwarnings(
        "ignore",
        message="No further splits with positive gain, best gain: -inf",
        category=UserWarning,
        module="lightgbm",
    )
    warnings.filterwarnings(
        "ignore",
        message="Stopped training because there are no more leaves that meet the split requirements",
        category=UserWarning,
        module="lightgbm",
    )
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    from catboost import CatBoostClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

from train_tft import (
    Config,
    SENSOR_FEATURES,
    STATIC_REALS,
    TIME_VARYING_REALS,
    build_model_dataframe,
    load_hourly_sensor_frames,
    load_label_frames,
)
from voice_features import build_voice_feature_table
from text_features import build_text_feature_table


@dataclass
class StrategyConfig:
    name: str
    model_type: str
    features: List[str]
    params: Dict
    transform: Optional[str] = None
    use_sample_weights: bool = False


def merge_feature_lists(columns: Iterable[str], *feature_groups: Iterable[str]) -> List[str]:
    column_set = set(columns)
    seen: set[str] = set()
    merged: List[str] = []
    for group in feature_groups:
        for feature in group:
            if feature in column_set and feature not in seen:
                merged.append(feature)
                seen.add(feature)
    return merged


def participant_stratified_split(
    dataset: pd.DataFrame,
    test_size: float,
    split_seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    entity_table = (
        dataset.reset_index()
        .drop_duplicates(subset=["ID", "survey_wave"])
        [["ID", "survey_wave", "target_binary"]]
    )
    participant_labels = entity_table.groupby("ID")["target_binary"].max()
    participant_ids = participant_labels.index.to_numpy()
    if len(participant_ids) < 2:
        raise ValueError("At least two participants are required to build a validation split.")

    stratify_labels: Optional[np.ndarray]
    if participant_labels.nunique() > 1:
        class_counts = participant_labels.value_counts()
        if (class_counts < 2).any():
            stratify_labels = None
        else:
            stratify_labels = participant_labels.values
    else:
        stratify_labels = None

    train_ids, val_ids = train_test_split(
        participant_ids,
        test_size=test_size,
        random_state=split_seed,
        stratify=stratify_labels,
    )
    train_entities = entity_table[entity_table["ID"].isin(train_ids)][["ID", "survey_wave"]]
    val_entities = entity_table[entity_table["ID"].isin(val_ids)][["ID", "survey_wave"]]
    train_idx = list(train_entities.itertuples(index=False, name=None))
    val_idx = list(val_entities.itertuples(index=False, name=None))
    return train_idx, val_idx


def build_feature_dataframe(
    history_hours: int,
    windows: Tuple[int, ...] = (6, 12, 24, 48, 72, 240),
) -> pd.DataFrame:
    cfg = Config(history_hours=history_hours, task="classification", use_class_weights=False)
    label_df = load_label_frames()
    sensor_frames = load_hourly_sensor_frames()
    model_df = build_model_dataframe(label_df, sensor_frames, cfg)

    model_df["target_binary"] = model_df["target_binary"].astype(float)
    aggregated_blocks = []
    for window in windows:
        subset = model_df[model_df["time_idx"] >= -window + 1]
        grouped = subset.groupby(["ID", "survey_wave"])[SENSOR_FEATURES].agg(
            ["mean", "std", "min", "max", "last"]
        )
        grouped.columns = [f"{feature}_{stat}_w{window}" for feature, stat in grouped.columns]
        aggregated_blocks.append(grouped)
    aggregated_df = pd.concat(aggregated_blocks, axis=1)

    final_df = model_df[model_df["time_idx"] == 0].copy().set_index(["ID", "survey_wave"])
    final_df = final_df.join(aggregated_df, how="left")

    timestamp_map = (
        label_df[["ID", "survey_wave", "survey_timestamp"]]
        .drop_duplicates()
        .set_index(["ID", "survey_wave"])
    )
    final_df = final_df.join(timestamp_map, how="left")

    if "survey_timestamp" in final_df.columns:
        final_df = (
            final_df.reset_index()
            .sort_values(["ID", "survey_timestamp"])
            .set_index(["ID", "survey_wave"])
        )
    else:
        final_df = final_df.sort_index(level=["ID", "survey_wave"])

    prev_label_columns = [
        "PHQ9_Score",
        "phq9_binary",
        "survey_timestamp",
        "GAD7_Score",
        "ISI_Score",
        "Loneliness_Score",
        "S_Score",
        "WHODAS_Score",
        "WHOQOL_Score",
        "PSS_Score",
        "DSM_Score",
        "literacy_Score",
    ]
    label_features = (
        label_df.sort_values(["ID", "survey_wave"]).set_index(["ID", "survey_wave"])[
            [col for col in prev_label_columns if col in label_df.columns]
        ]
    )
    prev_labels = label_features.groupby(level=0).shift(1)
    prev_labels = prev_labels.rename(columns={col: f"prev_{col}" for col in prev_labels.columns})
    final_df = final_df.join(prev_labels, how="left")
    if "prev_survey_timestamp" in final_df.columns and "survey_timestamp" in final_df.columns:
        final_df["hours_since_prev_survey"] = (
            (final_df["survey_timestamp"] - final_df["prev_survey_timestamp"])
            .dt.total_seconds()
            .div(3600.0)
        )
        final_df = final_df.drop(columns=["prev_survey_timestamp"])
    prev_numeric_cols = [
        col
        for col in final_df.columns
        if col.startswith("prev_") and col not in {"prev_survey_timestamp"}
    ]
    for col in prev_numeric_cols:
        final_df[col] = final_df[col].astype(float).fillna(0.0)
    if "hours_since_prev_survey" in final_df.columns:
        final_df["hours_since_prev_survey"] = final_df["hours_since_prev_survey"].fillna(0.0)

    ratio_values = {}
    delta_values = {}
    zscore_values = {}
    range_values = {}
    for window in windows:
        mean_cols = [col for col in aggregated_df.columns if col.endswith(f"_mean_w{window}")]
        for col in mean_cols:
            base = col.replace(f"_mean_w{window}", "")
            if base in final_df.columns:
                denom = final_df[col].replace(0.0, np.nan)
                ratio_values[f"{base}_ratio_w{window}"] = (
                    final_df[base] / denom
                ).replace([np.inf, -np.inf], np.nan)
                delta_values[f"{base}_delta_mean_w{window}"] = final_df[base] - final_df[col]
                std_col = col.replace("_mean_", "_std_")
                if std_col in final_df.columns:
                    std_denom = final_df[std_col].replace(0.0, np.nan)
                    zscore_values[f"{base}_zscore_w{window}"] = (
                        (final_df[base] - final_df[col]) / std_denom
                    ).replace([np.inf, -np.inf], np.nan)
                max_col = col.replace("_mean_", "_max_")
                min_col = col.replace("_mean_", "_min_")
                if max_col in final_df.columns and min_col in final_df.columns:
                    range_values[f"{base}_range_w{window}"] = final_df[max_col] - final_df[min_col]

    if ratio_values:
        ratio_df = pd.DataFrame(ratio_values).fillna(0.0)
        final_df = pd.concat([final_df, ratio_df], axis=1)
    if delta_values:
        delta_df = pd.DataFrame(delta_values).fillna(0.0)
        final_df = pd.concat([final_df, delta_df], axis=1)
    if zscore_values:
        zscore_df = pd.DataFrame(zscore_values).fillna(0.0)
        final_df = pd.concat([final_df, zscore_df], axis=1)
    if range_values:
        range_df = pd.DataFrame(range_values).fillna(0.0)
        final_df = pd.concat([final_df, range_df], axis=1)

    for feature in ["steps", "screen_time", "proximity"]:
        if feature in final_df.columns:
            final_df[f"{feature}_log1p"] = np.log1p(final_df[feature].clip(lower=0))

    ewm_frames = []
    ewm_spans = (6, 12, 24, 72)
    sorted_df = model_df.sort_values(["ID", "survey_wave", "time_idx"])
    for span in ewm_spans:
        subset = sorted_df[sorted_df["time_idx"] >= -span + 1]
        if subset.empty:
            continue
        grouped = subset.groupby(["ID", "survey_wave"], group_keys=False)
        try:
            ewm_stats = grouped.apply(
                lambda g: g[SENSOR_FEATURES].ewm(span=span, adjust=False).mean().iloc[-1],
                include_groups=False,
            )
        except TypeError:
            ewm_stats = grouped.apply(
                lambda g: g[SENSOR_FEATURES].ewm(span=span, adjust=False).mean().iloc[-1]
            )
        ewm_stats.columns = [f"{col}_ewm_span{span}" for col in ewm_stats.columns]
        ewm_frames.append(ewm_stats)
    if ewm_frames:
        ewm_df = pd.concat(ewm_frames, axis=1)
        final_df = final_df.join(ewm_df, how="left")

    trend_frames = []
    trend_windows = (12, 24, 72, 168)
    for window in trend_windows:
        subset = sorted_df[sorted_df["time_idx"] >= -window + 1]
        if subset.empty:
            continue
        agg = (
            subset.groupby(["ID", "survey_wave"])[SENSOR_FEATURES]
            .agg(["first", "last"])
        )
        first_vals = agg.xs("first", axis=1, level=1)
        last_vals = agg.xs("last", axis=1, level=1)
        trend = (last_vals - first_vals) / max(window - 1, 1)
        trend.columns = [f"{col}_trend_w{window}" for col in trend.columns]
        trend_frames.append(trend)
    if trend_frames:
        trend_df = pd.concat(trend_frames, axis=1)
        final_df = final_df.join(trend_df, how="left")

    if "Sex" in final_df.columns:
        final_df["Sex"] = final_df["Sex"].cat.codes.astype(np.int16)
    if "survey_wave" in final_df.index.names:
        survey_wave_vals = final_df.index.get_level_values("survey_wave").astype(int)
        final_df["survey_wave_num"] = survey_wave_vals
        final_df["is_wave2"] = (survey_wave_vals == 2).astype(int)

    cross_wave_base = [
        "ema_depression",
        "ema_anxiety",
        "ema_sleep",
        "ema_stress",
        "steps",
        "screen_time",
        "heart_rate",
        "proximity",
    ]
    cross_wave_cols = [col for col in cross_wave_base if col in final_df.columns]
    if cross_wave_cols:
        prev_wave_values = final_df[cross_wave_cols].groupby(level=0).shift(1)
        prev_wave_values = prev_wave_values.add_suffix("_prev_wave")
        final_df = pd.concat([final_df, prev_wave_values], axis=1)
        for col in cross_wave_cols:
            prev_col = f"{col}_prev_wave"
            if prev_col in final_df.columns:
                final_df[f"{col}_delta_prev_wave"] = final_df[col] - final_df[prev_col]
        prev_wave_cols = [f"{col}_prev_wave" for col in cross_wave_cols]
        final_df[prev_wave_cols] = final_df[prev_wave_cols].fillna(0.0)
        delta_cols = [f"{col}_delta_prev_wave" for col in cross_wave_cols]
        final_df[delta_cols] = final_df[delta_cols].fillna(0.0)

    # Attach cached voice-derived statistics (MFCC, pitch, etc.).
    voice_features = build_voice_feature_table(label_df)
    if not voice_features.empty:
        final_df = final_df.join(voice_features, how="left")

    # Attach text/survey category encodings.
    text_features = build_text_feature_table(label_df)
    if not text_features.empty:
        final_df = final_df.join(text_features, how="left")

    # Ensure voice/text vectors do not introduce NaNs later in the pipeline.
    voice_cols = [
        col
        for col in final_df.columns
        if col.startswith("voice_")
        or col.startswith("mfcc")
        or col.startswith("spec_")
        or col.startswith("chroma")
        or col.startswith("tonnetz")
        or col.startswith("pitch")
    ]
    if voice_cols:
        final_df[voice_cols] = final_df[voice_cols].fillna(0.0)

    text_cols = [col for col in final_df.columns if col.startswith("text_")]
    if text_cols:
        final_df[text_cols] = final_df[text_cols].fillna(0.0)

    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df[final_df["target_binary"].notna()]
    final_df["target_binary"] = final_df["target_binary"].astype(int)
    return final_df
def prepare_feature_context(
    dataset: pd.DataFrame,
    split_seed: int = 42,
    exclude_prev_survey: bool = False,
    top_k_features: int = 120,
    top_k_min: Optional[int] = None,
) -> Dict[str, object]:
    top_k_features = max(1, int(top_k_features))
    if exclude_prev_survey:
        drop_cols = [
            col
            for col in dataset.columns
            if col.startswith("prev_")
            or col.endswith("_prev_wave")
            or col.endswith("_delta_prev_wave")
        ]
        dataset = dataset.drop(columns=drop_cols, errors="ignore")

    columns = dataset.columns

    static_cols = [col for col in STATIC_REALS if col in columns]
    time_cols = [col for col in TIME_VARYING_REALS if col in columns]
    # Keep track of voice/text feature namespaces so specialised strategies can use them.
    voice_cols = [
        col
        for col in columns
        if col.startswith("voice_")
        or col.startswith("mfcc")
        or col.startswith("spec_")
        or col.startswith("duration_")
        or col.startswith("rms_")
        or col.startswith("zcr_")
        or col.startswith("chroma")
        or col.startswith("tonnetz")
        or col.startswith("pitch")
        or col.startswith("voiced_")
    ]
    text_cols = [col for col in columns if col.startswith("text_")]
    sensor_prefixes = tuple(f"{feature}_" for feature in SENSOR_FEATURES)
    sensor_cols = [
        col
        for col in columns
        if col not in voice_cols
        and not col.startswith("text_")
        and (
            col in SENSOR_FEATURES
            or any(col.startswith(prefix) for prefix in sensor_prefixes)
        )
    ]
    extra_base_features = [
        col
        for col in [
            "prev_PHQ9_Score",
            "prev_phq9_binary",
            "hours_since_prev_survey",
            "survey_wave_num",
            "is_wave2",
        ]
        if col in columns
    ]
    cross_wave_prev_cols = [col for col in columns if col.endswith("_prev_wave")]
    cross_wave_delta_cols = [col for col in columns if col.endswith("_delta_prev_wave")]
    feature_base = merge_feature_lists(
        columns,
        static_cols,
        time_cols,
        extra_base_features,
        cross_wave_prev_cols,
        cross_wave_delta_cols,
        voice_cols,
        text_cols,
    )

    windows = [6, 12, 24, 48, 72, 240]
    agg_cols_by_window = {
        window: [col for col in columns if col.endswith(f"w{window}")]
        for window in windows
    }
    agg_mean_std_w24 = [
        col for col in columns if col.endswith("_mean_w24") or col.endswith("_std_w24")
    ]
    agg_mean_std_w72 = [
        col for col in columns if col.endswith("_mean_w72") or col.endswith("_std_w72")
    ]
    agg_last_w12 = [col for col in columns if col.endswith("_last_w12")]
    agg_cols_all = merge_feature_lists(columns, *agg_cols_by_window.values())

    ratio_cols = [col for col in columns if "_ratio_w" in col]
    delta_cols = [col for col in columns if "_delta_mean_w" in col]
    log_cols = [col for col in columns if col.endswith("_log1p")]
    zscore_cols = [col for col in columns if "_zscore_w" in col]
    range_cols = [col for col in columns if "_range_w" in col]
    ewm_cols = [col for col in columns if "_ewm_span" in col]
    trend_cols = [col for col in columns if "_trend_w" in col]
    all_features = merge_feature_lists(
        columns,
        feature_base,
        agg_cols_all,
        ratio_cols,
        delta_cols,
        log_cols,
        zscore_cols,
        range_cols,
        ewm_cols,
        trend_cols,
        voice_cols,
        text_cols,
    )

    train_idx, val_idx = participant_stratified_split(
        dataset,
        test_size=0.2,
        split_seed=split_seed,
    )

    base_importance_strategy = StrategyConfig(
        name="HGB_importance",
        model_type="hgb",
        features=feature_base,
        params=dict(
            loss="log_loss",
            learning_rate=0.05,
            max_leaf_nodes=63,
            max_depth=6,
            max_iter=150,
            class_weight="balanced",
            random_state=42,
        ),
        transform=None,
        use_sample_weights=False,
    )
    importances_series = compute_feature_importance(
        dataset=dataset,
        features=feature_base,
        strategy=base_importance_strategy,
        train_index=train_idx,
    )
    top_feature_pool = importances_series.index.tolist()
    top_feature_names = top_feature_pool[:top_k_features]
    if not top_feature_names:
        top_feature_names = top_feature_pool
    top_k_default = len(top_feature_names)
    top_k_max = len(top_feature_pool)
    if top_k_min is None:
        top_k_min = min(80, top_k_default) if top_k_default >= 80 else max(1, top_k_default // 2 or 1)
    top_k_min = max(1, min(top_k_min, top_k_max))
    top_k_default = max(top_k_min, top_k_default)

    return dict(
        columns=list(columns),
        feature_base=feature_base,
        agg_mean_std_w24=agg_mean_std_w24,
        agg_mean_std_w72=agg_mean_std_w72,
        agg_last_w12=agg_last_w12,
        agg_cols_all=agg_cols_all,
        ratio_cols=ratio_cols,
        delta_cols=delta_cols,
        log_cols=log_cols,
        zscore_cols=zscore_cols,
        range_cols=range_cols,
        ewm_cols=ewm_cols,
        trend_cols=trend_cols,
        all_features=all_features,
        voice_cols=voice_cols,
        text_cols=text_cols,
        sensor_cols=sensor_cols,
        top_feature_names=top_feature_names,
        top_feature_pool=top_feature_pool,
        top_k_default=top_k_default,
        top_k_min=top_k_min,
        top_k_max=top_k_max,
        train_idx=train_idx,
        val_idx=val_idx,
        base_importance_strategy=base_importance_strategy,
        exclude_prev_survey=exclude_prev_survey,
    )


def compute_feature_importance(
    dataset: pd.DataFrame,
    features: List[str],
    strategy: StrategyConfig,
    train_index: List[Tuple[str, int]],
) -> pd.Series:
    feats = [f for f in features if f in dataset.columns]
    if strategy.model_type != "hgb":
        raise ValueError("Feature importance is only supported for HGB strategy")
    X = dataset[feats]
    y = dataset["target_binary"]
    X_train = X.loc[train_index].copy()
    y_train = y.loc[train_index]
    median = X_train.median()
    X_train = X_train.fillna(median)
    sample_weight = compute_sample_weights(y_train) if strategy.use_sample_weights else None
    model = HistGradientBoostingClassifier(**strategy.params)
    model.fit(X_train.values, y_train.values, sample_weight=sample_weight)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        perm = permutation_importance(
            model, X_train.values, y_train.values, n_repeats=5, random_state=42, scoring="roc_auc"
        )
        importances = perm.importances_mean
    return pd.Series(importances, index=feats).sort_values(ascending=False)


def evaluate_block_holdout(
    dataset: pd.DataFrame,
    strategy: StrategyConfig,
    features: List[str],
    quantile: float = 0.8,
) -> Dict[str, float]:
    if "survey_timestamp" not in dataset.columns:
        raise ValueError("survey_timestamp column missing for block validation")
    timestamp_df = (
        dataset[["target_binary", "survey_timestamp"]]
        .reset_index()
        .dropna(subset=["survey_timestamp"])
        .drop_duplicates()
        .sort_values("survey_timestamp")
    )
    participant_last_ts = (
        timestamp_df.groupby("ID")["survey_timestamp"].max().sort_values()
    )
    cutoff = int(len(participant_last_ts) * quantile)
    if cutoff == 0 or cutoff >= len(participant_last_ts):
        raise ValueError("Invalid block split; adjust quantile or ensure timestamps exist")
    train_ids = participant_last_ts.index[:cutoff]
    val_ids = participant_last_ts.index[cutoff:]
    train_blocks = timestamp_df[timestamp_df["ID"].isin(train_ids)][["ID", "survey_wave"]]
    val_blocks = timestamp_df[timestamp_df["ID"].isin(val_ids)][["ID", "survey_wave"]]
    train_idx = list(zip(train_blocks["ID"], train_blocks["survey_wave"]))
    val_idx = list(zip(val_blocks["ID"], val_blocks["survey_wave"]))
    metrics, _ = evaluate_strategy(
        dataset=dataset,
        features=features,
        strategy=strategy,
        train_index=train_idx,
        val_index=val_idx,
        cv_splits=3,
    )
    return metrics


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    total = len(y)
    weight_pos = total / (2 * pos) if pos > 0 else 1.0
    weight_neg = total / (2 * neg) if neg > 0 else 1.0
    return np.where(y.values == 1, weight_pos, weight_neg)


def evaluate_strategy(
    dataset: pd.DataFrame,
    features: List[str],
    strategy: StrategyConfig,
    train_index: List[Tuple[str, int]],
    val_index: List[Tuple[str, int]],
    cv_splits: int = 5,
    return_predictions: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, pd.Series]]]:
    X = dataset.drop(columns=["target_binary"])
    y = dataset["target_binary"]
    feats = [f for f in features if f in X.columns]
    if not feats:
        raise ValueError(f"No valid features found for strategy {strategy.name}")

    train_mi = pd.MultiIndex.from_tuples(train_index, names=dataset.index.names)
    val_mi = pd.MultiIndex.from_tuples(val_index, names=dataset.index.names)
    X_train_full = X.loc[train_mi, feats].copy()
    X_val = X.loc[val_mi, feats].copy()
    y_train_full = y.loc[train_mi]
    y_val = y.loc[val_mi]

    median = X_train_full.median()
    X_train = X_train_full.fillna(median)
    X_val = X_val.fillna(median)

    train_weights = None
    if strategy.use_sample_weights:
        train_weights = compute_sample_weights(y_train_full)

    if strategy.transform == "quantile":
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(X_train)),
            subsample=100000,
            random_state=42,
        )
        X_train_trans = qt.fit_transform(X_train)
        X_val_trans = qt.transform(X_val)
    else:
        X_train_trans = X_train.values
        X_val_trans = X_val.values

    if strategy.model_type == "hgb":
        model = HistGradientBoostingClassifier(**strategy.params)
        model.fit(X_train_trans, y_train_full.values, sample_weight=train_weights)
        val_pred = model.predict_proba(X_val_trans)[:, 1]
    elif strategy.model_type == "xgb":
        model = xgb.XGBClassifier(**strategy.params)
        model.fit(X_train_trans, y_train_full.values, sample_weight=train_weights)
        val_pred = model.predict_proba(X_val_trans)[:, 1]
    elif strategy.model_type == "lgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed. Install it with `pip install lightgbm`.")
        model = lgb.LGBMClassifier(**strategy.params)
        model.fit(X_train_trans, y_train_full.values, sample_weight=train_weights)
        val_pred = model.predict_proba(X_val_trans)[:, 1]
    elif strategy.model_type == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Install it with `pip install catboost`.")
        model = CatBoostClassifier(**strategy.params)
        model.fit(
            X_train_trans,
            y_train_full.values,
            sample_weight=train_weights,
            verbose=False,
        )
        val_pred = model.predict_proba(X_val_trans)[:, 1]
    else:
        raise ValueError(f"Unsupported model type: {strategy.model_type}")

    holdout_auc = roc_auc_score(y_val.values, val_pred)

    groups_train = X_train_full.index.get_level_values(0)
    oof_preds = np.zeros(len(X_train_full))
    cv_scores: List[float] = []
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(
        cv.split(X_train_full, y_train_full, groups_train)
    ):
        X_fold_train = X_train_full.iloc[train_fold_idx]
        X_fold_val = X_train_full.iloc[val_fold_idx]
        y_fold_train = y_train_full.iloc[train_fold_idx]
        y_fold_val = y_train_full.iloc[val_fold_idx]

        fold_median = X_fold_train.median()
        X_fold_train = X_fold_train.fillna(fold_median)
        X_fold_val = X_fold_val.fillna(fold_median)

        fold_weights = compute_sample_weights(y_fold_train) if strategy.use_sample_weights else None

        if strategy.transform == "quantile":
            qt_fold = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, len(X_fold_train)),
                subsample=100000,
                random_state=42,
            )
            X_fold_train_trans = qt_fold.fit_transform(X_fold_train)
            X_fold_val_trans = qt_fold.transform(X_fold_val)
        else:
            X_fold_train_trans = X_fold_train.values
            X_fold_val_trans = X_fold_val.values

        if strategy.model_type == "hgb":
            model_cv = HistGradientBoostingClassifier(**strategy.params)
            model_cv.fit(X_fold_train_trans, y_fold_train.values, sample_weight=fold_weights)
            val_pred_fold = model_cv.predict_proba(X_fold_val_trans)[:, 1]
        elif strategy.model_type == "xgb":
            model_cv = xgb.XGBClassifier(**strategy.params)
            model_cv.fit(X_fold_train_trans, y_fold_train.values, sample_weight=fold_weights)
            val_pred_fold = model_cv.predict_proba(X_fold_val_trans)[:, 1]
        elif strategy.model_type == "lgbm":
            if lgb is None:
                raise ImportError("lightgbm is not installed. Install it with `pip install lightgbm`.")
            model_cv = lgb.LGBMClassifier(**strategy.params)
            model_cv.fit(X_fold_train_trans, y_fold_train.values, sample_weight=fold_weights)
            val_pred_fold = model_cv.predict_proba(X_fold_val_trans)[:, 1]
        elif strategy.model_type == "catboost":
            if CatBoostClassifier is None:
                raise ImportError("catboost is not installed. Install it with `pip install catboost`.")
            model_cv = CatBoostClassifier(**strategy.params)
            model_cv.fit(
                X_fold_train_trans,
                y_fold_train.values,
                sample_weight=fold_weights,
                verbose=False,
            )
            val_pred_fold = model_cv.predict_proba(X_fold_val_trans)[:, 1]
        cv_score = roc_auc_score(y_fold_val.values, val_pred_fold)
        cv_scores.append(cv_score)
        oof_preds[val_fold_idx] = val_pred_fold

    metrics = {
        "holdout_auc": holdout_auc,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
    }

    prediction_payload = None
    if return_predictions:
        prediction_payload = {
            "oof": pd.Series(oof_preds, index=X_train_full.index, name=strategy.name),
            "holdout": pd.Series(val_pred, index=X_val.index, name=strategy.name),
            "y_train": y_train_full.copy(),
            "y_holdout": y_val.copy(),
        }

    return metrics, prediction_payload


def build_default_strategies(context: Dict[str, object]) -> List[StrategyConfig]:
    columns = context["columns"]  # type: ignore[index]
    feature_base = context["feature_base"]  # type: ignore[index]
    agg_mean_std_w24 = context["agg_mean_std_w24"]  # type: ignore[index]
    agg_mean_std_w72 = context["agg_mean_std_w72"]  # type: ignore[index]
    agg_last_w12 = context["agg_last_w12"]  # type: ignore[index]
    agg_cols_all = context["agg_cols_all"]  # type: ignore[index]
    ratio_cols = context["ratio_cols"]  # type: ignore[index]
    log_cols = context["log_cols"]  # type: ignore[index]
    zscore_cols = context["zscore_cols"]  # type: ignore[index]
    range_cols = context["range_cols"]  # type: ignore[index]
    ewm_cols = context["ewm_cols"]  # type: ignore[index]
    trend_cols = context["trend_cols"]  # type: ignore[index]
    all_features = context["all_features"]  # type: ignore[index]
    top_feature_pool = context["top_feature_pool"]  # type: ignore[index]
    voice_cols = context.get("voice_cols", [])  # type: ignore[assignment]
    text_cols = context.get("text_cols", [])  # type: ignore[assignment]
    sensor_cols = context.get("sensor_cols", [])  # type: ignore[assignment]
    top_k_default = context.get("top_k_default", len(top_feature_pool))  # type: ignore[arg-type]
    top_feature_names = top_feature_pool[:top_k_default]
    hgb_topk = top_feature_pool[: min(len(top_feature_pool), 94)]
    lgbm_topk = top_feature_pool[: min(len(top_feature_pool), 42)]
    xgb_topk = top_feature_pool[: min(len(top_feature_pool), 44)]
    base_importance_strategy = context["base_importance_strategy"]  # type: ignore[index]

    def merge_features(*groups: List[str]) -> List[str]:
        return merge_feature_lists(columns, *groups)

    strategies = [
        StrategyConfig(
            name="HGB_v1_base",
            model_type="hgb",
            features=merge_features(feature_base),
            params=dict(
                loss="log_loss",
                learning_rate=0.05,
                max_leaf_nodes=63,
                max_depth=6,
                max_iter=180,
                class_weight="balanced",
                random_state=42,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v2_roll",
            model_type="hgb",
            features=merge_features(feature_base, agg_mean_std_w24, agg_mean_std_w72),
            params=dict(
                loss="log_loss",
                learning_rate=0.032,
                max_leaf_nodes=127,
                max_depth=None,
                max_iter=420,
                l2_regularization=0.12,
                min_samples_leaf=18,
                class_weight="balanced",
                random_state=42,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v3_quantile",
            model_type="hgb",
            features=merge_features(feature_base, agg_mean_std_w24, agg_last_w12, log_cols),
            params=dict(
                loss="log_loss",
                learning_rate=0.02,
                max_leaf_nodes=255,
                max_depth=None,
                max_iter=650,
                l2_regularization=0.22,
                min_samples_leaf=15,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v4_ratio",
            model_type="hgb",
            features=merge_features(feature_base, agg_mean_std_w24, ratio_cols, zscore_cols),
            params=dict(
                loss="log_loss",
                learning_rate=0.028,
                max_leaf_nodes=191,
                max_depth=None,
                max_iter=520,
                l2_regularization=0.08,
                min_samples_leaf=22,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=True,
        ),
        StrategyConfig(
            name="HGB_top120",
            model_type="hgb",
            features=hgb_topk,
            params=dict(base_importance_strategy.params),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_optuna_best",
            model_type="hgb",
            features=hgb_topk,
            params=dict(
                loss="log_loss",
                learning_rate=0.011007694002855726,
                max_leaf_nodes=487,
                max_depth=3,
                max_iter=200,
                l2_regularization=0.010298368318423639,
                min_samples_leaf=34,
                max_bins=184,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="XGB_v5_full",
            model_type="xgb",
            features=merge_features(xgb_topk),
            params=dict(
                objective="binary:logistic",
                eval_metric="auc",
                learning_rate=0.0053343091361369305,
                max_depth=6,
                subsample=0.5543190206315183,
                colsample_bytree=0.4209596000140664,
                colsample_bylevel=0.4209596000140664,
                min_child_weight=8.643553072189608,
                n_estimators=503,
                reg_lambda=2.222603538440007,
                reg_alpha=0.5765904707390151,
                scale_pos_weight=1.0723021386057379,
                random_state=42,
                tree_method="hist",
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v6_ewm_trend",
            model_type="hgb",
            features=merge_features(
                feature_base,
                agg_mean_std_w24,
                agg_mean_std_w72,
                ewm_cols,
                trend_cols,
                zscore_cols,
                range_cols,
            ),
            params=dict(
                loss="log_loss",
                learning_rate=0.024,
                max_leaf_nodes=255,
                max_depth=None,
                max_iter=700,
                l2_regularization=0.18,
                min_samples_leaf=18,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=True,
        ),
        StrategyConfig(
            name="LGBM_v7_full",
            model_type="lgbm",
            features=merge_features(lgbm_topk),
            params=dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                learning_rate=0.011024790465649671,
                n_estimators=404,
                num_leaves=244,
                max_depth=-1,
                subsample=0.6134946108429382,
                colsample_bytree=0.7176925208164471,
                min_child_samples=117,
                reg_alpha=0.04438903330917268,
                reg_lambda=0.0008534870496295665,
                random_state=42,
                n_jobs=-1,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="LGBM_v8_top120",
            model_type="lgbm",
            features=merge_features(lgbm_topk, ewm_cols, trend_cols, log_cols),
            params=dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                learning_rate=0.011024790465649671,
                n_estimators=404,
                num_leaves=244,
                max_depth=-1,
                subsample=0.6134946108429382,
                colsample_bytree=0.7176925208164471,
                min_child_samples=117,
                reg_alpha=0.04438903330917268,
                reg_lambda=0.0008534870496295665,
                random_state=42,
                n_jobs=-1,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="CatBoost_v9_full",
            model_type="catboost",
            features=merge_features(
                feature_base,
                agg_cols_all,
                ratio_cols,
                zscore_cols,
                range_cols,
                ewm_cols,
                trend_cols,
                log_cols,
            ),
            params=dict(
                loss_function="Logloss",
                eval_metric="AUC",
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=4.0,
                iterations=1600,
                random_seed=42,
                grow_policy="SymmetricTree",
                bagging_temperature=0.6,
                allow_writing_files=False,
            ),
            transform=None,
            use_sample_weights=True,
        ),
    ]
    if voice_cols or text_cols:
        strategies.append(
            StrategyConfig(
                name="HGB_voice_text",
                model_type="hgb",
                features=merge_features(feature_base, voice_cols, text_cols),
                params=dict(
                    loss="log_loss",
                    learning_rate=0.03,
                    max_leaf_nodes=255,
                    max_depth=None,
                    max_iter=520,
                    l2_regularization=0.12,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                ),
                transform="quantile",
                use_sample_weights=False,
            )
        )
    if voice_cols:
        strategies.append(
            StrategyConfig(
                name="Voice_HGB",
                model_type="hgb",
                features=merge_features(voice_cols),
                params=dict(
                    loss="log_loss",
                    learning_rate=0.08,
                    max_leaf_nodes=127,
                    max_depth=4,
                    max_iter=220,
                    l2_regularization=0.12,
                    min_samples_leaf=12,
                    class_weight="balanced",
                    random_state=42,
                ),
                transform="quantile",
                use_sample_weights=False,
            )
        )
    if text_cols:
        strategies.append(
            StrategyConfig(
                name="Text_HGB",
                model_type="hgb",
                features=merge_features(text_cols),
                params=dict(
                    loss="log_loss",
                    learning_rate=0.06,
                    max_leaf_nodes=63,
                    max_depth=4,
                    max_iter=260,
                    class_weight="balanced",
                    random_state=42,
                ),
                transform="quantile",
                use_sample_weights=False,
            )
        )
    if sensor_cols:
        strategies.append(
            StrategyConfig(
                name="HGB_sensor_only",
                model_type="hgb",
                features=merge_features(sensor_cols),
                params=dict(
                    loss="log_loss",
                    learning_rate=0.05,
                    max_leaf_nodes=191,
                    max_depth=6,
                    max_iter=320,
                    class_weight="balanced",
                    random_state=42,
                ),
                transform=None,
                use_sample_weights=False,
            )
        )
    return strategies


def _extract_group_labels(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(0).to_numpy()
    return index.to_numpy()


def _tune_logistic_meta(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_cs: Optional[List[float]] = None,
) -> Tuple[StandardScaler, LogisticRegression, float, float]:
    if candidate_cs is None:
        candidate_cs = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    unique_groups = np.unique(groups)
    n_splits = max(2, min(5, len(unique_groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = -np.inf
    best_c = candidate_cs[0]
    for C in candidate_cs:
        fold_scores: List[float] = []
        for train_idx, val_idx in cv.split(X, y, groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            model = LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(X_train, y[train_idx])
            preds = model.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y[val_idx], preds))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_c = C
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_scaled, y)
    return scaler, model, best_c, best_score


def _tune_xgb_meta(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[xgb.XGBClassifier, Dict[str, float], float]:
    param_grid = [
        dict(learning_rate=lr, max_depth=md, n_estimators=ne, subsample=sub)
        for lr in (0.02, 0.05, 0.1)
        for md in (2, 3)
        for ne in (200, 400)
        for sub in (0.7, 0.9)
    ]
    unique_groups = np.unique(groups)
    n_splits = max(2, min(5, len(unique_groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = -np.inf
    best_params = param_grid[0]
    for params in param_grid:
        fold_scores: List[float] = []
        for train_idx, val_idx in cv.split(X, y, groups):
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                tree_method="hist",
                n_jobs=1,
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                subsample=params["subsample"],
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.05,
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[val_idx])[:, 1]
            fold_scores.append(roc_auc_score(y[val_idx], preds))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        n_jobs=1,
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        subsample=best_params["subsample"],
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.05,
    )
    final_model.fit(X, y)
    return final_model, best_params, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular models on engineered features.")
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=120,
        help="Number of top ranked features (by HGB importance) to keep for top-k strategies.",
    )
    parser.add_argument(
        "--top-k-min",
        type=int,
        default=None,
        help="Minimum feature count considered when strategies dynamically tune their top-k subset.",
    )
    args = parser.parse_args()

    overall_start = time.time()

    step_start = time.time()
    dataset = build_feature_dataframe(history_hours=args.history_hours)
    print(f"[Timer] Feature dataframe built in {time.time() - step_start:.1f}s")

    step_start = time.time()
    context = prepare_feature_context(
        dataset,
        split_seed=42,
        top_k_features=args.top_k_features,
        top_k_min=args.top_k_min,
    )
    print(f"[Timer] Feature context prepared in {time.time() - step_start:.1f}s")

    feature_base = context["feature_base"]
    agg_mean_std_w24 = context["agg_mean_std_w24"]
    agg_mean_std_w72 = context["agg_mean_std_w72"]
    agg_last_w12 = context["agg_last_w12"]
    agg_cols_all = context["agg_cols_all"]
    ratio_cols = context["ratio_cols"]
    log_cols = context["log_cols"]
    zscore_cols = context["zscore_cols"]
    range_cols = context["range_cols"]
    ewm_cols = context["ewm_cols"]
    trend_cols = context["trend_cols"]
    all_features = context["all_features"]
    top_feature_names = context["top_feature_names"]
    train_idx = context["train_idx"]
    val_idx = context["val_idx"]
    base_importance_strategy = context["base_importance_strategy"]
    voice_cols = context.get("voice_cols", [])
    text_cols = context.get("text_cols", [])

    def merge_features(*feature_groups: List[str]) -> List[str]:
        return merge_feature_lists(dataset.columns, *feature_groups)

    strategies = [
        StrategyConfig(
            name="HGB_v1_base",
            model_type="hgb",
            features=merge_features(feature_base),
            params=dict(
                loss="log_loss",
                learning_rate=0.05,
                max_leaf_nodes=63,
                max_depth=6,
                max_iter=180,
                class_weight="balanced",
                random_state=42,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v2_roll",
            model_type="hgb",
            features=merge_features(feature_base, agg_mean_std_w24, agg_mean_std_w72),
            params=dict(
                loss="log_loss",
                learning_rate=0.032,
                max_leaf_nodes=127,
                max_depth=None,
                max_iter=420,
                l2_regularization=0.12,
                min_samples_leaf=18,
                class_weight="balanced",
                random_state=42,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v3_quantile",
            model_type="hgb",
            features=merge_features(feature_base, agg_mean_std_w24, agg_last_w12, log_cols),
            params=dict(
                loss="log_loss",
                learning_rate=0.02,
                max_leaf_nodes=255,
                max_depth=None,
                max_iter=650,
                l2_regularization=0.22,
                min_samples_leaf=15,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_v4_ratio",
            model_type="hgb",
            features=merge_features(feature_base, agg_mean_std_w24, ratio_cols, zscore_cols),
            params=dict(
                loss="log_loss",
                learning_rate=0.028,
                max_leaf_nodes=191,
                max_depth=None,
                max_iter=520,
                l2_regularization=0.08,
                min_samples_leaf=22,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=True,
        ),
        StrategyConfig(
            name="HGB_top120",
            model_type="hgb",
            features=top_feature_names,
            params=dict(base_importance_strategy.params),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="HGB_optuna_best",
            model_type="hgb",
            features=top_feature_names,
            params=dict(
                loss="log_loss",
                learning_rate=0.015320758630314239,
                max_leaf_nodes=407,
                max_depth=3,
                max_iter=150,
                l2_regularization=0.4348147480451949,
                min_samples_leaf=80,
                max_bins=64,
                class_weight="balanced",
                random_state=42,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="XGB_v5_full",
            model_type="xgb",
            features=all_features,
            params=dict(
                objective="binary:logistic",
                eval_metric="auc",
                learning_rate=0.03,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                min_child_weight=2,
                n_estimators=1200,
                reg_lambda=1.0,
                reg_alpha=0.05,
                scale_pos_weight=1.4,
                random_state=42,
                tree_method="hist",
            ),
            transform=None,
            use_sample_weights=True,
        ),
        StrategyConfig(
            name="HGB_v6_ewm_trend",
            model_type="hgb",
            features=merge_features(
                feature_base,
                agg_mean_std_w24,
                agg_mean_std_w72,
                ewm_cols,
                trend_cols,
                zscore_cols,
                range_cols,
            ),
            params=dict(
                loss="log_loss",
                learning_rate=0.024,
                max_leaf_nodes=255,
                max_depth=None,
                max_iter=700,
                l2_regularization=0.18,
                min_samples_leaf=18,
                class_weight="balanced",
                random_state=42,
            ),
            transform="quantile",
            use_sample_weights=True,
        ),
        StrategyConfig(
            name="LGBM_v7_full",
            model_type="lgbm",
            features=all_features,
            params=dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                learning_rate=0.02,
                n_estimators=1500,
                num_leaves=95,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_samples=24,
                reg_alpha=0.2,
                reg_lambda=0.4,
                random_state=42,
                n_jobs=-1,
            ),
            transform=None,
            use_sample_weights=True,
        ),
        StrategyConfig(
            name="LGBM_v8_top120",
            model_type="lgbm",
            features=merge_features(top_feature_names, ewm_cols, trend_cols, log_cols),
            params=dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                learning_rate=0.035,
                n_estimators=1100,
                num_leaves=63,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_samples=16,
                reg_alpha=0.1,
                reg_lambda=0.25,
                random_state=42,
                n_jobs=-1,
            ),
            transform=None,
            use_sample_weights=False,
        ),
        StrategyConfig(
            name="CatBoost_v9_full",
            model_type="catboost",
            features=merge_features(
                feature_base,
                agg_cols_all,
                ratio_cols,
                zscore_cols,
                range_cols,
                ewm_cols,
                trend_cols,
                log_cols,
            ),
            params=dict(
                loss_function="Logloss",
                eval_metric="AUC",
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=4.0,
                iterations=1600,
                random_seed=42,
                grow_policy="SymmetricTree",
                bagging_temperature=0.6,
                allow_writing_files=False,
            ),
            transform=None,
            use_sample_weights=True,
        ),
    ]
    if voice_cols or text_cols:
        strategies.append(
            StrategyConfig(
                name="HGB_voice_text",
                model_type="hgb",
                features=merge_features(feature_base, voice_cols, text_cols),
                params=dict(
                    loss="log_loss",
                    learning_rate=0.03,
                    max_leaf_nodes=255,
                    max_depth=None,
                    max_iter=520,
                    l2_regularization=0.12,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                ),
                transform="quantile",
                use_sample_weights=False,
            )
        )
    if voice_cols:
        strategies.append(
            StrategyConfig(
                name="Voice_HGB",
                model_type="hgb",
                features=merge_features(voice_cols),
                params=dict(
                    loss="log_loss",
                    learning_rate=0.08,
                    max_leaf_nodes=127,
                    max_depth=4,
                    max_iter=220,
                    l2_regularization=0.12,
                    min_samples_leaf=12,
                    class_weight="balanced",
                    random_state=42,
                ),
                transform="quantile",
                use_sample_weights=False,
            )
        )

    os.makedirs("logs", exist_ok=True)
    results_summary = []
    stacking_payloads: Dict[str, Dict[str, pd.Series]] = {}
    total_strategies = len(strategies)
    cumulative_time = 0.0
    for idx, strategy in enumerate(strategies, 1):
        strategy_start = time.time()
        metrics, prediction_payload = evaluate_strategy(
            dataset=dataset,
            features=strategy.features,
            strategy=strategy,
            train_index=train_idx,
            val_index=val_idx,
            cv_splits=args.cv_folds,
            return_predictions=True,
        )
        elapsed = time.time() - strategy_start
        cumulative_time += elapsed
        avg_time = cumulative_time / idx
        remaining = avg_time * (total_strategies - idx)
        print(
            f"[Timer] {strategy.name} evaluated in {format_seconds(elapsed)} (ETA ~{format_seconds(max(0.0, remaining))})"
        )
        if prediction_payload is not None:
            stacking_payloads[strategy.name] = prediction_payload
        results_summary.append((strategy.name, metrics))
        print(
            f"{strategy.name}: holdout ROC-AUC {metrics['holdout_auc']:.4f} | "
            f"CV mean {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
        )

    stacking_results: Optional[Dict[str, object]] = None
    if stacking_payloads:
        sorted_strategies = sorted(
            results_summary, key=lambda item: item[1]["holdout_auc"], reverse=True
        )
        top_count = min(5, len(sorted_strategies))
        selected_names = [name for name, _ in sorted_strategies[:top_count]]

        train_mi = pd.MultiIndex.from_tuples(train_idx, names=dataset.index.names)
        val_mi = pd.MultiIndex.from_tuples(val_idx, names=dataset.index.names)
        meta_train_df = pd.DataFrame(index=train_mi)
        meta_holdout_df = pd.DataFrame(index=val_mi)

        base_prediction_cols: List[str] = []
        for name in selected_names:
            payload = stacking_payloads.get(name)
            if payload is None:
                continue
            meta_train_df[name] = payload["oof"]
            meta_holdout_df[name] = payload["holdout"]
            base_prediction_cols.append(name)

        if not meta_train_df.empty and base_prediction_cols:
            extra_meta_features = [
                "prev_PHQ9_Score",
                "prev_GAD7_Score",
                "prev_ISI_Score",
                "prev_WHOQOL_Score",
                "prev_PSS_Score",
                "prev_WHODAS_Score",
                "prev_Loneliness_Score",
                "prev_S_Score",
                "ema_depression",
                "ema_anxiety",
                "ema_sleep",
                "ema_stress",
                "steps",
                "screen_time",
                "heart_rate",
            ]
            if context.get("exclude_prev_survey"):
                extra_meta_features = [
                    col for col in extra_meta_features if not col.startswith("prev_")
                ]
            extra_meta_features = [
                col for col in extra_meta_features if col in dataset.columns
            ]
            if extra_meta_features:
                meta_train_df = meta_train_df.join(
                    dataset.loc[train_mi, extra_meta_features], how="left"
                )
                meta_holdout_df = meta_holdout_df.join(
                    dataset.loc[val_mi, extra_meta_features], how="left"
                )

            meta_train_df = meta_train_df.fillna(meta_train_df.mean())
            meta_holdout_df = meta_holdout_df.fillna(meta_train_df.mean())

            y_train_meta = stacking_payloads[selected_names[0]]["y_train"]
            y_holdout_meta = stacking_payloads[selected_names[0]]["y_holdout"]
            groups_meta = _extract_group_labels(meta_train_df.index)
            X_train_meta = meta_train_df.values.astype(np.float32)
            X_holdout_meta = meta_holdout_df.values.astype(np.float32)

            scaler, meta_clf, best_c, lr_cv = _tune_logistic_meta(
                X_train_meta, y_train_meta.values, groups_meta
            )
            holdout_meta_pred = meta_clf.predict_proba(scaler.transform(X_holdout_meta))[:, 1]
            stack_auc = roc_auc_score(y_holdout_meta.values, holdout_meta_pred)

            blend_pred = meta_holdout_df[base_prediction_cols].mean(axis=1)
            blend_auc = roc_auc_score(y_holdout_meta.values, blend_pred.values)

            meta_xgb_model, meta_xgb_params, xgb_cv = _tune_xgb_meta(
                X_train_meta, y_train_meta.values, groups_meta
            )
            xgb_meta_pred = meta_xgb_model.predict_proba(X_holdout_meta)[:, 1]
            xgb_meta_auc = roc_auc_score(y_holdout_meta.values, xgb_meta_pred)

            print(
                f"Blend (top-{len(selected_names)} mean): holdout ROC-AUC {blend_auc:.4f}"
            )
            print(
                f"Stacking (top-{len(selected_names)} LR, C={best_c:.3f}): holdout ROC-AUC {stack_auc:.4f}"
            )
            print(
                "Stacking "
                f"(top-{len(selected_names)} XGB, lr={meta_xgb_params['learning_rate']}, "
                f"depth={meta_xgb_params['max_depth']}, estimators={meta_xgb_params['n_estimators']}): "
                f"holdout ROC-AUC {xgb_meta_auc:.4f}"
            )

            stacking_results = {
                "selected_models": selected_names,
                "stack_auc": stack_auc,
                "blend_auc": blend_auc,
                "stack_xgb_auc": xgb_meta_auc,
                "stack_predictions": pd.Series(
                    holdout_meta_pred, index=meta_holdout_df.index, name="StackingMeta"
                ),
                "stack_xgb_predictions": pd.Series(
                    xgb_meta_pred, index=meta_holdout_df.index, name="StackingXGB"
                ),
                "y_holdout": y_holdout_meta,
                "logreg_params": {"C": best_c, "cv_mean": lr_cv},
                "xgb_params": {**meta_xgb_params, "cv_mean": xgb_cv},
            }

    best = max(results_summary, key=lambda item: item[1]["holdout_auc"])
    log_lines = [
        f"{name}: holdout={metrics['holdout_auc']:.4f}, cv_mean={metrics['cv_mean']:.4f}, cv_std={metrics['cv_std']:.4f}"
        for name, metrics in results_summary
    ]
    log_lines.append(f"Best strategy: {best[0]} with holdout ROC-AUC {best[1]['holdout_auc']:.4f}")

    try:
        best_strategy_cfg = next(s for s in strategies if s.name == best[0])
        block_metrics = evaluate_block_holdout(
            dataset=dataset,
            strategy=best_strategy_cfg,
            features=best_strategy_cfg.features,
            quantile=0.8,
        )
        block_line = (
            f"Block validation ({best[0]}): holdout={block_metrics['holdout_auc']:.4f}, "
            f"cv_mean={block_metrics['cv_mean']:.4f}, cv_std={block_metrics['cv_std']:.4f}"
        )
        print(block_line)
        log_lines.append(block_line)
    except Exception as exc:
        print(f"Block validation skipped: {exc}")

    if stacking_results is not None:
        selected_names = ",".join(stacking_results["selected_models"])  # type: ignore[index]
        log_lines.append(
            f"Blend (models={selected_names}): holdout={stacking_results['blend_auc']:.4f}"
        )
        log_lines.append(
            f"Stacking LR (models={selected_names}, C={stacking_results['logreg_params']['C']:.3f}): "
            f"holdout={stacking_results['stack_auc']:.4f}"
        )
        if "stack_xgb_auc" in stacking_results:
            log_lines.append(
                "Stacking XGB "
                f"(models={selected_names}, depth={stacking_results['xgb_params']['max_depth']}, "
                f"n_estimators={stacking_results['xgb_params']['n_estimators']}): "
                f"holdout={stacking_results['stack_xgb_auc']:.4f}"
            )

    with open("logs/tabular_results.txt", "w") as f:
        f.write("\n".join(log_lines))

    print(f"[Timer] Total runtime {format_seconds(time.time() - overall_start)}")


if __name__ == "__main__":
    main()
