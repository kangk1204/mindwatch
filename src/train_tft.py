import argparse
import math
import os
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, TorchNormalizer
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import lightning as L
import numpy as np

torch.set_float32_matmul_precision("medium")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "00_input_data")
LABEL_DIR = os.path.join(BASE_DIR, "00_label_data")


@dataclass
class Config:
    history_hours: int = 168
    min_coverage_ratio: float = 0.4
    random_seed: int = 42
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1
    loss_quantiles: Tuple[float, ...] = (0.5,)
    use_class_weights: bool = False
    task: str = "regression"


class BinaryCrossEntropy(MultiHorizonMetric):
    """Binary cross-entropy loss returning per-time-step values."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = y_pred.view(-1)
        target = target.float().view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return loss.view(*y_pred.shape[:-1], 1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y_pred)

    def to_quantiles(
        self, y_pred: torch.Tensor, quantiles: List[float] | None = None
    ) -> torch.Tensor:
        return self.to_prediction(y_pred)


SENSOR_FILE_MAP = {
    "전체대상자_심박수_250905.csv": "heart_rate",
    "전체대상자_심부체온_250901.csv": "core_temp",
    "전체대상자_깊은수면시간_250905.csv": "deep_sleep",
    "전체대상자_얕은수면시간_250905.csv": "light_sleep",
    "전체대상자_렘수면시간_250905.csv": "rem_sleep",
    "전체대상자_기상시간_250905.csv": "wake_time",
    "전체대상자_스크린타임_250905.csv": "screen_time",
    "전체대상자_걸음수_250905.csv": "steps",
    "전체대상자_근접센서_250905.csv": "proximity",
    "전체대상자_산소포화도_250905.csv": "spo2",
    "전체대상자_HRV_250901.csv": "hrv",
    "전체대상자_EMA_Stress_250905.csv": "ema_stress",
    "전체대상자_EMA_Depression_250905.csv": "ema_depression",
    "전체대상자_EMA_Anxiety_250905.csv": "ema_anxiety",
    "전체대상자_EMA_Sleep_250905.csv": "ema_sleep",
}


LABEL_SHEETS = [
    ("(PSS추가)전체대상자_천안설문_1회차_250905.xlsx", "전체대상자_천안설문_1회차_250905", 1),
    ("(PSS추가)전체대상자_천안설문_2회차_250905.xlsx", "전체대상자_천안설문_2회차_250905", 2),
]


SENSOR_FEATURES: List[str] = list(SENSOR_FILE_MAP.values())
DERIVED_SUFFIXES = ["diff", "roll_mean_24h", "roll_std_24h"]
DERIVED_SENSOR_FEATURES: List[str] = [
    f"{feature}_{suffix}" for feature in SENSOR_FEATURES for suffix in DERIVED_SUFFIXES
]

STATIC_CATEGORICALS = ["Sex"]
AGG_STATS = ["mean", "std"]
STATIC_AGG_REALS = [f"{feature}_{stat}" for feature in SENSOR_FEATURES for stat in AGG_STATS]
STATIC_REALS = ["age_years", "Height", "Weight"] + STATIC_AGG_REALS
TIME_VARYING_REALS: List[str] = SENSOR_FEATURES + DERIVED_SENSOR_FEATURES + [
    "hours_to_label",
    "is_decoder_step",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_normalized_path(directory: str, filename: str) -> Optional[str]:
    """Return an existing filesystem path regardless of Unicode normalization."""
    candidate = os.path.join(directory, filename)
    if os.path.exists(candidate):
        return candidate

    normalized_forms = {
        unicodedata.normalize("NFC", filename),
        unicodedata.normalize("NFD", filename),
    }
    for norm_name in normalized_forms:
        candidate = os.path.join(directory, norm_name)
        if os.path.exists(candidate):
            return candidate

    target_norm = unicodedata.normalize("NFC", filename)
    for entry in os.listdir(directory):
        if unicodedata.normalize("NFC", entry) == target_norm:
            candidate = os.path.join(directory, entry)
            if os.path.exists(candidate):
                return candidate
    return None


def load_label_frames() -> pd.DataFrame:
    frames = []
    for filename, sheet, wave in LABEL_SHEETS:
        expected_path = os.path.join(LABEL_DIR, filename)
        path = resolve_normalized_path(LABEL_DIR, filename)
        if path is None:
            raise FileNotFoundError(f"Label file missing: {expected_path}")
        df = pd.read_excel(path, sheet_name=sheet)
        df["survey_wave"] = wave
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"Timestamp": "survey_timestamp"})

    # drop impossible PHQ9 values (observed issue in raw sheet)
    combined.loc[combined["PHQ9_Score"] > 27, "PHQ9_Score"] = np.nan
    combined = combined.dropna(subset=["ID", "survey_timestamp", "PHQ9_Score"])

    combined["PHQ9_Score"] = combined["PHQ9_Score"].astype(float)
    combined["phq9_binary"] = (combined["PHQ9_Score"] >= 10).astype(int)
    combined["survey_timestamp"] = pd.to_datetime(combined["survey_timestamp"])
    return combined


def load_hourly_sensor_frames() -> Dict[str, pd.DataFrame]:
    sensor_frames: Dict[str, pd.DataFrame] = {}
    for filename, feature_name in SENSOR_FILE_MAP.items():
        expected_path = os.path.join(INPUT_DIR, filename)
        path = resolve_normalized_path(INPUT_DIR, filename)
        if path is None:
            raise FileNotFoundError(f"Sensor file missing: {expected_path}")
        df = pd.read_csv(path)
        if "-" not in df.columns:
            raise ValueError(f"Timestamp column '-' not found in {filename}")
        df = df.rename(columns={"-": "timestamp"})
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = timestamps.dt.tz_convert(None)
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        value_cols = [col for col in df.columns if col != "timestamp"]

        if feature_name == "wake_time":
            for col in value_cols:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[col] = (dt.dt.hour + dt.dt.minute / 60.0).astype(np.float32)
        else:
            df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

        df = df.set_index("timestamp")
        hourly = df.resample("1h").mean()
        hourly = hourly.astype(np.float32)
        sensor_frames[feature_name] = hourly
    return sensor_frames


def coverage_ratio(sequence: pd.DataFrame) -> float:
    valid = sequence.notna().sum().sum()
    total = sequence.shape[0] * sequence.shape[1]
    if total == 0:
        return 0.0
    return valid / total


def build_sample_sequence(
    participant_id: str,
    survey_ts: pd.Timestamp,
    sensor_frames: Dict[str, pd.DataFrame],
    history_hours: int,
) -> Optional[pd.DataFrame]:
    timeline = pd.date_range(
        end=survey_ts.floor("h"),
        periods=history_hours + 1,
        freq="1h",
    )
    data = pd.DataFrame(index=timeline)

    for feature_name, frame in sensor_frames.items():
        if participant_id not in frame.columns:
            data[feature_name] = np.nan
            continue
        series = frame[participant_id].reindex(timeline)
        data[feature_name] = series

    if coverage_ratio(data.iloc[:-1]) == 0:
        return None

    data = data.ffill()
    # still missing after forward-fill (e.g., leading gaps)
    data = data.fillna(0.0)

    for feature_name in SENSOR_FEATURES:
        series = data[feature_name]
        diff = series.diff().fillna(0.0).astype(np.float32)
        roll_mean = (
            series.shift(1)
            .rolling(window=24, min_periods=1)
            .mean()
            .fillna(0.0)
            .astype(np.float32)
        )
        roll_std = (
            series.shift(1)
            .rolling(window=24, min_periods=1)
            .std(ddof=0)
            .fillna(0.0)
            .astype(np.float32)
        )
        data[f"{feature_name}_diff"] = diff
        data[f"{feature_name}_roll_mean_24h"] = roll_mean
        data[f"{feature_name}_roll_std_24h"] = roll_std

    hours_to_label = (timeline - timeline[-1]).total_seconds() / 3600.0
    data["hours_to_label"] = hours_to_label.astype(np.float32)
    is_decoder = np.zeros(len(timeline), dtype=np.float32)
    is_decoder[-1] = 1.0
    data["is_decoder_step"] = is_decoder
    return data


def build_model_dataframe(
    label_df: pd.DataFrame,
    sensor_frames: Dict[str, pd.DataFrame],
    cfg: Config,
) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for _, row in label_df.iterrows():
        seq = build_sample_sequence(
            participant_id=row["ID"],
            survey_ts=row["survey_timestamp"],
            sensor_frames=sensor_frames,
            history_hours=cfg.history_hours,
        )
        if seq is None:
            continue
        seq["ID"] = row["ID"]
        seq["survey_wave"] = row["survey_wave"]
        seq["time_idx"] = np.arange(-cfg.history_hours, 1, dtype=np.int32)
        seq["target_score"] = 0.0
        seq.iloc[-1, seq.columns.get_loc("target_score")] = row["PHQ9_Score"]
        seq["target_binary"] = 0.0
        seq.iloc[-1, seq.columns.get_loc("target_binary")] = float(row["phq9_binary"])

        birthdate = row.get("Age", np.nan)
        age_years = np.nan
        if pd.notna(birthdate):
            birth_ts = pd.to_datetime(birthdate, errors="coerce")
            if pd.notna(birth_ts):
                age_years = (row["survey_timestamp"] - birth_ts).days / 365.25
        seq["age_years"] = age_years
        seq["Height"] = row.get("Height", np.nan)
        seq["Weight"] = row.get("Weight", np.nan)
        for feature_name in SENSOR_FILE_MAP.values():
            mean_val = seq[feature_name].mean()
            std_val = seq[feature_name].std(ddof=0)
            seq[f"{feature_name}_mean"] = np.float32(mean_val) if np.isfinite(mean_val) else np.float32(0.0)
            seq[f"{feature_name}_std"] = np.float32(std_val) if np.isfinite(std_val) else np.float32(0.0)

        for col in STATIC_CATEGORICALS:
            value = row.get(col, "Unknown")
            if isinstance(value, float) and math.isnan(value):
                value = "Unknown"
            seq[col] = value

        if coverage_ratio(seq[TIME_VARYING_REALS]) < cfg.min_coverage_ratio:
            continue

        records.append(seq.reset_index(drop=True))

    if not records:
        raise RuntimeError("No sequences constructed - check data availability.")

    model_df = pd.concat(records, ignore_index=True)
    for column in STATIC_REALS:
        if column in model_df.columns:
            model_df[column] = model_df[column].astype(np.float32)

    for column in STATIC_CATEGORICALS:
        model_df[column] = model_df[column].fillna("Unknown").astype("category")

    return model_df


def participant_stratified_split(
    entity_targets: pd.DataFrame,
    val_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    participant_labels = (
        entity_targets[["ID", "target_binary"]]
        .groupby("ID")["target_binary"]
        .max()
    )
    stratify_labels: Optional[np.ndarray]
    if participant_labels.nunique() > 1:
        stratify_labels = participant_labels.values
    else:
        stratify_labels = None
    participant_ids = participant_labels.index.to_numpy()
    train_ids, val_ids = train_test_split(
        participant_ids,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify_labels,
    )
    train_entities = entity_targets[entity_targets["ID"].isin(train_ids)][["ID", "survey_wave"]]
    val_entities = entity_targets[entity_targets["ID"].isin(val_ids)][["ID", "survey_wave"]]
    return train_entities, val_entities


def make_datasets(
    model_df: pd.DataFrame,
    cfg: Config,
    val_fraction: float = 0.2,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    entity_targets = (
        model_df[model_df["time_idx"] == 0][["ID", "survey_wave", "target_binary"]]
        .drop_duplicates()
    )
    train_entities, val_entities = participant_stratified_split(
        entity_targets, val_fraction=val_fraction, random_state=cfg.random_seed
    )

    def subset(df: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(entities, on=["ID", "survey_wave"], how="inner")
        return merged

    train_df = subset(model_df, train_entities).copy()
    val_df = subset(model_df, val_entities).copy()

    if cfg.use_class_weights:
        train_targets = (
            train_df[train_df["time_idx"] == 0][["ID", "survey_wave", "target_binary"]]
            .drop_duplicates()
        )
        pos = (train_targets["target_binary"] >= 1).sum()
        neg = len(train_targets) - pos
        weight_pos = len(train_targets) / (2 * pos) if pos > 0 else 1.0
        weight_neg = len(train_targets) / (2 * neg) if neg > 0 else 1.0
        train_targets["sample_weight"] = np.where(
            train_targets["target_binary"] >= 1,
            weight_pos,
            weight_neg,
        ).astype(np.float32)
        weight_map = train_targets.set_index(["ID", "survey_wave"])["sample_weight"]
        train_df = train_df.join(weight_map, on=["ID", "survey_wave"])
        val_df["sample_weight"] = 1.0
    else:
        if "sample_weight" in train_df.columns:
            train_df = train_df.drop(columns=["sample_weight"])
        if "sample_weight" in val_df.columns:
            val_df = val_df.drop(columns=["sample_weight"])

    if cfg.task == "classification":
        target_col = "target_binary"
        target_normalizer = None
    else:
        target_col = "target_score"
        target_normalizer = "auto"

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["ID", "survey_wave"],
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=["hours_to_label", "is_decoder_step"],
        time_varying_unknown_reals=[
            col for col in TIME_VARYING_REALS if col not in {"hours_to_label", "is_decoder_step"}
        ],
        max_encoder_length=cfg.history_hours,
        max_prediction_length=1,
        min_encoder_length=cfg.history_hours,
        min_prediction_length=1,
        target_normalizer=target_normalizer,
        add_relative_time_idx=False,
        add_encoder_length=False,
        scalers={"*": TorchNormalizer(method="standard")},
        weight="sample_weight" if cfg.use_class_weights else None,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    return training, validation


def train_tft(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    cfg: Config,
) -> Tuple[TemporalFusionTransformer, DataLoader, DataLoader]:
    num_workers = max(1, (os.cpu_count() or 1) // 2)
    train_loader = training.to_dataloader(
        train=True,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
    )

    if cfg.task == "classification":
        loss = BinaryCrossEntropy()
        output_size = 1
    else:
        loss = QuantileLoss(quantiles=list(cfg.loss_quantiles))
        output_size = len(cfg.loss_quantiles)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=4,
        dropout=cfg.dropout,
        hidden_continuous_size=64,
        loss=loss,
        output_size=output_size,
        lstm_layers=cfg.lstm_layers,
        reduce_on_plateau_patience=3,
    )

    strategy = "auto"
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=0.1,
        accelerator="auto",
        log_every_n_steps=50,
        callbacks=[],
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return model, train_loader, val_loader, trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer for PHQ9 prediction.")
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        history_hours=args.history_hours,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        lstm_layers=args.lstm_layers,
        random_seed=args.seed,
        loss_quantiles=tuple(args.quantiles),
        use_class_weights=args.use_class_weights,
        task=args.task,
    )

    set_seed(cfg.random_seed)

    label_df = load_label_frames()
    sensor_frames = load_hourly_sensor_frames()
    model_df = build_model_dataframe(label_df, sensor_frames, cfg)

    training, validation = make_datasets(model_df, cfg)
    model, train_loader, val_loader, trainer = train_tft(training, validation, cfg)

    prediction_obj = model.predict(
        val_loader, return_y=True, return_index=True, trainer_kwargs=dict(logger=False)
    )
    prediction_tensor = prediction_obj.output.detach()
    if cfg.task == "classification":
        logits = prediction_tensor[..., 0]
        if logits.ndim == 2:
            logits = logits[:, 0]
        val_predictions = torch.sigmoid(logits).cpu().numpy()
    else:
        if prediction_tensor.ndim == 3:
            if prediction_tensor.size(-1) > 1:
                try:
                    median_idx = cfg.loss_quantiles.index(0.5)
                except ValueError:
                    median_idx = prediction_tensor.size(-1) // 2
                prediction_tensor = prediction_tensor[..., median_idx]
            else:
                prediction_tensor = prediction_tensor[..., 0]
        val_predictions = prediction_tensor.squeeze(-1).reshape(-1).cpu().numpy()
    true_tensor = prediction_obj.y[0].detach()
    val_targets = true_tensor.reshape(-1).cpu().numpy()

    if np.isnan(val_predictions).any():
        nan_mask = np.isnan(val_predictions)
        dropped = int(nan_mask.sum())
        val_predictions = val_predictions[~nan_mask]
        val_targets = val_targets[~nan_mask]
        print(f"Dropped {dropped} samples with NaN predictions before ROC-AUC computation.")
    val_auc = roc_auc(val_targets, val_predictions)
    print(f"Validation ROC-AUC: {val_auc:.4f}")


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    unique = np.unique(y_true)
    if np.array_equal(unique, [0]) or np.array_equal(unique, [1]) or np.array_equal(unique, [0, 1]):
        y_binary = y_true.astype(int)
    else:
        y_binary = (y_true >= 10).astype(int)
    return roc_auc_score(y_binary, y_score)


if __name__ == "__main__":
    main()
