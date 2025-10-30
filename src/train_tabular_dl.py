import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from run_tabular_models import (
    build_feature_dataframe,
    build_default_strategies,
    prepare_feature_context,
)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore

    HAS_TABNET = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TABNET = False


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(np.array(self.targets[idx], dtype=np.float32)),
        )


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_numpy_arrays(
    dataset: pd.DataFrame,
    features: List[str],
    train_idx: List[Tuple[str, int]],
    val_idx: List[Tuple[str, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_index = pd.MultiIndex.from_tuples(train_idx, names=dataset.index.names)
    val_index = pd.MultiIndex.from_tuples(val_idx, names=dataset.index.names)

    X = dataset[features].copy()
    train_median = X.loc[train_index].median()
    X = X.fillna(train_median)

    X_train = X.loc[train_index].to_numpy(dtype=np.float32)
    X_val = X.loc[val_index].to_numpy(dtype=np.float32)

    y_train = dataset.loc[train_index, "target_binary"].to_numpy(dtype=np.float32)
    y_val = dataset.loc[val_index, "target_binary"].to_numpy(dtype=np.float32)
    return X_train, y_train, X_val, y_val


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
) -> Dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.hidden_layers,
        dropout=args.dropout,
    ).to(device)

    pos_ratio = y_train.mean()
    pos_weight = torch.tensor((1 - pos_ratio) / max(pos_ratio, 1e-6), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_probs)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(torch.from_numpy(X_train).to(device))).cpu().numpy()
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()

    return {
        "train_auc": float(roc_auc_score(y_train, train_probs)),
        "holdout_auc": float(roc_auc_score(y_val, val_probs)),
        "val_probs": val_probs.tolist(),
        "train_probs": train_probs.tolist(),
    }


def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
) -> Dict[str, float]:
    if not HAS_TABNET:
        raise ImportError("pytorch-tabnet is not installed. Install via `pip install pytorch-tabnet`.")

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = len(y_train) / (2 * max(pos_count, 1e-6))
    neg_weight = len(y_train) / (2 * max(neg_count, 1e-6))
    sample_weights = np.where(y_train == 1, pos_weight, neg_weight)
    tabnet = TabNetClassifier(
        n_d=args.tabnet_dim,
        n_a=args.tabnet_dim,
        n_steps=args.tabnet_steps,
        gamma=args.tabnet_gamma,
        lambda_sparse=args.tabnet_lambda_sparse,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=args.learning_rate),
        scheduler_params=None,
        scheduler_fn=None,
        seed=args.split_seed,
    )
    tabnet.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=max(1, args.batch_size // 2),
        num_workers=0,
        drop_last=False,
        weights=sample_weights,
    )
    train_probs = tabnet.predict_proba(X_train)[:, 1]
    val_probs = tabnet.predict_proba(X_val)[:, 1]
    return {
        "train_auc": float(roc_auc_score(y_train, train_probs)),
        "holdout_auc": float(roc_auc_score(y_val, val_probs)),
        "val_probs": val_probs.tolist(),
        "train_probs": train_probs.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular deep learning models (MLP / TabNet).")
    parser.add_argument("--model-type", choices=["mlp", "tabnet"], default="mlp")
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--exclude-prev-survey", action="store_true")
    parser.add_argument("--top-k-features", type=int, default=120, help="Number of top-importance features to retain for feature-based models.")
    parser.add_argument("--top-k-min", type=int, default=None, help="Minimum feature count when top-k selection is tuned.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tabnet-dim", type=int, default=32)
    parser.add_argument("--tabnet-steps", type=int, default=3)
    parser.add_argument("--tabnet-gamma", type=float, default=1.3)
    parser.add_argument("--tabnet-lambda-sparse", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="logs")
    args = parser.parse_args()

    dataset = build_feature_dataframe(history_hours=args.history_hours)
    context = prepare_feature_context(
        dataset,
        split_seed=args.split_seed,
        exclude_prev_survey=args.exclude_prev_survey,
        top_k_features=args.top_k_features,
        top_k_min=args.top_k_min,
    )
    strategies = build_default_strategies(context)
    best_strategy = next(s for s in strategies if s.name == "HGB_optuna_best")
    features = [f for f in best_strategy.features if f in dataset.columns]

    train_idx = context["train_idx"]  # type: ignore[index]
    val_idx = context["val_idx"]  # type: ignore[index]

    X_train, y_train, X_val, y_val = build_numpy_arrays(dataset, features, train_idx, val_idx)

    if args.model_type == "mlp":
        metrics = train_mlp(X_train, y_train, X_val, y_val, args)
    else:
        metrics = train_tabnet(X_train, y_train, X_val, y_val, args)

    roc_auc = metrics["holdout_auc"]
    preds = metrics.pop("val_probs")
    train_preds = metrics.pop("train_probs")

    threshold = 0.5
    val_pred_labels = (np.array(preds) >= threshold).astype(int)
    acc = accuracy_score(y_val, val_pred_labels)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario = "no_prev" if args.exclude_prev_survey else "with_prev"
    prefix = f"tabular_{args.model_type}_{scenario}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{prefix}.json")
    output_payload = dict(
        model=args.model_type,
        scenario=scenario,
        history_hours=args.history_hours,
        split_seed=args.split_seed,
        exclude_prev=args.exclude_prev_survey,
        holdout_auc=roc_auc,
        holdout_accuracy=acc,
        metrics=metrics,
    )
    with open(output_path, "w") as fp:
        json.dump(output_payload, fp, indent=2)

    print(f"Saved results to {output_path}")
    print(f"Holdout ROC-AUC: {roc_auc:.4f} | Accuracy@0.5: {acc:.4f}")


if __name__ == "__main__":
    main()
