import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from run_tabular_models import (
    StrategyConfig,
    build_default_strategies,
    build_feature_dataframe,
    evaluate_strategy,
    prepare_feature_context,
)


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    specificity = ((y_pred == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "mcc": float(mcc),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
    }


def select_best_threshold(y_true: np.ndarray, y_scores: np.ndarray, mode: str = "f1") -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    metrics = []
    for thr in thresholds:
        stats = compute_threshold_metrics(y_true, y_scores, thr)
        stats["tpr"] = float(((y_scores >= thr) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
        stats["fpr"] = float(((y_scores >= thr) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
        metrics.append(stats)

    key = "f1" if mode == "f1" else "f2"
    best = max(metrics, key=lambda item: (item[key], -item["threshold"]))
    return best


def find_strategy(strategy_name: str, strategies: Tuple[StrategyConfig, ...]) -> StrategyConfig:
    for strategy in strategies:
        if strategy.name == strategy_name:
            return strategy
    raise ValueError(f"Strategy '{strategy_name}' not found. Available: {[s.name for s in strategies]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a tabular strategy with rich metrics and plots.")
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default="HGB_optuna_best")
    parser.add_argument("--exclude-prev-survey", action="store_true", help="Drop all prev_* and cross-wave survey features for ablation.")
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    dataset = build_feature_dataframe(history_hours=args.history_hours)
    context = prepare_feature_context(
        dataset,
        split_seed=args.split_seed,
        exclude_prev_survey=args.exclude_prev_survey,
    )
    strategies = build_default_strategies(context)
    strategy = find_strategy(args.strategy, tuple(strategies))

    metrics, preds = evaluate_strategy(
        dataset=dataset,
        features=strategy.features,
        strategy=strategy,
        train_index=context["train_idx"],  # type: ignore[index]
        val_index=context["val_idx"],  # type: ignore[index]
        cv_splits=args.cv_folds,
        return_predictions=True,
    )
    if preds is None:
        raise RuntimeError("Predictions were not returned; cannot compute curve-based metrics.")

    y_true = preds["y_holdout"].values  # type: ignore[index]
    y_scores = preds["holdout"].values  # type: ignore[index]
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    best_f1 = select_best_threshold(y_true, y_scores, mode="f1")
    best_f2 = select_best_threshold(y_true, y_scores, mode="f2")
    default_metrics = compute_threshold_metrics(y_true, y_scores, threshold=0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario = "no_prev" if args.exclude_prev_survey else "with_prev"
    prefix = args.output_prefix or f"{strategy.name}_{scenario}_{timestamp}"

    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{strategy.name} (AUC={roc_auc:.3f})", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {strategy.name}")
    plt.legend(loc="lower right")
    roc_path = os.path.join("plots", f"roc_{prefix}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    results = {
        "strategy": strategy.name,
        "history_hours": args.history_hours,
        "cv_folds": args.cv_folds,
        "split_seed": args.split_seed,
        "holdout_metrics": metrics,
        "roc_auc": roc_auc,
        "roc_curve": roc_data,
        "best_f1": best_f1,
        "best_f2": best_f2,
        "threshold_0_5": default_metrics,
        "roc_plot": roc_path,
        "scenario": scenario,
    }

    output_path = os.path.join("logs", f"evaluation_{prefix}.json")
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Saved evaluation summary to {output_path}")
    print(f"ROC curve image: {roc_path}")


if __name__ == "__main__":
    main()
