import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
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
    parser.add_argument("--top-k-features", type=int, default=120, help="Number of top-importance features for top-k strategies.")
    parser.add_argument("--top-k-min", type=int, default=None, help="Minimum feature count when strategies tune top-k subsets.")
    args = parser.parse_args()

    overall_start = time.time()

    step_start = time.time()
    dataset = build_feature_dataframe(history_hours=args.history_hours)
    print(f"[Timer] Feature dataframe built in {time.time() - step_start:.1f}s")

    step_start = time.time()
    context = prepare_feature_context(
        dataset,
        split_seed=args.split_seed,
        exclude_prev_survey=args.exclude_prev_survey,
        top_k_features=args.top_k_features,
        top_k_min=args.top_k_min,
    )
    print(f"[Timer] Feature context prepared in {time.time() - step_start:.1f}s")
    strategies = build_default_strategies(context)
    strategy = find_strategy(args.strategy, tuple(strategies))

    step_start = time.time()
    metrics, preds = evaluate_strategy(
        dataset=dataset,
        features=strategy.features,
        strategy=strategy,
        train_index=context["train_idx"],  # type: ignore[index]
        val_index=context["val_idx"],  # type: ignore[index]
        cv_splits=args.cv_folds,
        return_predictions=True,
    )
    print(f"[Timer] Strategy evaluation completed in {time.time() - step_start:.1f}s")
    if preds is None:
        raise RuntimeError("Predictions were not returned; cannot compute curve-based metrics.")

    y_true = preds["y_holdout"].values  # type: ignore[index]
    y_scores = preds["holdout"].values  # type: ignore[index]
    best_f1 = select_best_threshold(y_true, y_scores, mode="f1")
    best_f2 = select_best_threshold(y_true, y_scores, mode="f2")
    default_metrics = compute_threshold_metrics(y_true, y_scores, threshold=0.5)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    frac_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="uniform")
    cm_default = confusion_matrix(y_true, (y_scores >= 0.5).astype(int))
    cm_best_f1 = confusion_matrix(y_true, (y_scores >= best_f1["threshold"]).astype(int))
    cm_best_f2 = confusion_matrix(y_true, (y_scores >= best_f2["threshold"]).astype(int))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario = "no_prev" if args.exclude_prev_survey else "with_prev"
    prefix = args.output_prefix or f"{strategy.name}_{scenario}_{timestamp}"

    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    step_start = time.time()

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

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#d62728", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {strategy.name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    pr_path = os.path.join("plots", f"pr_{prefix}.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", color="#2ca02c")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {strategy.name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    calib_path = os.path.join("plots", f"calibration_{prefix}.png")
    plt.tight_layout()
    plt.savefig(calib_path, dpi=300)
    plt.close()

    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    pr_data = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }
    results = {
        "strategy": strategy.name,
        "history_hours": args.history_hours,
        "cv_folds": args.cv_folds,
        "split_seed": args.split_seed,
        "holdout_metrics": metrics,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "calibration": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_of_positives": frac_pos.tolist(),
        },
        "confusion_matrices": {
            "default": cm_default.tolist(),
            "best_f1": cm_best_f1.tolist(),
            "best_f2": cm_best_f2.tolist(),
        },
        "best_f1": best_f1,
        "best_f2": best_f2,
        "threshold_0_5": default_metrics,
        "roc_plot": roc_path,
        "pr_plot": pr_path,
        "calibration_plot": calib_path,
        "scenario": scenario,
    }

    output_path = os.path.join("logs", f"evaluation_{prefix}.json")
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Saved evaluation summary to {output_path}")
    print(f"ROC curve image: {roc_path}")
    print(f"PR curve image: {pr_path}")
    print(f"Calibration curve image: {calib_path}")
    print(
        f"[Timer] Total evaluation runtime {time.time() - overall_start:.1f}s"
    )


if __name__ == "__main__":
    main()
