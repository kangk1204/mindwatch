"""
Generate SHAP-style explanations and feature rankings for the tuned XGB model.

This script rebuilds the tabular feature matrix, fits the Optuna-selected XGB model,
and computes Tree SHAP values on a sampled validation holdout. The resulting mean
absolute SHAP scores are saved to TSV, and bar / beeswarm visualisations are exported
as SVGs for publication-friendly usage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402

from run_full_pipeline import build_xgb_strategy  # noqa: E402
from run_tabular_models import (  # noqa: E402
    build_feature_dataframe,
    prepare_feature_context,
)

FONT_DIR = Path("fonts")
FONT_CANDIDATES = [
    "NotoSansKR-Regular.otf",
    "NanumGothic.otf",
    "NanumGothic-Regular.ttf",
    "NanumGothic.ttf",
]

FONT_PROP = None

for font_name in FONT_CANDIDATES:
    font_path = FONT_DIR / font_name
    if font_path.exists():
        try:
            font_manager.fontManager.addfont(str(font_path))
        except RuntimeError:
            continue
        else:
            font_prop = font_manager.FontProperties(fname=str(font_path))
            matplotlib.rcParams["font.family"] = font_prop.get_name()
            matplotlib.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            FONT_PROP = font_prop
            break

if FONT_PROP is None:
    FONT_PROP = font_manager.FontProperties()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP feature rankings and plots for the tuned XGB model."
    )
    parser.add_argument(
        "--optuna-json",
        type=Path,
        default=Path("results/20251103_voice_text_v3/optuna_xgb_full_20251102_133236.json"),
        help="Path to Optuna summary JSON containing best_params for xgb_full.",
    )
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=120)
    parser.add_argument("--top-k-min", type=int, default=40)
    parser.add_argument("--sample-size", type=int, default=200, help="Number of holdout samples used for SHAP.")
    parser.add_argument(
        "--background-size",
        type=int,
        default=500,
        help="Number of background rows drawn from the training split for TreeExplainer.",
    )
    parser.add_argument("--output-tsv", type=Path, default=Path("results/shap_feature_importance.tsv"))
    parser.add_argument("--output-bar", type=Path, default=Path("plots/shap_bar_top_features.svg"))
    parser.add_argument("--output-beeswarm", type=Path, default=Path("plots/shap_beeswarm.svg"))
    parser.add_argument("--max-display", type=int, default=20, help="Number of features shown on the plots.")
    return parser.parse_args()


def _multiindex(index_pairs: Sequence[tuple[str, int]], names: Sequence[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(index_pairs, names=names)


def main() -> None:
    args = parse_args()

    if not args.optuna_json.exists():
        raise FileNotFoundError(
            f"Optuna summary not found at {args.optuna_json}. Run the tuning pipeline first."
        )

    with args.optuna_json.open("r", encoding="utf-8") as fp:
        optuna_summary = json.load(fp)
    best_params = optuna_summary.get("best_params")
    if not best_params:
        raise ValueError(f"No best_params present in {args.optuna_json}")

    dataset = build_feature_dataframe(history_hours=args.history_hours)
    context = prepare_feature_context(
        dataset,
        split_seed=args.split_seed,
        exclude_prev_survey=False,
        top_k_features=args.top_k_features,
        top_k_min=args.top_k_min,
    )

    strategy = build_xgb_strategy(context, best_params)
    features = [feature for feature in strategy.features if feature in dataset.columns]
    if not features:
        raise ValueError("Strategy features do not overlap with dataset columns.")

    train_index = _multiindex(context["train_idx"], dataset.index.names)  # type: ignore[arg-type]
    val_index = _multiindex(context["val_idx"], dataset.index.names)  # type: ignore[arg-type]

    X_train = dataset.loc[train_index, features].copy()
    y_train = dataset.loc[train_index, "target_binary"].copy()
    X_val = dataset.loc[val_index, features].copy()

    median = X_train.median()
    X_train = X_train.fillna(median)
    X_val = X_val.fillna(median)

    model = xgb.XGBClassifier(**strategy.params)
    model.fit(X_train.values, y_train.values)

    background_size = min(args.background_size, len(X_train))
    sample_size = min(args.sample_size, len(X_val))
    rng = np.random.default_rng(seed=42)
    background_idx = rng.choice(len(X_train), size=background_size, replace=False)
    sample_idx = rng.choice(len(X_val), size=sample_size, replace=False)
    background = X_train.iloc[background_idx]
    sample = X_val.iloc[sample_idx]

    def predict_proba(data: np.ndarray) -> np.ndarray:
        return model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(
        predict_proba,
        background,
        algorithm="permutation",
    )
    max_evals = max(2 * sample.shape[1] + 1, 501)
    explanation = explainer(sample, max_evals=max_evals)

    shap_values = np.asarray(explanation.values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output_tsv, sep="\t", index=False)

    args.output_bar.parent.mkdir(parents=True, exist_ok=True)
    top_features = importance.head(args.max_display)
    fig, ax = plt.subplots(figsize=(7, max(4, len(top_features) * 0.4)))
    ax.barh(
        top_features["feature"][::-1],
        top_features["mean_abs_shap"][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP value|", fontproperties=FONT_PROP)
    ax.set_ylabel("Feature", fontproperties=FONT_PROP)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_PROP)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_PROP)
    fig.tight_layout()
    fig.savefig(args.output_bar, bbox_inches="tight")
    plt.close(fig)

    feature_order = top_features["feature"].tolist()
    feature_index = {name: idx for idx, name in enumerate(sample.columns)}
    fig, ax = plt.subplots(figsize=(8.5, max(4, len(feature_order) * 0.45)))
    cmap = plt.get_cmap("coolwarm")
    rng_jitter = np.random.default_rng(seed=1234)
    data_subset = sample[feature_order]
    finite_vals = data_subset.replace([np.inf, -np.inf], np.nan).values.flatten()
    finite_vals = finite_vals[~np.isnan(finite_vals)]
    if finite_vals.size == 0 or np.nanmin(finite_vals) == np.nanmax(finite_vals):
        value_norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        value_norm = Normalize(vmin=float(np.nanmin(finite_vals)), vmax=float(np.nanmax(finite_vals)))

    for row_idx, feature in enumerate(reversed(feature_order)):
        col_idx = feature_index[feature]
        values = sample.iloc[:, col_idx].to_numpy()
        shap_vals = shap_values[:, col_idx]
        cleaned_vals = np.nan_to_num(values, nan=value_norm.vmin if np.isfinite(value_norm.vmin) else 0.0)
        colors = cmap(value_norm(cleaned_vals))
        jitter = (rng_jitter.random(len(shap_vals)) - 0.5) * 0.6
        ax.scatter(
            shap_vals,
            np.full_like(shap_vals, row_idx) + jitter,
            c=colors,
            s=14,
            alpha=0.65,
            edgecolors="none",
        )

    ax.set_yticks(np.arange(len(feature_order)))
    ax.set_yticklabels(list(reversed(feature_order)))
    ax.set_xlabel("SHAP value (impact on model output)", fontproperties=FONT_PROP)
    ax.set_ylabel("Feature", fontproperties=FONT_PROP)
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_PROP)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_PROP)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=value_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalised feature value", rotation=270, labelpad=15, fontproperties=FONT_PROP)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT_PROP)

    fig.tight_layout()
    fig.savefig(args.output_beeswarm, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
