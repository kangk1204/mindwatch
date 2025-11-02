import argparse
import json
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_tabular_models import (
    StrategyConfig,
    build_default_strategies,
    build_feature_dataframe,
    evaluate_block_holdout,
    evaluate_strategy,
    prepare_feature_context,
    _extract_group_labels,
    _tune_logistic_meta,
    _tune_xgb_meta,
)


LOG_DIR = Path("logs")


def format_seconds(seconds: float) -> str:
    """Return human-readable formatting for elapsed seconds."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"



def run_optuna_batch(args: argparse.Namespace, storage_uri: str) -> None:
    cmd = [
        args.python_exec,
        str(Path(__file__).parent / "run_optuna_batch.py"),
        "--strategies",
        *args.strategies,
        "--history-hours",
        str(args.history_hours),
        "--cv-folds",
        str(args.cv_folds),
        "--trials",
        str(args.tuning_trials),
        "--split-seed",
        str(args.split_seed),
        "--top-k-features",
        str(args.top_k_features),
        "--top-k-min",
        str(args.top_k_min),
    ]
    if args.study_name:
        cmd.extend(["--study-name", args.study_name])
    if storage_uri:
        cmd.extend(["--storage", storage_uri])
    if args.exclude_prev_survey:
        cmd.append("--exclude-prev-survey")
    if args.use_gpu:
        cmd.append("--use-gpu")
    if args.block_validation:
        cmd.append("--block-validation")

    print(f"\n[Pipeline] Launching Optuna batch: {' '.join(cmd)}\n", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)
    duration = time.time() - start
    print(f"[Timer] Optuna batch completed in {format_seconds(duration)}")


def backfill_optuna_summaries(strategies: List[str], target_dir: Path) -> None:
    """Ensure optuna_{strategy}_*.json exists in target_dir by copying from default logs if needed."""
    default_log_dir = Path("logs")
    if default_log_dir.resolve() == target_dir.resolve():
        return
    for strat in strategies:
        pattern = f"optuna_{strat}_*.json"
        if list(target_dir.glob(pattern)):
            continue
        candidates = sorted(default_log_dir.glob(pattern))
        if candidates:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidates[-1], target_dir / candidates[-1].name)


def latest_optuna_summary(strategy: str, search_dir: Path) -> Dict:
    pattern = f"optuna_{strategy}_*.json"
    files = sorted(search_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Optuna summary found for strategy '{strategy}' in {search_dir}. "
            "Run with --run-tuning or point --optuna-dir to a directory containing optuna_{strategy}_*.json."
        )
    with files[-1].open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_hgb_strategy(context: Dict[str, object], params: Dict) -> StrategyConfig:
    pool = context["top_feature_pool"]  # type: ignore[index]
    top_k = int(params.get("top_k", context.get("top_k_default", len(pool))))  # type: ignore[arg-type]
    features = pool[:top_k]
    config_params = {
        "loss": "log_loss",
        "learning_rate": params["learning_rate"],
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "max_depth": int(params["max_depth"]),
        "max_iter": int(params["max_iter"]),
        "l2_regularization": float(params["l2_regularization"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_bins": int(params["max_bins"]),
        "class_weight": "balanced",
        "random_state": 42,
    }
    transform = "quantile" if params.get("use_quantile_transform", False) else None
    use_weights = bool(params.get("use_sample_weights", False))
    return StrategyConfig(
        name="HGB_optuna_best",
        model_type="hgb",
        features=features,
        params=config_params,
        transform=transform,
        use_sample_weights=use_weights,
    )


def build_lgbm_strategy(context: Dict[str, object], params: Dict) -> StrategyConfig:
    base_params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        random_state=42,
        n_jobs=-1,
    )
    config_params = dict(base_params)
    config_params.update(
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        n_estimators=int(params["n_estimators"]),
    )

    feature_mode = params.get("feature_mode", "topk")
    if feature_mode == "all":
        features = context["all_features"]  # type: ignore[index]
    else:
        pool = context["top_feature_pool"]  # type: ignore[index]
        top_k = int(params.get("top_k", len(pool)))
        features = pool[:top_k]

    use_weights = bool(params.get("use_sample_weights", False))
    return StrategyConfig(
        name="LGBM_optuna_best",
        model_type="lgbm",
        features=features,  # type: ignore[arg-type]
        params=config_params,
        transform=None,
        use_sample_weights=use_weights,
    )


def build_xgb_strategy(context: Dict[str, object], params: Dict) -> StrategyConfig:
    feature_mode = params.get("feature_mode", "topk")
    if feature_mode == "all":
        features = context["all_features"]  # type: ignore[index]
    else:
        pool = context["top_feature_pool"]  # type: ignore[index]
        top_k = int(params.get("top_k", len(pool)))
        features = pool[:top_k]

    config_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        colsample_bylevel=float(params.get("colsample_bytree", params["colsample_bytree"])),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        gamma=float(params["gamma"]),
        n_estimators=int(params["n_estimators"]),
        scale_pos_weight=float(params["scale_pos_weight"]),
    )

    use_weights = bool(params.get("use_sample_weights", False))
    return StrategyConfig(
        name="XGB_optuna_best",
        model_type="xgb",
        features=features,  # type: ignore[arg-type]
        params=config_params,
        transform=None,
        use_sample_weights=use_weights,
    )


def evaluate_strategies(
    dataset: pd.DataFrame,
    context: Dict[str, object],
    strategies: List[StrategyConfig],
    cv_folds: int,
    exclude_prev_survey: bool = False,
) -> Dict[str, Dict[str, float]]:
    train_idx = context["train_idx"]  # type: ignore[index]
    val_idx = context["val_idx"]  # type: ignore[index]
    results = {}
    stacking_payloads: Dict[str, Dict[str, pd.Series]] = {}

    total = len(strategies)
    cumulative = [0.0]
    for idx, strategy in enumerate(strategies, 1):
        start = time.time()
        metrics, payload = evaluate_strategy(
            dataset=dataset,
            features=strategy.features,
            strategy=strategy,
            train_index=train_idx,
            val_index=val_idx,
            cv_splits=cv_folds,
            return_predictions=True,
        )
        elapsed = time.time() - start
        cumulative[0] += elapsed
        avg = cumulative[0] / idx
        remaining = avg * (total - idx)
        print(
            f"[Timer] {strategy.name} evaluated in {format_seconds(elapsed)} "
            f"(ETA ~{format_seconds(max(0.0, remaining))})"
        )
        results[strategy.name] = metrics
        if payload:
            stacking_payloads[strategy.name] = payload
        print(
            f"{strategy.name}: holdout ROC-AUC {metrics['holdout_auc']:.4f} | "
            f"CV mean {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
        )

    stacking_info: Dict[str, float] = {}
    if stacking_payloads:
        stack_start = time.time()
        sorted_names = sorted(
            results.items(), key=lambda kv: kv[1]["holdout_auc"], reverse=True
        )
        top_names = [name for name, _ in sorted_names[:5]]

        train_idx_pairs = context["train_idx"]  # type: ignore[index]
        val_idx_pairs = context["val_idx"]  # type: ignore[index]
        train_mi = pd.MultiIndex.from_tuples(train_idx_pairs, names=dataset.index.names)
        val_mi = pd.MultiIndex.from_tuples(val_idx_pairs, names=dataset.index.names)
        meta_train = pd.DataFrame(index=train_mi)
        meta_holdout = pd.DataFrame(index=val_mi)
        base_cols: List[str] = []

        for name in top_names:
            payload = stacking_payloads.get(name)
            if not payload:
                continue
            meta_train[name] = payload["oof"]
            meta_holdout[name] = payload["holdout"]
            base_cols.append(name)

        if base_cols:
            extra_cols = [
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
            if exclude_prev_survey:
                extra_cols = [col for col in extra_cols if not col.startswith("prev_")]
            extra_cols = [col for col in extra_cols if col in dataset.columns]
            if extra_cols:
                meta_train = meta_train.join(dataset.loc[train_mi, extra_cols], how="left")
                meta_holdout = meta_holdout.join(dataset.loc[val_mi, extra_cols], how="left")

            meta_train = meta_train.fillna(meta_train.mean())
            meta_holdout = meta_holdout.fillna(meta_train.mean())

            y_train = stacking_payloads[top_names[0]]["y_train"]
            y_holdout = stacking_payloads[top_names[0]]["y_holdout"]
            groups = _extract_group_labels(meta_train.index)

            scaler, lr_model, best_c, lr_cv = _tune_logistic_meta(
                meta_train.values.astype(np.float32),
                y_train.values,
                groups,
                candidate_cs=[0.1, 0.3, 1.0, 3.0, 10.0],
            )
            holdout_lr = lr_model.predict_proba(scaler.transform(meta_holdout.values.astype(np.float32)))[:, 1]
            lr_auc = float(roc_auc_score(y_holdout.values, holdout_lr))

            xgb_model, xgb_params, xgb_cv = _tune_xgb_meta(
                meta_train.values.astype(np.float32),
                y_train.values,
                groups,
            )
            holdout_xgb = xgb_model.predict_proba(meta_holdout.values.astype(np.float32))[:, 1]
            xgb_auc = float(roc_auc_score(y_holdout.values, holdout_xgb))

            blend = meta_holdout[base_cols].mean(axis=1)
            blend_auc = float(roc_auc_score(y_holdout.values, blend.values))

            print(f"Blend (top-{len(base_cols)} mean): holdout ROC-AUC {blend_auc:.4f}")
            print(
                f"Stacking (top-{len(base_cols)} LR, C={best_c:.3f}): holdout ROC-AUC {lr_auc:.4f}"
            )
            print(
                "Stacking "
                f"(top-{len(base_cols)} XGB, depth={xgb_params['max_depth']}, "
                f"n_estimators={xgb_params['n_estimators']}): holdout ROC-AUC {xgb_auc:.4f}"
            )

            stacking_info = {
                "blend_auc": blend_auc,
                "lr_auc": lr_auc,
                "lr_C": best_c,
                "lr_cv": lr_cv,
                "xgb_auc": xgb_auc,
                "xgb_params": xgb_params,
                "xgb_cv": xgb_cv,
                "models": base_cols,
            }
        print(f"[Timer] Stacking evaluation completed in {format_seconds(time.time() - stack_start)}")

    best_name, best_metrics = max(results.items(), key=lambda kv: kv[1]["holdout_auc"])
    best_strategy = next(s for s in strategies if s.name == best_name)
    block_metrics = evaluate_block_holdout(
        dataset=dataset,
        strategy=best_strategy,
        features=best_strategy.features,
        quantile=0.8,
    )

    results["best"] = {"name": best_name, **best_metrics, "block": block_metrics}
    if stacking_info:
        results["stacking"] = stacking_info
    return results


def log_results(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    lines: List[str] = []
    for name, metrics in results.items():
        if name in {"best", "stacking"}:
            continue
        lines.append(
            f"{name}: holdout={metrics['holdout_auc']:.4f}, "
            f"cv_mean={metrics['cv_mean']:.4f}, cv_std={metrics['cv_std']:.4f}"
        )

    best = results["best"]
    lines.append(
        f"Best strategy: {best['name']} with holdout ROC-AUC {best['holdout_auc']:.4f}"
    )
    block = best["block"]
    lines.append(
        f"Block validation ({best['name']}): holdout={block['holdout_auc']:.4f}, "
        f"cv_mean={block['cv_mean']:.4f}, cv_std={block['cv_std']:.4f}"
    )

    if "stacking" in results:
        stack = results["stacking"]
        cols = ",".join(stack["models"])
        lines.append(f"Blend (models={cols}): holdout={stack['blend_auc']:.4f}")
        lines.append(
            f"Stacking LR (models={cols}, C={stack['lr_C']:.3f}): holdout={stack['lr_auc']:.4f}"
        )
        lines.append(
            "Stacking XGB (models={cols}, depth={depth}, n_estimators={n_estimators}): holdout={auc:.4f}".format(
                cols=cols,
                depth=stack["xgb_params"]["max_depth"],
                n_estimators=stack["xgb_params"]["n_estimators"],
                auc=stack["xgb_auc"],
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Pipeline] Results saved to {output_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Mindwatch tuning + evaluation pipeline.")
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=120)
    parser.add_argument("--top-k-min", type=int, default=40)
    parser.add_argument("--strategies", nargs="+", default=["hgb_top120", "lgbm_full", "xgb_full"])
    parser.add_argument("--tuning-trials", type=int, default=200)
    parser.add_argument("--run-tuning", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--block-validation", action="store_true")
    parser.add_argument("--exclude-prev-survey", action="store_true")
    parser.add_argument("--study-name", type=str, default="full_pipeline")
    parser.add_argument("--storage", type=str, default="sqlite:///logs/optuna_full_pipeline.db")
    parser.add_argument("--optuna-dir", type=str, default=None, help="Directory containing optuna_*.json files (defaults to --output-dir)")
    parser.add_argument("--output-dir", type=str, default="logs", help="Directory to store Optuna summaries and pipeline results.")
    parser.add_argument("--python-exec", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.python_exec = args.python_exec or sys.executable

    overall_start = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir = Path(args.optuna_dir) if args.optuna_dir else output_dir
    optuna_dir.mkdir(parents=True, exist_ok=True)

    global LOG_DIR
    LOG_DIR = optuna_dir

    storage_uri = args.storage
    default_storage = "sqlite:///logs/optuna_full_pipeline.db"
    if storage_uri == default_storage:
        storage_uri = f"sqlite:///{(optuna_dir / 'optuna_full_pipeline.db').as_posix()}"

    missing_summaries = []
    for strat in args.strategies:
        pattern = f"optuna_{strat}_*.json"
        if not list(optuna_dir.glob(pattern)):
            missing_summaries.append(strat)

    if missing_summaries and not args.run_tuning:
        missing_strats = ", ".join(missing_summaries)
        print(
            "[Pipeline] Missing Optuna summaries for strategies: "
            f"{missing_strats}. Running tuning automatically with {args.tuning_trials} trials."
        )
        args.run_tuning = True

    if args.run_tuning:
        tuning_start = time.time()
        run_optuna_batch(args, storage_uri)
        print(f"[Timer] Tuning stage completed in {format_seconds(time.time() - tuning_start)}")
        backfill_optuna_summaries(args.strategies, optuna_dir)
    else:
        backfill_optuna_summaries(args.strategies, optuna_dir)

    step_start = time.time()
    dataset = build_feature_dataframe(history_hours=args.history_hours)
    print(f"[Timer] Feature dataframe built in {format_seconds(time.time() - step_start)}")

    if args.exclude_prev_survey:
        drop_cols = [
            col
            for col in dataset.columns
            if col.startswith("prev_")
            or col.endswith("_prev_wave")
            or col.endswith("_delta_prev_wave")
        ]
        dataset = dataset.drop(columns=drop_cols, errors="ignore")

    step_start = time.time()
    context = prepare_feature_context(
        dataset,
        split_seed=args.split_seed,
        exclude_prev_survey=args.exclude_prev_survey,
        top_k_features=args.top_k_features,
        top_k_min=args.top_k_min,
    )
    print(f"[Timer] Feature context prepared in {format_seconds(time.time() - step_start)}")

    defaults = {cfg.name: cfg for cfg in build_default_strategies(context)}

    strategy_builders = {
        "hgb_top120": build_hgb_strategy,
        "lgbm_full": build_lgbm_strategy,
        "xgb_full": build_xgb_strategy,
    }

    requested_keys = list(dict.fromkeys(args.strategies))
    tuned_configs: Dict[str, StrategyConfig] = {}
    for key in requested_keys:
        builder = strategy_builders.get(key)
        if builder is None:
            warnings.warn(f"Unknown strategy '{key}' requested; skipping tuned build.")
            continue
        try:
            summary = latest_optuna_summary(key, optuna_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}. Provide optuna_{key}_*.json or run with --run-tuning."
            ) from exc
        tuned_configs[key] = builder(context, summary["best_params"])

    if "hgb_top120" in tuned_configs and "HGB_top120" in defaults:
        tuned_hgb = tuned_configs["hgb_top120"]
        defaults["HGB_top120"] = StrategyConfig(
            name="HGB_top120",
            model_type="hgb",
            features=tuned_hgb.features,
            params=dict(defaults["HGB_top120"].params),
            transform=defaults["HGB_top120"].transform,
            use_sample_weights=defaults["HGB_top120"].use_sample_weights,
        )

    for cfg in tuned_configs.values():
        defaults[cfg.name] = cfg

    strategies_to_eval: List[StrategyConfig] = []
    for name in ["HGB_v1_base", "HGB_v3_quantile", "HGB_top120"]:
        cfg = defaults.get(name)
        if cfg:
            strategies_to_eval.append(cfg)

    if "Voice_HGB" in defaults:
        strategies_to_eval.insert(0, defaults["Voice_HGB"])
    if "HGB_voice_text" in defaults:
        insert_idx = 1 if strategies_to_eval else 0
        strategies_to_eval.insert(insert_idx, defaults["HGB_voice_text"])

    for key in requested_keys:
        tuned_cfg = tuned_configs.get(key)
        if tuned_cfg:
            strategies_to_eval.append(tuned_cfg)

    if not strategies_to_eval:
        raise RuntimeError("No strategies available for evaluation. Check --strategies input.")

    step_start = time.time()
    results = evaluate_strategies(
        dataset,
        context,
        strategies_to_eval,
        args.cv_folds,
        exclude_prev_survey=args.exclude_prev_survey,
    )
    print(f"[Timer] Strategy evaluations finished in {format_seconds(time.time() - step_start)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"full_pipeline_results_{timestamp}.txt"
    log_results(results, output_path)

    print(f"[Timer] Total pipeline runtime {format_seconds(time.time() - overall_start)}")


if __name__ == "__main__":
    main()
