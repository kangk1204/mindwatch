import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional

import optuna

from run_tabular_models import (
    StrategyConfig,
    build_feature_dataframe,
    evaluate_block_holdout,
    evaluate_strategy,
    prepare_feature_context,
)


def build_hgb_top120(context: Dict[str, object], use_gpu: bool = False, trial: Optional[optuna.trial.Trial] = None, params: Optional[Dict[str, float]] = None) -> StrategyConfig:
    feature_names = context["top_feature_names"]  # type: ignore[index]
    param_keys = [
        "learning_rate",
        "max_leaf_nodes",
        "max_depth",
        "max_iter",
        "l2_regularization",
        "min_samples_leaf",
        "max_bins",
    ]
    if trial is not None:
        config_params = {
            "loss": "log_loss",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 511, step=8),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "max_iter": trial.suggest_int("max_iter", 150, 900, step=25),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-5, 1.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
            "max_bins": trial.suggest_int("max_bins", 64, 255),
            "class_weight": "balanced",
            "random_state": 42,
        }
        use_weights = trial.suggest_categorical("use_sample_weights", [True, False])
        use_quantile = trial.suggest_categorical("use_quantile_transform", [False, True])
    elif params is not None:
        config_params = {
            "loss": "log_loss",
            "class_weight": "balanced",
            "random_state": 42,
        }
        for key in param_keys:
            config_params[key] = params[key]
        use_weights = bool(params.get("use_sample_weights", False))
        use_quantile = bool(params.get("use_quantile_transform", False))
    else:
        raise ValueError("Either trial or params must be provided")

    transform = "quantile" if use_quantile else None
    return StrategyConfig(
        name="HGB_top120_optuna",
        model_type="hgb",
        features=feature_names,  # type: ignore[arg-type]
        params=config_params,
        transform=transform,
        use_sample_weights=use_weights,
    )


def build_lgbm_full(context: Dict[str, object], use_gpu: bool = False, trial: Optional[optuna.trial.Trial] = None, params: Optional[Dict[str, float]] = None) -> StrategyConfig:
    feature_names = context["all_features"]  # type: ignore[index]
    param_keys = [
        "learning_rate",
        "num_leaves",
        "subsample",
        "colsample_bytree",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "n_estimators",
    ]
    base_params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        random_state=42,
        n_jobs=-1,
    )
    if use_gpu:
        base_params.update(dict(device="gpu"))
    if trial is not None:
        config_params = dict(base_params)
        config_params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            n_estimators=trial.suggest_int("n_estimators", 400, 2000),
        )
        use_weights = trial.suggest_categorical("use_sample_weights", [True, False])
    elif params is not None:
        config_params = dict(base_params)
        for key in param_keys:
            config_params[key] = params[key]
        use_weights = bool(params.get("use_sample_weights", True))
    else:
        raise ValueError("Either trial or params must be provided")

    return StrategyConfig(
        name="LGBM_full_optuna",
        model_type="lgbm",
        features=feature_names,  # type: ignore[arg-type]
        params=config_params,
        transform=None,
        use_sample_weights=use_weights,
    )


def build_xgb_full(context: Dict[str, object], use_gpu: bool = False, trial: Optional[optuna.trial.Trial] = None, params: Optional[Dict[str, float]] = None) -> StrategyConfig:
    feature_names = context["all_features"]  # type: ignore[index]
    param_keys = [
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "n_estimators",
        "gamma",
    ]
    base_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    if use_gpu:
        base_params.update(dict(tree_method="gpu_hist", predictor="gpu_predictor"))
    if trial is not None:
        config_params = dict(base_params)
        config_params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            n_estimators=trial.suggest_int("n_estimators", 400, 2000),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 4.0),
        )
        use_weights = trial.suggest_categorical("use_sample_weights", [True, False])
    elif params is not None:
        config_params = dict(base_params)
        for key in param_keys:
            config_params[key] = params[key]
        config_params["scale_pos_weight"] = params.get("scale_pos_weight", 1.0)
        use_weights = bool(params.get("use_sample_weights", True))
    else:
        raise ValueError("Either trial or params must be provided")

    return StrategyConfig(
        name="XGB_full_optuna",
        model_type="xgb",
        features=feature_names,  # type: ignore[arg-type]
        params=config_params,
        transform=None,
        use_sample_weights=use_weights,
    )


STRATEGY_BUILDERS = {
    "hgb_top120": build_hgb_top120,
    "lgbm_full": build_lgbm_full,
    "xgb_full": build_xgb_full,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for tabular boosters.")
    parser.add_argument("--history-hours", type=int, default=240, help="Sensor history window used for feature building.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of StratifiedGroupKFold splits.")
    parser.add_argument("--strategy", choices=list(STRATEGY_BUILDERS.keys()), default="hgb_top120", help="Model/feature template to tune.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) for the study.")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI (e.g., sqlite:///optuna.db).")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU-accelerated training when supported (XGB/LGBM/CatBoost).")
    parser.add_argument("--exclude-prev-survey", action="store_true", help="Drop prev_* survey features before tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed for Optuna.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/validation split.")
    parser.add_argument("--block-validation", action="store_true", help="Evaluate best trial on time-block holdout.")
    args = parser.parse_args()

    if args.direction != "maximize":
        raise ValueError("Direction must be 'maximize' for ROC-AUC tuning.")

    dataset = build_feature_dataframe(history_hours=args.history_hours)
    context = prepare_feature_context(
        dataset,
        split_seed=args.split_seed,
        exclude_prev_survey=args.exclude_prev_survey,
    )
    train_idx = context["train_idx"]  # type: ignore[index]
    val_idx = context["val_idx"]  # type: ignore[index]

    builder = STRATEGY_BUILDERS[args.strategy]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None and args.study_name is not None,
        sampler=sampler,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        strategy = builder(context, use_gpu=args.use_gpu, trial=trial)
        metrics, _ = evaluate_strategy(
            dataset=dataset,
            features=strategy.features,
            strategy=strategy,
            train_index=train_idx,
            val_index=val_idx,
            cv_splits=args.cv_folds,
            return_predictions=False,
        )
        trial.set_user_attr("holdout_auc", metrics["holdout_auc"])
        trial.set_user_attr("cv_mean", metrics["cv_mean"])
        trial.set_user_attr("cv_std", metrics["cv_std"])
        return metrics["holdout_auc"]

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_strategy = builder(context, use_gpu=args.use_gpu, params=best_trial.params)
    best_metrics, _ = evaluate_strategy(
        dataset=dataset,
        features=best_strategy.features,
        strategy=best_strategy,
        train_index=train_idx,
        val_index=val_idx,
        cv_splits=args.cv_folds,
        return_predictions=False,
    )

    print("\nBest trial:")
    print(f"  Trial #{best_trial.number}")
    print(f"  Holdout ROC-AUC: {best_metrics['holdout_auc']:.4f}")
    print(f"  CV mean ± std : {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
    print(f"  Params: {json.dumps(best_trial.params, indent=2)}")

    block_metrics = None
    if args.block_validation:
        block_metrics = evaluate_block_holdout(
            dataset=dataset,
            strategy=best_strategy,
            features=best_strategy.features,
            quantile=0.8,
        )
        print(
            f"Time block holdout: ROC-AUC {block_metrics['holdout_auc']:.4f} | "
            f"CV {block_metrics['cv_mean']:.4f} ± {block_metrics['cv_std']:.4f}"
        )

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("logs", f"optuna_{args.strategy}_{timestamp}.json")
    payload = {
        "strategy": args.strategy,
        "history_hours": args.history_hours,
        "cv_folds": args.cv_folds,
        "trials": args.trials,
        "best_trial_number": best_trial.number,
        "best_params": best_trial.params,
        "best_metrics": best_metrics,
        "block_metrics": block_metrics,
    }
    with open(output_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\nSaved study summary to {output_path}")


if __name__ == "__main__":
    main()
