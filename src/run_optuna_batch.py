import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: List[str]) -> int:
    print(f"\n===== Running: {' '.join(cmd)} =====\n", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}", file=sys.stderr)
    return proc.returncode


def build_strategy_command(args: argparse.Namespace, strategy: str) -> List[str]:
    base = [
        sys.executable,
        str(Path(__file__).parent / "tune_tabular_models.py"),
        "--strategy",
        strategy,
        "--history-hours",
        str(args.history_hours),
        "--cv-folds",
        str(args.cv_folds),
        "--trials",
        str(args.trials),
        "--split-seed",
        str(args.split_seed),
        "--top-k-features",
        str(args.top_k_features),
    ]
    if args.exclude_prev_survey:
        base.append("--exclude-prev-survey")
    if args.use_gpu:
        base.append("--use-gpu")
    if args.block_validation:
        base.append("--block-validation")
    if args.top_k_min is not None:
        base.extend(["--top-k-min", str(args.top_k_min)])
    if args.storage:
        base.extend(["--storage", args.storage])
    if args.study_name:
        base.extend(["--study-name", f"{args.study_name}_{strategy}"])
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequentially run Optuna tuning for multiple strategies.")
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["hgb_top120", "lgbm_full", "xgb_full"],
        help="List of strategy keys to tune sequentially.",
    )
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=120)
    parser.add_argument("--top-k-min", type=int, default=None)
    parser.add_argument("--exclude-prev-survey", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--block-validation", action="store_true")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI (applied to every run).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Base study name; strategy suffix is appended automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for strategy in args.strategies:
        cmd = build_strategy_command(args, strategy)
        code = run_command(cmd)
        if code != 0:
            print(f"Stopping batch because {strategy} tuning failed.", file=sys.stderr)
            sys.exit(code)
    print("\nAll strategy tuning runs completed successfully.")


if __name__ == "__main__":
    main()
