"""
Visualize model performance from full_pipeline_results_*.txt for reviewer-facing figures.

Outputs (default: plots/):
- model_auc_summary.png : Bar chart of holdout AUC for all reported models (single, stacking, late fusion).
- model_block_vs_holdout.png : Holdout vs. block-validation AUC for the best single model (if available).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def find_latest_results(base_dir: Path) -> Path:
    candidates = list(base_dir.rglob("full_pipeline_results_*.txt"))
    if not candidates:
        raise FileNotFoundError("No full_pipeline_results_*.txt found under results/ or logs/.")
    return sorted(candidates)[-1]


def parse_results(path: Path) -> Dict[str, object]:
    model_entries: List[Dict[str, float]] = []
    stacking_entries: List[Dict[str, float]] = []
    late_fusion: Optional[Dict[str, object]] = None
    block_entry: Optional[Dict[str, float]] = None

    line_re = re.compile(
        r"^(?P<name>[\w/+]+): holdout=(?P<holdout>0\.\d+), cv_mean=(?P<cv_mean>0\.\d+), cv_std=(?P<cv_std>0\.\d+)"
    )
    block_re = re.compile(
        r"^Block validation \((?P<name>.+)\): holdout=(?P<holdout>0\.\d+), cv_mean=(?P<cv_mean>0\.\d+), cv_std=(?P<cv_std>0\.\d+)"
    )
    stack_re = re.compile(r"^(?P<label>(Blend|Stacking [A-Z]+).+): holdout=(?P<holdout>0\.\d+)")
    fusion_re = re.compile(r"^Late fusion \((?P<modal>.+)\): holdout=(?P<holdout>0\.\d+)")

    for line in path.read_text(encoding="utf-8").splitlines():
        m = line_re.match(line.strip())
        if m:
            model_entries.append(
                dict(
                    name=m.group("name"),
                    holdout=float(m.group("holdout")),
                    cv_mean=float(m.group("cv_mean")),
                    cv_std=float(m.group("cv_std")),
                )
            )
            continue
        m = block_re.match(line.strip())
        if m:
            block_entry = dict(
                name=m.group("name"),
                holdout=float(m.group("holdout")),
                cv_mean=float(m.group("cv_mean")),
                cv_std=float(m.group("cv_std")),
            )
            continue
        m = stack_re.match(line.strip())
        if m:
            stacking_entries.append(dict(name=m.group("label"), holdout=float(m.group("holdout"))))
            continue
        m = fusion_re.match(line.strip())
        if m:
            late_fusion = dict(name=f"Late fusion ({m.group('modal')})", holdout=float(m.group("holdout")))

    return dict(models=model_entries, stacking=stacking_entries, fusion=late_fusion, block=block_entry)


def plot_auc_bars(entries: List[Dict[str, float]], output_path: Path) -> None:
    if not entries:
        return
    entries_sorted = sorted(entries, key=lambda x: x["holdout"], reverse=True)
    names = [e["name"] for e in entries_sorted]
    aucs = [e["holdout"] for e in entries_sorted]

    plt.figure(figsize=(max(8, len(names) * 0.6), 5))
    bars = plt.barh(names, aucs, color="#4C72B0")
    plt.xlabel("Holdout ROC-AUC")
    plt.title("Model performance (holdout)")
    plt.gca().invert_yaxis()
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{auc:.3f}", va="center", fontsize=8)
    plt.xlim(0, min(1.0, max(aucs) + 0.05))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_block_vs_holdout(best_name: str, holdout_auc: float, block_entry: Dict[str, float], output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    labels = ["Holdout", "Block"]
    values = [holdout_auc, block_entry["holdout"]]
    colors = ["#4C72B0", "#DD8452"]
    bars = plt.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, min(1.0, max(values) + 0.05))
    plt.ylabel("ROC-AUC")
    plt.title(f"{best_name}: Holdout vs. Block validation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize model results from full_pipeline_results_*.txt")
    parser.add_argument("--results-file", type=str, default=None, help="Path to full_pipeline_results_*.txt")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        # Try results/ then logs/
        base = Path("results") if Path("results").exists() else Path(".")
        results_path = find_latest_results(base)
    parsed = parse_results(results_path)

    # Build combined entries: single models + stacking + fusion
    entries = list(parsed["models"])  # type: ignore[index]
    entries += list(parsed.get("stacking", []))  # type: ignore[arg-type]
    fusion_entry = parsed.get("fusion")
    if fusion_entry:
        entries.append(fusion_entry)  # type: ignore[arg-type]

    summary_path = output_dir / "model_auc_summary.png"
    plot_auc_bars(entries, summary_path)
    print(f"Saved: {summary_path}")

    block_entry = parsed.get("block")
    if block_entry:
        # Best single model assumed as max of model_entries
        if parsed["models"]:  # type: ignore[index]
            best = max(parsed["models"], key=lambda x: x["holdout"])  # type: ignore[index]
            block_path = output_dir / "model_block_vs_holdout.png"
            plot_block_vs_holdout(best["name"], best["holdout"], block_entry, block_path)
            print(f"Saved: {block_path}")
        else:
            print("No single-model entries found to plot block vs holdout.")


if __name__ == "__main__":
    main()
