import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


LOG_DIR = Path("logs")
OUTPUT_METRICS = LOG_DIR / "publication_table_metrics.csv"
OUTPUT_MARKDOWN = LOG_DIR / "publication_table_metrics.md"


def parse_line(line: str) -> dict | None:
    pattern = re.compile(
        r"^(?P<name>[\w_]+): holdout=(?P<holdout>0\.\d+), cv_mean=(?P<cv_mean>0\.\d+), cv_std=(?P<cv_std>0\.\d+)"
    )
    match = pattern.match(line.strip())
    if not match:
        return None
    return {
        "Model": match.group("name"),
        "Holdout ROC-AUC": float(match.group("holdout")),
        "CV Mean": float(match.group("cv_mean")),
        "CV Std": float(match.group("cv_std")),
    }


def parse_stacking(line: str) -> tuple[str, float]:
    pattern = re.compile(r"(?:Stacking|Blend) ([^:]+): holdout=(0\.\d+)")
    match = pattern.search(line)
    if not match:
        raise ValueError(f"Unable to parse stacking line: {line}")
    label = match.group(1).strip()
    value = float(match.group(2))
    return label, value


def latest_full_pipeline_log() -> Path:
    files = sorted(LOG_DIR.glob("full_pipeline_results_*.txt"))
    if not files:
        raise FileNotFoundError("No full_pipeline_results_*.txt found in logs directory.")
    return files[-1]


def build_tables(log_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    stacking_rows: list[tuple[str, float]] = []

    for line in log_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_line(line)
        if parsed:
            rows.append(parsed)
            continue
        if line.startswith("Stacking") or line.startswith("Blend"):
            label, value = parse_stacking(line)
            stacking_rows.append((label, value))

    df = pd.DataFrame(rows).sort_values("Holdout ROC-AUC", ascending=False)

    if stacking_rows:
        for label, value in stacking_rows:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{"Model": label, "Holdout ROC-AUC": value, "CV Mean": None, "CV Std": None}]
                    ),
                ],
                ignore_index=True,
            )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready tables from pipeline results.")
    parser.add_argument("--log-path", type=str, default=None)
    args = parser.parse_args()

    log_path = Path(args.log_path) if args.log_path else latest_full_pipeline_log()
    df = build_tables(log_path)

    df.to_csv(OUTPUT_METRICS, index=False)
    markdown = df.to_markdown(index=False, floatfmt=".4f")
    OUTPUT_MARKDOWN.write_text(markdown, encoding="utf-8")

    print(f"Saved metrics table to {OUTPUT_METRICS}")
    print(f"Saved markdown table to {OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
