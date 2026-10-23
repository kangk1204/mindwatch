"""Private local execution. This command does not upload data or telemetry."""
import argparse
from pathlib import Path

import pandas as pd

from .pipeline import evaluate_frozen, private_output, train_and_freeze


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        # Preserve leading zeroes in participant/sample keys.
        return pd.read_csv(path, dtype={"subject_id": "string", "sample_id": "string"})
    raise ValueError("Use a local CSV or Parquet input")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    train = stages.add_parser("train")
    for role in ("train", "selection", "calibration"):
        train.add_argument(f"--{role}-visits", required=True)
    train.add_argument("--ema", required=True)
    train.add_argument("--output", required=True)
    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--test-visits", required=True)
    evaluate.add_argument("--ema", required=True)
    evaluate.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    # Reject unsafe destination before loading any private inputs.
    private_output(args.output if args.stage == "train" else args.run)
    if args.stage == "train":
        train_and_freeze(*(read_table(getattr(args, f"{role}_visits")) for role in ("train", "selection", "calibration")),
                         read_table(args.ema), args.output)
    else:
        evaluate_frozen(args.run, read_table(args.test_visits), read_table(args.ema))
    print("Completed locally. Artifacts remain in the specified private directory.")


if __name__ == "__main__":
    main()
