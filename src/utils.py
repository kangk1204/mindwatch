"""Shared helpers required by the legacy standalone entry points."""
from pathlib import Path
import os

import yaml


def load_config(path=None):
    location = Path(path or os.environ.get("MINDWATCH_CONFIG", Path(__file__).resolve().parents[1]/"config.yaml"))
    with location.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    for key in ("input_data_dir", "label_data_dir", "voice_data_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError("Missing data directory configuration")
    return config


def get_data_paths(config):
    return {name: Path(config[key]).expanduser().resolve() for name, key in
            (("input", "input_data_dir"), ("label", "label_data_dir"), ("voice", "voice_data_dir"))}


def generate_plots(y_true, y_scores, model_name, output_dir, prefix):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    y, p = np.asarray(y_true), np.asarray(y_scores, float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Aligned finite predictions required")
    if Path(prefix).name != prefix or prefix in ("", ".", ".."):
        raise ValueError("Plot prefix must be a filename component")
    from mindwatch.pipeline import private_output
    directory = private_output(output_dir)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for suffix, display in (("roc", RocCurveDisplay), ("pr", PrecisionRecallDisplay)):
        fig, ax = plt.subplots()
        try:
            display.from_predictions(y, p, name=model_name, ax=ax)
            fig.tight_layout()
            destination = directory/f"{prefix}_{suffix}.png"
            fig.savefig(destination)
            destination.chmod(0o600)
        finally:
            plt.close(fig)
