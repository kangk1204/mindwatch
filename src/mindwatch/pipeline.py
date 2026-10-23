"""Separate development/freeze and fixed-model evaluation stages.

This is a participant-separated retrospective research workflow. Calendar
generalization needs an additional prospective cutoff protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_dataset, build_features, validate_visits
from .models import calibrate, fit_head
from .validation import assert_disjoint, classification_metrics

WINDOWS = (14, 30, 60, 90)
CANDIDATES = ("H", *(f"compact{w}" for w in WINDOWS))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def private_output(path) -> Path:
    """Refuse output under any Git checkout, including through a symlink."""
    result = Path(path).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    if result == package_root or package_root in result.parents:
        raise ValueError("Artifacts must be outside the source checkout")
    if any((parent/".git").exists() for parent in (result, *result.parents)):
        raise ValueError("Artifacts must be outside every Git checkout")
    return result


def _json(path, data):
    # Exclusive creation avoids overwriting previous research artifacts.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def train_and_freeze(train_visits, selection_visits, calibration_visits, ema, output):
    """Do not pass test visits here. Roles are disjoint at participant level."""
    folder = private_output(output)
    if folder.exists():
        raise FileExistsError("Use a new private run directory")
    roles = {"train": validate_visits(train_visits),
             "selection": validate_visits(selection_visits),
             "calibration": validate_visits(calibration_visits)}
    people = {role: tuple(sorted(set(data.subject_id))) for role, data in roles.items()}
    assert_disjoint(*people.values())
    cache = {(role, w): build_dataset(data, ema, w) for role, data in roles.items() for w in WINDOWS}
    for role in roles:
        if not len(cache[role, 14][0]):
            raise ValueError("Each development role needs labeled follow-up samples")
    if cache["train", 14][0].target.nunique() != 2 or cache["selection", 14][0].target.nunique() != 2:
        raise ValueError("Training and selection require both outcome classes")
    heads, calibrators, trace = {}, {}, []
    for candidate in CANDIDATES:
        window = 14 if candidate == "H" else int(candidate.removeprefix("compact"))
        tf, tx = cache["train", window]
        vf, vx = cache["selection", window]
        cf, cx = cache["calibration", window]
        head = fit_head(tx, tf.target, tf.subject_id, "H" if candidate == "H" else "compact")
        scores = classification_metrics(vf.target, head.predict(vx), vf.subject_id)
        trace.append({"candidate": candidate, "window_days": window,
                      "raw_AUROC": scores["AUROC"], "raw_Brier": scores["Brier"]})
        heads[candidate] = head
        calibrators[candidate] = calibrate(cf.target, head.predict(cx), cf.subject_id)
    # Prespecified selection rule; fewer inputs alone do not imply lower complexity.
    chosen = min(trace, key=lambda t: (-t["raw_AUROC"], t["raw_Brier"],
                                      t["candidate"] != "H", t["window_days"]))["candidate"]
    bundle = {"heads": heads, "calibrators": calibrators, "selected": chosen, "people": people}
    folder.mkdir(parents=True, mode=0o700)
    folder.chmod(0o700)
    path = folder/"models.joblib"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        joblib.dump(bundle, stream, compress=3)
    import importlib.metadata
    _json(folder/"freeze.json", {
        "model_sha256": sha256(path), "source_sha256": source_digest(),
        "selected": chosen, "selection": trace, "candidates": list(CANDIDATES),
        "task": "current_phq9_ge10", "test_used_for_selection": False,
        "calendar_validation": False,
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
    })
    return chosen


def load_frozen(folder):
    """Load only your own trusted local artifacts; joblib is not a safe interchange format."""
    folder = private_output(folder)
    freeze = json.loads((folder/"freeze.json").read_text(encoding="utf-8"))
    if sha256(folder/"models.joblib") != freeze["model_sha256"] or source_digest() != freeze["source_sha256"]:
        raise ValueError("Frozen source or artifact checksum changed")
    bundle = joblib.load(folder/"models.joblib")
    if bundle["selected"] != freeze["selected"]:
        raise ValueError("Frozen selection mismatch")
    return bundle


def predict_frozen(folder, visits, ema, candidate=None):
    """Queries may have missing current scores; no current outcomes go to a head."""
    bundle = load_frozen(folder)
    candidate = candidate or bundle["selected"]
    if candidate not in CANDIDATES:
        raise ValueError("Unknown candidate")
    window = 14 if candidate == "H" else int(candidate.removeprefix("compact"))
    metadata, x = build_features(visits, ema, window)
    raw = bundle["heads"][candidate].predict(x)
    ca = bundle["calibrators"][candidate]
    out = metadata.copy()
    out["candidate"] = candidate
    out["raw_prediction"] = raw
    out["prediction"] = ca.predict(raw)
    out["threshold"] = ca.threshold
    out["calibration_status"] = ca.status
    return out


def evaluate_frozen(folder, test_visits, ema):
    """Evaluate fixed candidates and the frozen selection without any refitting."""
    folder = private_output(folder)
    bundle = load_frozen(folder)
    test = validate_visits(test_visits)
    assert_disjoint(set(test.subject_id), *(bundle["people"].values()))
    if (folder/"evaluation.json").exists() or (folder/"predictions.csv").exists():
        raise FileExistsError("Evaluation already exists; do not overwrite it")
    rows, metrics = [], []
    for candidate in CANDIDATES:
        window = 14 if candidate == "H" else int(candidate.removeprefix("compact"))
        metadata, x = build_dataset(test, ema, window)
        if not len(metadata):
            raise ValueError("No eligible labeled follow-up samples")
        raw = bundle["heads"][candidate].predict(x)
        ca = bundle["calibrators"][candidate]
        z = metadata.copy()
        z["candidate"], z["raw_prediction"], z["prediction"] = candidate, raw, ca.predict(raw)
        z["selected_before_test"] = candidate == bundle["selected"]
        rows.append(z)
        metrics.append({"candidate": candidate, "selected_before_test": candidate == bundle["selected"],
                        "raw": classification_metrics(z.target, raw, z.subject_id),
                        "calibrated": classification_metrics(z.target, z.prediction, z.subject_id, ca.threshold),
                        "calibration_status": ca.status})
    path = folder/"predictions.csv"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        pd.concat(rows, ignore_index=True).to_csv(stream, index=False)
    _json(folder/"evaluation.json", {"metrics": metrics, "freeze_sha256": sha256(folder/"freeze.json"),
                                    "predictions_sha256": sha256(path), "refit": False,
                                    "scope": "participant-separated retrospective evaluation; independence depends on prior access"})
    return metrics
