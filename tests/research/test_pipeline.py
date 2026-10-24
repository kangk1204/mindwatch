"""Synthetic loader/inference integration; never reads research datasets."""
import json

import numpy as np
import pandas as pd
import pytest

from mindwatch.research.benchmarks import BenchmarkConfig, finalize, run_development
from mindwatch.research.cli import main
from mindwatch.research.pipeline import load_model_frame
from mindwatch.research.privacy import sha256, write_json, write_parquet


def prepared_synthetic(tmp_path):
    rng = np.random.default_rng(71)
    n = 80
    subjects = np.repeat(np.arange(n), 2)
    sample = [f"synthetic-sample-{i}" for i in range(len(subjects))]
    visits = pd.DataFrame({
        "sample_id": sample, "subject_id": [f"synthetic-person-{i}" for i in subjects],
        "index_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(len(subjects)), unit="D"),
        "y_phq9": np.repeat(np.arange(n) % 2, 2).astype(float),
    })
    split = visits[["sample_id", "subject_id", "index_date"]].copy()
    split["partition"] = np.where(subjects < 60, "development", "temporal_test")
    split.loc[len(split)-1, "partition"] = "excluded_after_cutoff"
    features = pd.DataFrame({
        "sample_id": sample, "sensor__steps__mean": visits.y_phq9 + rng.normal(size=len(visits)),
        "sensor__steps__coverage": 1.0, "quality__sensor_eligible": 1,
        "quality__sensor_observed_days": 14, "demo__group": (subjects % 3 == 0).astype(float),
    })
    # The same matched training cohort as the original implementation.
    features.loc[0, "quality__sensor_eligible"] = 0
    features.loc[120:123, "quality__sensor_eligible"] = 0
    features.loc[120:123, "quality__sensor_observed_days"] = 2
    visits.loc[124, "y_phq9"] = np.nan
    dataset = tmp_path / "synthetic_dataset"
    write_parquet(dataset / "visits.parquet", visits)
    write_parquet(dataset / "features_14d.parquet", features)
    write_parquet(dataset / "split_manifest.parquet", split)
    return dataset, visits, features, split


def cfg(models=("prevalence", "elasticnet")):
    return BenchmarkConfig(models=models, outer_folds=2, inner_folds=2, calibration_fraction=.25,
                           bootstrap_replicates=25, parameter_grids={"elasticnet": [{"C": .1, "l1_ratio": 0.0}]})


def test_development_loader_partitions_before_target_filtering(tmp_path):
    dataset, visits, features, split = prepared_synthetic(tmp_path)
    development, columns = load_model_frame(dataset)
    assert development.partition.eq("development").all()
    assert len(development) == 119
    summary = development.attrs["cohort_eligibility_summary"]
    assert set(summary["partitions"]) == {"development"}
    assert summary["partitions"]["development"]["labelled_sensor_ineligible_visits"] == 1
    changed = visits.copy()
    mask = split.partition.ne("development")
    changed.loc[mask, "y_phq9"] = 98765.0
    write_parquet(dataset / "visits.parquet", changed)
    second, second_columns = load_model_frame(dataset)
    pd.testing.assert_frame_equal(development, second)
    assert columns == second_columns


def test_old_development_fingerprint_is_preserved_and_temporal_coverage_is_full(tmp_path):
    dataset, visits, features, split = prepared_synthetic(tmp_path)
    legacy = visits.merge(features, on="sample_id").merge(split[["sample_id", "partition"]], on="sample_id")
    legacy = legacy.loc[legacy.y_phq9.notna() & legacy.quality__sensor_eligible.eq(1)]
    development, columns = load_model_frame(dataset)
    out = tmp_path / "frozen_legacy_model"
    report = run_development(legacy, columns, "y_phq9", out, cfg())
    assert report["status"] == "selection_frozen"
    original_selection = (out / "selection.json").read_bytes()
    full_frame, _ = load_model_frame(dataset, phase="finalize")
    old_dev = legacy.loc[legacy.partition.eq("development")].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_dev, development.reset_index(drop=True), check_like=True)
    temporal = full_frame.loc[full_frame.partition.eq("temporal_test")]
    assert len(temporal) == 38  # 40, minus excluded row and missing label.
    assert temporal.quality__sensor_eligible.eq(0).sum() == 4
    result = finalize(full_frame, out)
    assert result["metrics"]["n_labelled_visits"] == 38
    assert result["metrics"]["n_predicted_visits"] == 34
    assert result["metrics"]["coverage"] < 1
    predicted = pd.read_parquet(out / "temporal_predictions.parquet")
    assert predicted.status.eq("abstain_study_sensor_eligibility").sum() == 4
    assert original_selection == (out / "selection.json").read_bytes()


def test_demo_comparison_retains_matched_training_and_full_temporal_denominator(tmp_path):
    dataset, _, _, _ = prepared_synthetic(tmp_path)
    development, columns = load_model_frame(dataset, feature_set="demo")
    config = cfg(models=("demographic",))
    config.observation_feature_columns = tuple(columns)
    out = tmp_path / "demo_model"
    assert run_development(development, columns, "y_phq9", out, config)["status"] == "selection_frozen"
    temporal, _ = load_model_frame(dataset, feature_set="demo", phase="finalize")
    final = finalize(temporal, out)
    assert final["metrics"]["n_labelled_visits"] == 38
    assert final["metrics"]["n_predicted_visits"] == 34


def test_speech_loader_requires_verified_untampered_pooling_receipt(tmp_path):
    dataset, visits, _, _ = prepared_synthetic(tmp_path)
    speech = pd.DataFrame({"sample_id": visits.sample_id, "audio__free_speech__pitch": 100.0})
    speech_path = dataset / "speech_features_14d.parquet"
    receipt = dataset / "speech_pooling_14d.json"
    write_parquet(speech_path, speech)
    with pytest.raises(ValueError):
        load_model_frame(dataset, feature_set="sensor+audio")
    write_json(receipt, {"metadata_gates_verified": False, "features_sha256": sha256(speech_path)})
    with pytest.raises(ValueError):
        load_model_frame(dataset, feature_set="sensor+audio")
    write_json(receipt, {"metadata_gates_verified": True, "features_sha256": sha256(speech_path)})
    frame, columns = load_model_frame(dataset, feature_set="sensor+audio")
    assert "audio__free_speech__pitch" in columns
    speech["audio__free_speech__pitch"] = 500.0
    write_parquet(speech_path, speech)
    with pytest.raises(ValueError):
        load_model_frame(dataset, feature_set="sensor+audio")


def test_cli_finalization_gates_before_loader_and_dispatches_sequence(monkeypatch, tmp_path, capsys):
    from mindwatch.research import pipeline, sequence_run
    from mindwatch.research import benchmarks
    called = []
    monkeypatch.setattr(pipeline, "load_model_frame", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Tabular loader must not run")))
    missing = tmp_path / "missing"
    assert main(["finalize", "--dataset", str(tmp_path), "--model-dir", str(missing)]) == 1
    assert json.loads(capsys.readouterr().err)['error_type']=='ValueError'
    selected = tmp_path / "sequence_model"
    write_json(selected / "selection.json", {"sealed_selection": True, "target_column": "y_phq9", "feature_columns": ["sensor__seq_0000"], "config": {"sequence_length": 7}})
    def sequence_loader(dataset, window, target, phase):
        called.append((window, target, phase))
        return pd.DataFrame(), [], 0
    monkeypatch.setattr(sequence_run, "sequence_frame", sequence_loader)
    monkeypatch.setattr(benchmarks, "finalize", lambda frame, output: {"status": "synthetic_test_only"})
    assert main(["finalize", "--dataset", str(tmp_path), "--model-dir", str(selected)]) == 0
    assert called == [(7, "y_phq9", "finalize")]


def test_cli_unsealed_selection_is_blocked_before_any_loader(monkeypatch,tmp_path,capsys):
    from mindwatch.research import pipeline
    called=[]
    def forbidden_loader(*args,**kwargs):
        called.append(True)
        raise AssertionError('A later loader error is not proof of a freeze gate')
    monkeypatch.setattr(pipeline,'load_model_frame',forbidden_loader)
    selected=tmp_path/'unsealed'
    write_json(selected/'selection.json',{'sealed_selection':False,'target_column':'y_phq9',
                                          'feature_columns':['sensor__steps'],'config':{}})
    assert main(['finalize','--dataset',str(tmp_path),'--model-dir',str(selected)])==1
    assert called==[]
    assert json.loads(capsys.readouterr().err)['error_type']=='ValueError'
