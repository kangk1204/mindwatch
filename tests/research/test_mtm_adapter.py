import pytest
pytest.importorskip("torch")
"""Exact-source MTM software checks; cached public code, never cohort data."""
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from sklearn.base import clone

from mindwatch.research.mtm_adapter import (MTM_ACKNOWLEDGEMENT, MTM_HASHES, MTM_REVISION,
                                 OfficialMTMClassifier, load_official_mtm,
                                 verify_mtm_source)
from mindwatch.research.neural import flatten_sequence, past_only_delta


@pytest.fixture
def official_source():
    root = Path(__file__).resolve().parents[1] / ".cache" / "mtm"
    if not (root / "mtm" / "mtm.py").exists():
        pytest.skip("Prepare pinned public MTM source explicitly before integration tests")
    return root


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def packed():
    rng = np.random.default_rng(42)
    y = np.arange(8) % 2
    x = rng.normal(size=(8, 14, 3)).astype(np.float32) + (2*y-1)[:, None, None]
    mask = (rng.uniform(size=x.shape) > 0.4).astype(np.float32)
    mask[:, 0] = 1
    x[mask == 0] = np.nan
    demo = np.column_stack((np.ones(8), np.arange(8) % 2)).astype(np.float32)
    return x, mask, past_only_delta(mask), demo, y


def test_exact_public_source_and_license_are_verified(official_source):
    metadata = verify_mtm_source(official_source)
    assert metadata["revision"] == MTM_REVISION
    assert metadata["sha256"] == MTM_HASHES
    assert metadata["required_publication_acknowledgement"] == MTM_ACKNOWLEDGEMENT
    cls = load_official_mtm(official_source)
    assert cls.__name__ == "MTM"
    assert cls.__module__.startswith("_klosdom_official_mtm_")


def test_modified_or_missing_source_is_never_executed(tmp_path):
    with pytest.raises(ValueError, match="missing/unsafe"):
        verify_mtm_source(tmp_path)
    (tmp_path / "LICENCE.txt").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_mtm_source(tmp_path)


def test_official_adapter_real_training_masking_and_serialization(official_source):
    x, mask, delta, demo, y = packed()
    X = flatten_sequence(x, mask, delta, demo)
    model = OfficialMTMClassifier(source_dir=official_source, sequence_length=14,
                                  n_demographics=2, hidden_dim=8, epochs=2,
                                  dropout=0, batch_size=8, seed=5)
    assert clone(model).get_params() == model.get_params()
    model.fit(X, y, groups=np.arange(8))
    assert np.isfinite([r["train_loss"] for r in model.history_]).all()
    p = model.predict_proba(X)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert not model.provenance_["upstream_source_modified"]
    assert model.provenance_["probability"] == "softmax_two_CE_logits"
    altered = x.copy()
    altered[mask == 0] = 1e6
    np.testing.assert_allclose(p, model.predict_proba(flatten_sequence(altered, mask, delta, demo)))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(p, restored.predict_proba(X))
    repeated = clone(model).fit(X, y, groups=np.arange(8))
    np.testing.assert_array_equal(p, repeated.predict_proba(X))
    mask[0] = 0
    x[0] = np.nan
    absent = model.predict_proba(flatten_sequence(x, mask, delta, demo))
    assert np.isnan(absent[0]).all()
    assert np.isfinite(absent[1:]).all()


def test_official_forward_probability_matches_two_logit_cross_entropy(official_source):
    x, mask, delta, _, y = packed()
    X = flatten_sequence(x, mask, delta)
    model = OfficialMTMClassifier(source_dir=official_source, sequence_length=14,
                                  hidden_dim=8, epochs=1, dropout=0, batch_size=8).fit(X, y)
    batch = model._unpack(X).tensors(torch.device("cpu"))
    with torch.no_grad():
        values, observed, _ = model.model_.prepare(batch["values"], batch["mask"], batch["delta"])
        times = torch.arange(14).expand(8, -1)
        logits = model.model_.upstream(values, ~observed.bool(), times, torch.empty(8, 0))
        expected = torch.softmax(logits, dim=1).numpy()
    np.testing.assert_allclose(expected, model.predict_proba(X))


def test_mtm_validation_groups_and_no_automatic_network(official_source):
    x, mask, delta, _, y = packed()
    X = flatten_sequence(x, mask, delta)
    with pytest.raises(ValueError, match="explicit local source_dir"):
        OfficialMTMClassifier(epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="overlap"):
        OfficialMTMClassifier(source_dir=official_source, epochs=1).fit(
            X, y, groups=np.arange(8), validation_data=(X, y, np.arange(8)))


def test_official_checkpoint_loads_in_fresh_interpreter(official_source, tmp_path):
    import joblib
    x, mask, delta, _, y = packed()
    X = flatten_sequence(x, mask, delta)
    model = OfficialMTMClassifier(source_dir=official_source, hidden_dim=8,
                                  epochs=1, dropout=0).fit(X, y)
    checkpoint, arrays = tmp_path / "synthetic.joblib", tmp_path / "synthetic.npz"
    joblib.dump(model, checkpoint)
    np.savez(arrays, X=X, expected=model.predict_proba(X))
    result = subprocess.run(
        [sys.executable, "-c", "import joblib,numpy as np,sys; "
         "m=joblib.load(sys.argv[1]); a=np.load(sys.argv[2]); "
         "np.testing.assert_allclose(m.predict_proba(a['X']),a['expected'])",
         str(checkpoint), str(arrays)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
