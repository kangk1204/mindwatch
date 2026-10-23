"""Synthetic calculation tests only; these establish no clinical validity."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mindwatch.research.evaluation import (
    binary_metrics, holm_adjust, paired_cluster_bootstrap,
    participant_weights, select_sensitivity_threshold,
    weighted_calibration_curve,
)


def test_equal_participant_weights_and_weighted_metrics():
    subjects = ["a", "a", "a", "b", "c"]
    y, p = [0, 0, 1, 1, 0], [0.1, 0.7, 0.6, 0.8, 0.2]
    weights = participant_weights(subjects)
    assert weights[:3].sum() == pytest.approx(1)
    assert weights[3] == weights[4] == 1
    metrics = binary_metrics(y, p, subjects)
    assert metrics["auroc"] == pytest.approx(roc_auc_score(y, p, sample_weight=weights))
    assert metrics["auprc"] == pytest.approx(average_precision_score(y, p, sample_weight=weights))
    assert metrics["brier"] == pytest.approx(np.average((np.array(y)-p)**2, weights=weights))


def test_abstention_coverage_and_single_class_are_explicit():
    metrics = binary_metrics([1, 1, 0], [0.9, np.nan, np.nan], ["a", "a", "b"])
    assert metrics["coverage"] == pytest.approx(0.25)
    assert metrics["status"] == "single_class"
    assert metrics["auroc"] is None and metrics["auprc"] is None
    assert metrics["specificity"] is None
    assert binary_metrics([0], [np.nan], ["a"])["status"] == "no_predictions"


def test_sensitivity_operating_point_is_highest_weighted_threshold():
    # Participant a's two positive visits share one unit of positive weight.
    result = select_sensitivity_threshold([1, 1, 1, 0], [0.9, 0.5, 0.7, 0.6], ["a", "a", "b", "c"], target=0.8)
    assert result["threshold"] == 0.5
    assert result["calibration_sensitivity"] == 1
    assert select_sensitivity_threshold([0], [0.7], ["a"])["threshold"] is None
    with pytest.raises(ValueError):
        select_sensitivity_threshold([0, 1], [0.3, 0.7], ["a", "b"], target=0)


def test_vector_fold_specific_thresholds():
    result = binary_metrics([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.6], list("abcd"), [0.3, 0.3, 0.5, 0.5])
    assert result["sensitivity"] == result["specificity"] == 1


def test_cluster_paired_bootstrap_is_paired_and_reproducible():
    subjects = np.repeat(np.arange(16), 2)
    y = np.repeat(np.arange(16) % 2, 2)
    p = np.where(y, 0.8, 0.2)
    first = paired_cluster_bootstrap(y, p, p, subjects, n_bootstrap=100, seed=9)
    second = paired_cluster_bootstrap(y, p, p, subjects, n_bootstrap=100, seed=9)
    assert first == second
    assert first["delta_a_minus_b"] == first["ci_lower"] == first["ci_upper"] == 0
    assert first["p_value"] == 1
    assert first["n_valid"] > 90
    # Repeating every visit within a participant does not change estimates/CIs.
    repeated = paired_cluster_bootstrap(np.repeat(y, 2), np.repeat(p, 2), np.repeat(1-p, 2), np.repeat(subjects, 2), n_bootstrap=100, seed=9)
    original = paired_cluster_bootstrap(y, p, 1-p, subjects, n_bootstrap=100, seed=9)
    assert repeated == original


def test_holm_includes_prespecified_missing_hypothesis():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adjusted = holm_adjust({"a": 0.01, "b": None, "c": 0.03})
    assert adjusted == {"a": 0.03, "b": None, "c": 0.06}


def test_calibration_degeneracy_and_input_validation():
    constant = binary_metrics([0, 1, 0, 1], [0.5]*4, list("abcd"))
    assert constant["calibration_status"] == "constant_predictions"
    separated = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], list("abcd"))
    assert separated["calibration_slope"] is None
    with pytest.raises(ValueError):
        binary_metrics([0, 2], [0.1, 0.8], ["a", "b"])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1, 1.2], ["a", "b"])
    with pytest.raises(ValueError):
        participant_weights(["a", None])


def test_calibration_curve_uses_global_participant_weights():
    bins = weighted_calibration_curve([0, 1, 1], [0.2, 0.2, 0.2], ["a", "a", "b"], n_bins=5)
    assert len(bins) == 1
    assert bins[0]["observed_frequency"] == pytest.approx(0.75)
    assert bins[0]["mean_predicted"] == pytest.approx(0.2)
    assert bins[0]["weight"] == 2
