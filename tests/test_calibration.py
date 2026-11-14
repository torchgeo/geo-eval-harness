"""Tests for calibration metrics helper."""

import numpy as np
import pytest
import torch
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError

from geo_eval_harness.calibration import (
    apply_temperature,
    compute_calibration_metrics,
    fit_temperature,
)


def test_perfect_calibration_singlelabel():
    rng = np.random.default_rng(0)
    n, c = 200, 4
    y_true = rng.integers(0, c, size=n)
    y_proba = np.eye(c, dtype=np.float32)[y_true]
    out = compute_calibration_metrics(y_true, y_proba, multi_label=False)
    assert set(out) == {"ece", "rms_ce", "mce"}
    for v in out.values():
        assert v == pytest.approx(0.0, abs=1e-5)


def test_worst_calibration_singlelabel():
    """Certain but wrong predictions have the largest calibration error."""
    n, c = 200, 4
    y_true = np.zeros(n, dtype=np.int64)
    y_proba = np.zeros((n, c), dtype=np.float32)
    y_proba[:, 1] = 1.0
    out = compute_calibration_metrics(y_true, y_proba, multi_label=False)
    assert out["ece"] == pytest.approx(1.0, abs=1e-4)
    assert out["mce"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_singlelabel_calibration_matches_multiclass_reference(n_classes: int) -> None:
    rng = np.random.default_rng(8)
    targets = rng.integers(n_classes, size=100)
    probabilities = rng.dirichlet(np.ones(n_classes), size=100).astype(np.float32)
    actual = compute_calibration_metrics(targets, probabilities, multi_label=False)
    tensor = torch.from_numpy(probabilities)
    tensor = tensor / tensor.sum(dim=1, keepdim=True)
    for key, norm in (("ece", "l1"), ("rms_ce", "l2"), ("mce", "max")):
        reference = MulticlassCalibrationError(n_classes, n_bins=15, norm=norm)(
            tensor, torch.from_numpy(targets)
        )
        assert actual[key] == pytest.approx(reference.item())


def test_calibration_respects_probability_column_labels() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    expected = compute_calibration_metrics(np.array([0, 1, 0]), probabilities, multi_label=False)
    actual = compute_calibration_metrics(
        np.array([7, 2, 7]),
        probabilities,
        multi_label=False,
        class_labels=np.array([7, 2]),
    )
    assert actual == pytest.approx(expected)


def test_calibration_handles_a_single_trained_class_and_an_unseen_target() -> None:
    actual = compute_calibration_metrics(
        np.array([2, 7]),
        np.ones((2, 1), dtype=np.float32),
        multi_label=False,
        class_labels=np.array([2]),
    )
    assert actual == pytest.approx({"ece": 0.5, "rms_ce": 0.5, "mce": 0.5})


def test_temperature_respects_logit_column_labels() -> None:
    logits = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]])
    expected = fit_temperature(logits, np.array([0, 1, 0, 1]), multi_label=False)
    actual = fit_temperature(
        logits,
        np.array([7, 2, 7, 2]),
        multi_label=False,
        class_labels=np.array([7, 2]),
    )
    assert actual == pytest.approx(expected)


def test_temperature_rejects_unseen_validation_classes() -> None:
    with pytest.raises(ValueError, match="validation classes present in training"):
        fit_temperature(
            np.array([[1.0, -1.0]]),
            np.array([9]),
            multi_label=False,
            class_labels=np.array([2, 7]),
        )


@pytest.mark.parametrize("n_bins", [2, 15])
def test_multilabel_macro_average_excludes_constant_labels(n_bins: int) -> None:
    targets = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    probabilities = torch.tensor(
        [[0.1, 0.8, 1.0, 0.0], [0.6, 0.3, 1.0, 0.0], [0.2, 0.9, 1.0, 0.0], [0.4, 0.6, 1.0, 0.0]]
    )
    out = compute_calibration_metrics(
        targets.numpy(), probabilities.numpy(), multi_label=True, n_bins=n_bins
    )
    for key, norm in (("ece", "l1"), ("rms_ce", "l2"), ("mce", "max")):
        expected = np.mean(
            [
                BinaryCalibrationError(n_bins=n_bins, norm=norm)(
                    probabilities[:, column], targets[:, column]
                ).item()
                for column in (0, 1)
            ]
        )
        assert out[key] == pytest.approx(expected)


def test_multilabel_without_varying_labels_reports_undefined_calibration() -> None:
    targets = np.array([[0, 1], [0, 1]])
    out = compute_calibration_metrics(targets, targets.astype(np.float32), multi_label=True)
    assert set(out) == {"ece", "rms_ce", "mce"}
    assert all(np.isnan(value) for value in out.values())


def test_multilabel_perfect_calibration():
    rng = np.random.default_rng(1)
    n, c = 80, 3
    y_true = rng.integers(0, 2, size=(n, c))
    # Ensure each column has both classes so no column is skipped.
    y_true[0] = 0
    y_true[1] = 1
    y_proba = y_true.astype(np.float32)
    out = compute_calibration_metrics(y_true, y_proba, multi_label=True)
    for v in out.values():
        assert v == pytest.approx(0.0, abs=1e-5)


def test_temperature_overconfident_singlelabel():
    """Increasing temperature should soften overconfident predictions."""
    rng = np.random.default_rng(0)
    n, c = 1000, 4
    y_true = rng.integers(0, c, size=n)
    # Half the predictions are wrong despite near-certain probabilities.
    pred = y_true.copy()
    flip = rng.choice(n, size=n // 2, replace=False)
    pred[flip] = (y_true[flip] + 1) % c
    logits = np.full((n, c), -5.0, dtype=np.float32)
    logits[np.arange(n), pred] = 5.0
    t = fit_temperature(logits, y_true, multi_label=False)
    assert t > 1.5


def test_temperature_underconfident_singlelabel():
    """Correct but low-confidence predictions need a temperature below 1."""
    rng = np.random.default_rng(0)
    n, c = 500, 4
    y_true = rng.integers(0, c, size=n)
    logits = np.full((n, c), -0.5, dtype=np.float32)
    logits[np.arange(n), y_true] = 0.5
    t = fit_temperature(logits, y_true, multi_label=False)
    assert t < 1.0


def test_temperature_scaling_reduces_ece():
    """Fit and evaluate temperature on the same split for this calibration check."""
    rng = np.random.default_rng(0)
    n, c = 1000, 4
    y_true = rng.integers(0, c, size=n)
    logits = np.full((n, c), -5.0, dtype=np.float32)
    logits[np.arange(n), y_true] = 5.0
    # Inject some errors so calibration isn't trivially zero.
    flip = rng.choice(n, size=300, replace=False)
    wrong = (y_true[flip] + 1) % c
    logits[flip, y_true[flip]] = -5.0
    logits[flip, wrong] = 5.0
    raw_probs = apply_temperature(logits, 1.0, multi_label=False)
    raw_cal = compute_calibration_metrics(y_true, raw_probs, multi_label=False)
    t = fit_temperature(logits, y_true, multi_label=False)
    ts_probs = apply_temperature(logits, t, multi_label=False)
    ts_cal = compute_calibration_metrics(y_true, ts_probs, multi_label=False)
    assert ts_cal["ece"] < raw_cal["ece"]


def test_temperature_multilabel_reduces_negative_log_likelihood() -> None:
    rng = np.random.default_rng(0)
    n, c = 200, 5
    y_true = rng.integers(0, 2, size=(n, c))
    logits = rng.normal(size=(n, c)).astype(np.float32) * 3.0
    t = fit_temperature(logits, y_true, multi_label=True)
    assert t > 1
    probs = apply_temperature(logits, t, multi_label=True)
    raw_loss = np.mean(np.logaddexp(0, logits) - y_true * logits)
    scaled_loss = np.mean(np.logaddexp(0, logits / t) - y_true * logits / t)
    assert scaled_loss < raw_loss
    np.testing.assert_allclose(probs, 1 / (1 + np.exp(-logits / t)), rtol=1e-6)


@pytest.mark.parametrize("multi_label", [False, True])
def test_apply_temperature_matches_probability_definition(*, multi_label: bool) -> None:
    logits = np.array([[0.0, 2.0], [-4.0, 1.0]], dtype=np.float32)
    scaled = logits / 2.0
    expected = (
        1 / (1 + np.exp(-scaled))
        if multi_label
        else np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
    )
    np.testing.assert_allclose(
        apply_temperature(logits, 2.0, multi_label=multi_label), expected, rtol=1e-6
    )
