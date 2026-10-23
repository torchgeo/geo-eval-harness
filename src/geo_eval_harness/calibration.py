"""Calibration metrics for classification probes.

Report ECE (L1), RMS calibration error (L2), and maximum calibration error (MCE).
Linear probes also support single-parameter temperature scaling (Guo et al., 2017).
"""

from typing import Literal

import numpy as np
import torch
from torchmetrics.classification import BinaryCalibrationError

_NORMS: tuple[Literal["l1", "l2", "max"], ...] = ("l1", "l2", "max")
_KEYS: dict[str, str] = {"l1": "ece", "l2": "rms_ce", "max": "mce"}


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    multi_label: bool,
    n_bins: int = 15,
    class_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute calibration metrics from class probabilities.

    Args:
        y_true: For single-label, integer class labels of shape ``(N,)``.
            For multi-label, binary indicators of shape ``(N, C)``.
        y_proba: Predicted probabilities of shape ``(N, C)``. For
            multi-label these are per-label sigmoid probabilities (not
            required to sum to 1).
        multi_label: Whether the task is multi-label.
        n_bins: Number of bins used by torchmetrics calibration error.
        class_labels: Single-label class IDs in probability-column order.
            ``None`` means columns already correspond to ``0..C-1``.

    Returns:
        Dict with keys ``ece`` (L1), ``rms_ce`` (L2), ``mce`` (max).
        Multi-label values are macro-averaged over labels with both positive and negative examples.
    """
    probs = torch.as_tensor(y_proba, dtype=torch.float32)

    if multi_label:
        targets = torch.as_tensor(y_true, dtype=torch.long)
        n_classes = probs.shape[1]
        per_class: dict[str, list[float]] = {key: [] for key in _KEYS.values()}
        for c in range(n_classes):
            t_c = targets[:, c]
            if t_c.min() == t_c.max():
                continue
            p_c = probs[:, c].clamp(0.0, 1.0)
            for norm in _NORMS:
                metric = BinaryCalibrationError(n_bins=n_bins, norm=norm)
                per_class[_KEYS[norm]].append(float(metric(p_c, t_c).item()))
        return {
            key: (float(np.mean(vals)) if vals else float("nan")) for key, vals in per_class.items()
        }

    targets = torch.as_tensor(y_true, dtype=torch.long)
    # Float32 rounding can leave probability rows slightly off a sum of one.
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    confidence, predictions = probs.max(dim=1)
    if class_labels is not None:
        classes = torch.as_tensor(class_labels, dtype=torch.long)
        if classes.ndim != 1 or len(classes) != probs.shape[1]:
            raise ValueError("class_labels must name every probability column.")
        predictions = classes[predictions]
    correct = (predictions == targets).long()
    out: dict[str, float] = {}
    for norm in _NORMS:
        metric = BinaryCalibrationError(n_bins=n_bins, norm=norm)
        out[_KEYS[norm]] = float(metric(confidence, correct).item())
    return out


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    multi_label: bool,
    max_iter: int = 100,
    class_labels: np.ndarray | None = None,
) -> float:
    """Fit a single temperature ``T`` by minimizing held-out negative log-likelihood.

    Use ``T = exp(log_T)`` to keep temperature positive with an unconstrained optimizer.
    This follows Guo et al. (2017), "On Calibration of Modern Neural Networks".

    Args:
        logits: Raw scores of shape ``(N, C)``.
        y_true: Integer class labels ``(N,)`` (single-label) or binary
            indicators ``(N, C)`` (multi-label).
        multi_label: Whether to fit T against per-label BCE (sigmoid) or
            multi-class NLL (softmax).
        max_iter: LBFGS iterations.
        class_labels: Single-label class IDs in logit-column order.
            ``None`` means targets already index the logit columns.

    Returns:
        Fitted temperature ``T > 0``. ``T > 1`` means the raw model was
        overconfident; ``T < 1`` underconfident.
    """
    z = torch.as_tensor(logits, dtype=torch.float64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    if multi_label:
        y = torch.as_tensor(y_true, dtype=torch.float64)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(z / log_T.exp(), y)
            loss.backward()
            return loss
    else:
        y = torch.as_tensor(y_true, dtype=torch.long)
        if class_labels is not None:
            classes = torch.as_tensor(class_labels, dtype=torch.long)
            if classes.ndim != 1 or len(classes) != z.shape[1]:
                raise ValueError("class_labels must name every logit column.")
            matches = y[:, None] == classes
            if not matches.any(dim=1).all():
                raise ValueError(
                    "Temperature scaling requires validation classes present in training."
                )
            y = matches.long().argmax(dim=1)
        loss_fn = torch.nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(z / log_T.exp(), y)
            loss.backward()
            return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float, *, multi_label: bool) -> np.ndarray:
    """Convert logits to probabilities at the given temperature."""
    z = torch.as_tensor(logits, dtype=torch.float32) / float(temperature)
    if multi_label:
        return torch.sigmoid(z).numpy()
    return torch.softmax(z, dim=1).numpy()
