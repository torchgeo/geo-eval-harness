"""Boundary IoU geometry and metric accumulation."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from geo_eval_harness.boundary_iou import MulticlassBoundaryIoU


@pytest.mark.parametrize(("ratio", "logits"), [(0.02, False), (0.1, True)])
def test_matches_reference_erosion_including_image_edges(ratio: float, *, logits: bool) -> None:
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, :20, 8:24] = 1
    preds = target.roll(2, dims=-1)
    metric = MulticlassBoundaryIoU(3, ignore_index=None, dilation_ratio=ratio)
    inputs = F.one_hot(preds, 3).permute(0, 3, 1, 2).float() if logits else preds
    metric.update(inputs, target)
    width = max(1, round(ratio * np.hypot(32, 32)))
    expected = []
    for label in (0, 1):
        bands = []
        for labels in (target, preds):
            mask = labels.numpy() == label
            eroded = binary_erosion(mask, structure=np.ones((1, 3, 3)), iterations=width)
            bands.append(mask & ~eroded)
        expected.append((bands[0] & bands[1]).sum() / (bands[0] | bands[1]).sum())
    assert metric.boundary_width == width
    assert metric.compute().item() == pytest.approx(np.mean(expected))
    assert metric.union[2] == 0


def test_ignored_predictions_cannot_change_boundaries_or_the_score() -> None:
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 1
    target[:, :10] = 255
    preds = torch.where(target == 255, 0, target).roll(2, dims=-1)
    metric = MulticlassBoundaryIoU(2)
    metric.update(preds, target)
    expected = metric.compute()
    metric.reset()
    preds[target == 255] = 1
    metric.update(preds, target)
    torch.testing.assert_close(metric.compute(), expected)
    metric.reset()
    metric.update(preds, torch.full_like(target, 255))
    assert metric.compute().isnan()


def test_accumulates_batches_and_reset_clears_counts() -> None:
    target = torch.zeros(2, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    preds = target.clone()
    preds[0] = preds[0].roll(2, dims=-1)
    metric = MulticlassBoundaryIoU(2)
    for prediction, labels in zip(preds.split(1), target.split(1), strict=True):
        metric.update(prediction, labels)
    pooled = MulticlassBoundaryIoU(2)
    pooled.update(preds, target)
    torch.testing.assert_close(metric.compute(), pooled.compute())
    metric.reset()
    assert metric.intersection.sum() == metric.union.sum() == 0
    metric.update(target, target)
    assert metric.compute() == 1


@pytest.mark.parametrize("settings", [{"num_classes": 0}, {"num_classes": 2, "dilation_ratio": 0}])
def test_rejects_invalid_settings(settings: dict) -> None:
    with pytest.raises(ValueError, match=r"num_classes.*dilation_ratio"):
        MulticlassBoundaryIoU(**settings)


def test_rejects_mismatched_label_shapes() -> None:
    with pytest.raises(ValueError, match="matching"):
        MulticlassBoundaryIoU(2).update(torch.zeros(1, 8, 8), torch.zeros(1, 4, 4))
