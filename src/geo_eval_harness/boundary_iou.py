"""Boundary IoU from Cheng et al. (CVPR 2021), https://arxiv.org/abs/2103.16562."""

import math

import torch
import torch.nn.functional as F
from torchmetrics import Metric


def _square_pool(mask: torch.Tensor, width: int, *, erode: bool) -> torch.Tensor:
    """Erode or dilate a binary mask, treating the image exterior as background."""
    sign = -1.0 if erode else 1.0
    padded = sign * F.pad(mask.unsqueeze(1).float(), (width,) * 4)
    kernel = 2 * width + 1
    # Separable pooling avoids a quadratic kernel cost at large image sizes.
    pooled = F.max_pool2d(padded, kernel_size=(kernel, 1), stride=1)
    pooled = F.max_pool2d(pooled, kernel_size=(1, kernel), stride=1)
    return (sign * pooled).squeeze(1) > 0.5


class MulticlassBoundaryIoU(Metric):
    """Mean per-class IoU of the bands inside predicted and target contours.

    Bands span 2% of the image diagonal, rounded to at least one pixel. Pixels within that
    distance of an ignored target are excluded because their contours depend on missing labels.
    Intersections and unions accumulate over the dataset; absent classes do not affect the mean.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Target value excluded from evaluation, or None.
        dilation_ratio: Boundary width as a fraction of the image diagonal.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    intersection: torch.Tensor
    union: torch.Tensor

    def __init__(
        self, num_classes: int, ignore_index: int | None = 255, dilation_ratio: float = 0.02
    ) -> None:
        super().__init__()
        if num_classes < 1 or not 0 < dilation_ratio <= 1:
            raise ValueError("num_classes must be positive and dilation_ratio must be in (0, 1].")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dilation_ratio = dilation_ratio
        self.boundary_width = 0
        self.add_state(
            "intersection", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "union", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate logits or label predictions against ``(N, H, W)`` targets."""
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)
        if target.ndim != 3 or preds.shape != target.shape:
            raise ValueError("Predictions and targets must have matching (N, H, W) shapes.")
        width = max(1, round(self.dilation_ratio * math.hypot(*target.shape[-2:])))
        self.boundary_width = width
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = ~_square_pool(target == self.ignore_index, width, erode=False)
        present = torch.unique(torch.cat((preds.flatten(), target.flatten())))
        for label in present.tolist():
            if not 0 <= label < self.num_classes:
                continue
            predicted = preds == label
            expected = target == label
            predicted = predicted & ~_square_pool(predicted, width, erode=True) & valid
            expected = expected & ~_square_pool(expected, width, erode=True) & valid
            self.intersection[label] += (predicted & expected).sum()
            self.union[label] += (predicted | expected).sum()

    def compute(self) -> torch.Tensor:
        """Return dataset-level mean Boundary IoU, excluding classes with an empty union."""
        return torch.nanmean(self.intersection.double() / self.union)
