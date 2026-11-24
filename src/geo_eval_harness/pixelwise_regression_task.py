"""Loss and metrics for pixelwise regression using the shared dense training engine."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config.schema import SegmentationConfig
from .segmentation_probe import GPUTensorCache, SegmentationProbe
from .segmentation_task import DenseSolver, SegMetrics, build_seg_probe_and_solver


class PixelwiseMSE(nn.Module):
    """Mean squared error for a single output channel and float spatial targets."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Validate spatial shapes and compute MSE in float32."""
        if predictions.ndim != 4 or predictions.shape[1] != 1:
            raise ValueError("Pixelwise regression expects predictions shaped (B, 1, H, W).")
        predictions = predictions.squeeze(1).float()
        if predictions.shape != targets.shape:
            raise ValueError("Prediction shape does not match target shape.")
        return nn.functional.mse_loss(predictions, targets.float())


class PixelwiseRegressionSolver(DenseSolver):
    """Train with MSE and select the epoch with lowest physical-unit mean image RMSE."""

    target_dtype = torch.float32
    selection_metric = "RMSE"
    higher_is_better = False

    def __init__(
        self, model: SegmentationProbe, *, target_scale: float, num_classes: int = 1, **kwargs: Any
    ) -> None:
        if num_classes != 1:
            raise ValueError("Pixelwise regression requires one output channel.")
        self.target_scale = target_scale
        super().__init__(model, num_classes=0, criterion=PixelwiseMSE(), **kwargs)

    def init_metrics(self) -> None:
        """Regression metrics are accumulated per evaluation pass."""

    @torch.no_grad()
    def _evaluate(
        self,
        data: DataLoader | GPUTensorCache,
        batch_size: int = 64,
        *,
        collect_confusions: bool = False,
    ) -> SegMetrics | tuple[SegMetrics, torch.Tensor]:
        """Return physical-unit RMSE and optionally per-image bootstrap observations."""
        self.model.eval()
        scores = []
        loss_sum = torch.zeros((), device=self.device)
        pixels = 0
        for predictions, targets in self._evaluation_batches(data, batch_size):
            loss_sum += self.criterion(predictions, targets) * targets.numel()
            pixels += targets.numel()
            rmse = (predictions.squeeze(1).float() - targets).square().flatten(1).mean(1).sqrt()
            scores.append((rmse * self.target_scale).cpu())
        per_image = torch.cat(scores)
        metrics = {"RMSE": per_image.mean().item(), "loss": (loss_sum / pixels).item()}
        return (metrics, per_image) if collect_confusions else metrics


def build_pixelwise_regression_probe_and_solver(
    model: nn.Module,
    config: SegmentationConfig,
    device: torch.device,
    target_scale: float,
) -> tuple[SegmentationProbe, DenseSolver]:
    """Build a one-channel probe with the shared dense configuration."""
    return build_seg_probe_and_solver(model, 1, config, device, target_scale=target_scale)
