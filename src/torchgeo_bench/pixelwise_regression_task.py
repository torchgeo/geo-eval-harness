"""Training and evaluation for pixelwise regression probes."""

import logging
import math
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config.schema import SegmentationConfig
from .segmentation_probe import CachedFeaturesDataset, GPUTensorCache, SegmentationProbe

logger = logging.getLogger(__name__)

RegressionMetrics = dict[str, float]


class PixelwiseRegressionSolver:
    """Train and evaluate a single-output pixelwise regression probe."""

    def __init__(
        self,
        model: SegmentationProbe,
        *,
        target_scale: float,
        lr: float = 1e-3,
        device: str = "cuda",
        lr_scheduler: str = "cosine",
    ) -> None:
        """Configure regression optimization and physical-unit reporting.

        Args:
            model: Frozen-backbone spatial probe.
            target_scale: Factor that restores normalized targets to reported units.
            lr: AdamW learning rate.
            device: Training device.
            lr_scheduler: ``"cosine"`` or ``"none"``.
        """
        self.model = model.to(device)
        self.target_scale = target_scale
        self.device = device
        self.device_type = torch.device(device).type
        self.lr_scheduler_type = lr_scheduler
        self.optimizer = torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.0,
        )
        self.criterion = nn.MSELoss()
        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _make_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Return the configured learning-rate scheduler."""
        if self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        if self.lr_scheduler_type == "none":
            return None
        raise ValueError(
            f"Unknown lr_scheduler {self.lr_scheduler_type!r}. Expected 'cosine' or 'none'."
        )

    def _prepare_outputs(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return matching ``(B, H, W)`` prediction and target tensors."""
        if predictions.ndim != 4 or predictions.shape[1] != 1:
            raise ValueError(
                f"Pixelwise regression expects predictions shaped (B, 1, H, W), got {tuple(predictions.shape)}."
            )
        predictions = predictions.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.to(device=self.device, dtype=torch.float32)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match target shape {tuple(targets.shape)}."
            )
        return predictions, targets

    def _raw_batch(self, batch: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract images and targets from a dense-task batch."""
        if isinstance(batch, dict):
            return batch["image"].to(self.device), batch["mask"]
        assert isinstance(batch, (list, tuple))
        return batch[0].to(self.device), batch[1]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        *,
        verbose: bool = True,
    ) -> float | None:
        """Train from images and optionally return final validation RMSE."""
        scheduler = self._make_scheduler(epochs)
        last_val_rmse: float | None = None
        for epoch in range(epochs):
            self.model.train()
            self.model.backbone.eval()
            batches = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not verbose,
            )
            for batch in batches:
                images, targets = self._raw_batch(batch)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    predictions, targets = self._prepare_outputs(self.model(images), targets)
                    loss: torch.Tensor = self.criterion(predictions, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if scheduler is not None:
                scheduler.step()
            if val_loader is not None:
                metrics, _ = self.evaluate(val_loader)
                last_val_rmse = metrics["RMSE"]
                if verbose:
                    logger.info("Epoch %d Val RMSE: %.17g", epoch + 1, last_val_rmse)
        return last_val_rmse

    def fit_cached(
        self,
        train_cache: CachedFeaturesDataset | GPUTensorCache,
        val_cache: CachedFeaturesDataset | GPUTensorCache | None = None,
        batch_size: int = 64,
        epochs: int = 10,
        *,
        verbose: bool = True,
    ) -> float | None:
        """Train the regression head on cached backbone features."""
        gpu_train = (
            train_cache
            if isinstance(train_cache, GPUTensorCache)
            else GPUTensorCache.from_cached(train_cache, self.device, target_dtype=torch.float32)
        )
        gpu_val = (
            GPUTensorCache.from_cached(val_cache, self.device, target_dtype=torch.float32)
            if isinstance(val_cache, CachedFeaturesDataset)
            else val_cache
        )
        scheduler = self._make_scheduler(epochs)
        input_hw = (gpu_train.masks.shape[-2], gpu_train.masks.shape[-1])
        last_val_rmse: float | None = None
        num_batches = math.ceil(len(gpu_train) / batch_size)

        for epoch in range(epochs):
            self.model.train()
            self.model.backbone.eval()
            batches = tqdm(
                gpu_train.shuffled_batches(batch_size),
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not verbose,
            )
            for features, targets in batches:
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    predictions = self.model.head(features, *input_hw)
                    predictions, targets = self._prepare_outputs(predictions, targets)
                    loss: torch.Tensor = self.criterion(predictions, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if scheduler is not None:
                scheduler.step()
            if gpu_val is not None:
                metrics, _ = self._evaluate(gpu_val, batch_size)
                last_val_rmse = metrics["RMSE"]
                if verbose:
                    logger.info("Epoch %d Val RMSE: %.17g", epoch + 1, last_val_rmse)
        return last_val_rmse

    def _evaluation_batches(
        self, data: DataLoader | GPUTensorCache, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield predictions and targets from images or cached features."""
        if isinstance(data, GPUTensorCache):
            input_hw = (data.masks.shape[-2], data.masks.shape[-1])
            for features, targets in data.ordered_batches(batch_size):
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    yield self._prepare_outputs(self.model.head(features, *input_hw), targets)
            return
        for batch in data:
            images, targets = self._raw_batch(batch)
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                yield self._prepare_outputs(self.model(images), targets)

    @torch.no_grad()
    def _evaluate(
        self, data: DataLoader | GPUTensorCache, batch_size: int = 64
    ) -> tuple[RegressionMetrics, torch.Tensor]:
        """Return mean per-image RMSE and the individual image scores."""
        self.model.eval()
        scores = []
        for predictions, targets in self._evaluation_batches(data, batch_size):
            per_image = (predictions - targets).square().flatten(1).mean(dim=1).sqrt()
            scores.append((per_image * self.target_scale).cpu())
        per_image_rmse = torch.cat(scores)
        return {"RMSE": per_image_rmse.mean().item()}, per_image_rmse

    def evaluate(self, dataloader: DataLoader) -> tuple[RegressionMetrics, torch.Tensor]:
        """Evaluate images using mean per-image RMSE in reported units."""
        return self._evaluate(dataloader)

    def evaluate_cached(
        self, cache: CachedFeaturesDataset, batch_size: int = 64
    ) -> tuple[RegressionMetrics, torch.Tensor]:
        """Evaluate cached features using mean per-image RMSE in reported units."""
        gpu_cache = GPUTensorCache.from_cached(cache, self.device, target_dtype=torch.float32)
        return self._evaluate(gpu_cache, batch_size)


def build_pixelwise_regression_probe_and_solver(
    model: nn.Module,
    config: SegmentationConfig,
    device: torch.device,
    target_scale: float,
) -> tuple[SegmentationProbe, PixelwiseRegressionSolver]:
    """Build a single-output frozen-backbone regression probe and solver."""
    if not config.layers:
        raise ValueError(
            "Pixelwise regression requires segmentation.layers to name spatial backbone layers."
        )
    probe = SegmentationProbe(
        backbone=model,
        layer_names=config.layers,
        num_classes=1,
        head_type=config.head,
        freeze_backbone=True,
        temporal_pool=config.temporal_pool,
    )
    solver = PixelwiseRegressionSolver(
        probe,
        target_scale=target_scale,
        lr=config.learning_rate,
        device=str(device),
        lr_scheduler=config.scheduler,
    )
    return probe, solver
