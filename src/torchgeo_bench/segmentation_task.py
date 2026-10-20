"""Training and evaluation for segmentation probes."""

import copy
import logging
import math
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm.auto import tqdm

from .config.schema import EarlyStoppingConfig, SegmentationConfig
from .segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
)

logger = logging.getLogger(__name__)

SegMetrics = dict[str, float]


class ValidationStopping:
    """Decide when validation mIoU has stopped improving.

    Patience resets only after a gain of more than ``min_delta`` over the last reference score, so a
    long run of tiny improvements cannot postpone stopping indefinitely. The earliest strict maximum
    is the checkpoint to keep.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Start with no observed checks."""
        self.config = config
        self.best_miou: float | None = None
        self.best_epoch: int | None = None
        self.reference: float | None = None
        self.stale_checks = 0

    def observe(self, epoch: int, miou: float) -> tuple[bool, bool]:
        """Record one check and return whether it is a new best and whether to stop."""
        improved = self.best_miou is None or miou > self.best_miou
        if improved:
            self.best_miou, self.best_epoch = miou, epoch
        if self.reference is None or miou > self.reference + self.config.min_delta:
            self.reference, self.stale_checks = miou, 0
        else:
            self.stale_checks += 1
        stop = epoch >= self.config.min_epochs and self.stale_checks >= self.config.patience
        return improved, stop


class SegmentationSolver:
    """Train and evaluate a segmentation probe."""

    def __init__(  # noqa: PLR0913 -- Public constructor options.
        self,
        model: SegmentationProbe,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cuda",
        criterion: nn.Module | None = None,
        lr_scheduler: str = "cosine",
        ignore_index: int = 255,
    ) -> None:
        """Configure the optimizer, loss, and segmentation metrics.

        Args:
            model: The SegmentationProbe model to train.
            num_classes: Number of segmentation classes.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            device: Device to run training on ('cuda' or 'cpu').
            criterion: Loss module. Defaults to CrossEntropyLoss with ignore_index.
            lr_scheduler: LR schedule: "cosine" (CosineAnnealingLR) or "none" (constant LR).
            ignore_index: Label value to ignore in loss and metrics (default: 255).
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler
        self.val_history: list[float] = []
        self.lr_history: list[dict[str, float]] = []
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None

        self.ignore_index = ignore_index
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.criterion = (
            criterion
            if criterion is not None
            else nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        )

        self.metric = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_fw_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="weighted",
        )
        self.metric_precision = MulticlassPrecision(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_recall = MulticlassRecall(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_f1 = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self._all_metrics = [
            self.metric,
            self.metric_fw_iou,
            self.metric_precision,
            self.metric_recall,
            self.metric_f1,
        ]

        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.device_type = torch.device(device).type

    def _make_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Return a CosineAnnealingLR scheduler, or None for constant LR."""
        if self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        if self.lr_scheduler_type == "none":
            return None
        raise ValueError(
            f"Unknown lr_scheduler {self.lr_scheduler_type!r}. Expected 'cosine' or 'none'."
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        *,
        verbose: bool = True,
    ) -> float | None:
        """Train the segmentation probe.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader for per-epoch mIoU logging.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.

        Returns:
            Val mIoU from the final epoch if val_loader is given, else None.
        """
        scheduler = self._make_scheduler(epochs)
        last_val_miou: float | None = None
        self.val_history = []

        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = tqdm(train_loader, desc=desc, disable=not verbose)
            for batch in batches:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device).long()
                else:
                    images, masks = batch[0].to(self.device), batch[1].to(self.device).long()

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss: torch.Tensor = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if scheduler is not None:
                scheduler.step()

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                assert isinstance(val_metrics, dict)
                last_val_miou = val_metrics["mIoU"]
                self.val_history.append(last_val_miou)
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch + 1, last_val_miou)

        return last_val_miou

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        collect_confusions: bool = False,
    ) -> SegMetrics | tuple[SegMetrics, torch.Tensor]:
        """Evaluate the model on a dataloader and return segmentation metrics.

        Args:
            dataloader: Evaluation data loader.
            collect_confusions: If True, also return per-image confusion matrices.

        Returns:
            Metrics, optionally followed by per-image confusion matrices.
        """
        return self._evaluate(
            dataloader,
            collect_confusions=collect_confusions,
        )

    def fit_cached(  # noqa: PLR0913 -- Public training options.
        self,
        train_cache: "CachedFeaturesDataset | GPUTensorCache",
        val_cache: "CachedFeaturesDataset | GPUTensorCache | None" = None,
        batch_size: int = 64,
        epochs: int = 10,
        *,
        verbose: bool = True,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> float | None:
        """Train the segmentation head on cached backbone features.

        Cached features go directly to ``self.model.head``; the backbone is not called.

        Keep the full cache on the training device to avoid per-batch transfers and stacking.
        Pass a ``GPUTensorCache`` as ``train_cache`` or ``val_cache`` to reuse it across calls.

        Args:
            train_cache: Pre-extracted training features from
                :meth:`SegmentationProbe.extract_segmentation_features`, or a GPU cache.
            val_cache: Optional validation cache for per-epoch mIoU logging. Required for
                early stopping.
            batch_size: Batch size for iterating over cached data.
            epochs: Number of training epochs. Ignored when early stopping is enabled, which
                uses ``early_stopping.max_epochs`` instead.
            verbose: Whether to show progress bars and epoch logs.
            early_stopping: When enabled, stop once validation mIoU stops improving and
                restore the head weights from the best validation check.

        Returns:
            Validation mIoU of the weights left in the head when val_cache is given (the final
            epoch, or the best check with early stopping), otherwise None.
        """
        gpu_train = self._device_cache(train_cache)
        gpu_val = None if val_cache is None else self._device_cache(val_cache)
        if early_stopping is not None and early_stopping.enabled:
            if gpu_val is None:
                raise ValueError("Early stopping requires a validation cache.")
            if self.lr_scheduler_type != "none":
                raise ValueError("Early stopping requires lr_scheduler='none'.")
            return self._fit_until_validation_stops(
                gpu_train, gpu_val, batch_size, early_stopping, verbose=verbose
            )

        scheduler = self._make_scheduler(epochs)
        last_val_miou: float | None = None
        self.val_history = []
        self.best_epoch = self.stopped_epoch = None

        for epoch in range(epochs):
            self._train_epoch(gpu_train, batch_size, f"Epoch {epoch + 1}/{epochs}", verbose=verbose)
            if scheduler is not None:
                scheduler.step()

            if gpu_val is not None:
                val_metrics = self._evaluate(gpu_val, batch_size)
                assert isinstance(val_metrics, dict)
                last_val_miou = val_metrics["mIoU"]
                self.val_history.append(last_val_miou)
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch + 1, last_val_miou)

        return last_val_miou

    def _device_cache(self, cache: "CachedFeaturesDataset | GPUTensorCache") -> GPUTensorCache:
        """Move a CPU cache to the training device once; reuse device caches as-is."""
        if isinstance(cache, GPUTensorCache):
            return cache
        return GPUTensorCache.from_cached(cache, self.device)

    def _train_epoch(
        self, gpu_train: GPUTensorCache, batch_size: int, desc: str, *, verbose: bool
    ) -> None:
        """Run one shuffled pass over the cached training features."""
        self.model.train()
        if self.model.freeze_backbone:
            self.model.backbone.eval()
        input_hw = (gpu_train.masks.shape[-2], gpu_train.masks.shape[-1])
        batches = tqdm(
            gpu_train.shuffled_batches(batch_size),
            total=math.ceil(len(gpu_train) / batch_size),
            desc=desc,
            disable=not verbose,
        )
        for features, masks in batches:
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model.head(features, *input_hw)
                loss: torch.Tensor = self.criterion(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def _fit_until_validation_stops(
        self,
        gpu_train: GPUTensorCache,
        gpu_val: GPUTensorCache,
        batch_size: int,
        config: EarlyStoppingConfig,
        *,
        verbose: bool,
    ) -> float:
        """Train with validation checks at epoch 0 and every ``check_every`` epochs."""
        tracker = ValidationStopping(config)
        self.val_history = []
        best_state: dict[str, torch.Tensor] | None = None
        epoch = 0
        while True:
            if epoch % config.check_every == 0 or epoch == config.max_epochs:
                val_metrics = self._evaluate(gpu_val, batch_size)
                assert isinstance(val_metrics, dict)
                miou = val_metrics["mIoU"]
                self.val_history.append(miou)
                improved, stop = tracker.observe(epoch, miou)
                if improved:
                    best_state = copy.deepcopy(self.model.head.state_dict())
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch, miou)
                if stop or epoch == config.max_epochs:
                    break
            epoch += 1
            self._train_epoch(
                gpu_train, batch_size, f"Epoch {epoch}/{config.max_epochs}", verbose=verbose
            )
        assert best_state is not None
        assert tracker.best_miou is not None
        self.model.head.load_state_dict(best_state)
        self.best_epoch, self.stopped_epoch = tracker.best_epoch, epoch
        return tracker.best_miou

    def select_learning_rate_cached(  # noqa: PLR0913 -- Mirrors fit_cached plus the grid.
        self,
        train_cache: "CachedFeaturesDataset | GPUTensorCache",
        val_cache: "CachedFeaturesDataset | GPUTensorCache",
        learning_rates: list[float],
        batch_size: int = 64,
        epochs: int = 10,
        *,
        seed: int,
        verbose: bool = True,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> float:
        """Fit the same initial head at each rate and keep the one best on validation.

        Every rate starts from identical head weights and batch order, so rates differ only in
        the learning rate. Exact validation ties go to the lower rate.

        Returns:
            The selected learning rate; the head is left holding its fitted weights.
        """
        if not learning_rates:
            raise ValueError("learning_rates must not be empty.")
        gpu_train, gpu_val = self._device_cache(train_cache), self._device_cache(val_cache)
        initial = copy.deepcopy(self.model.head.state_dict())
        self.lr_history = []
        best: tuple[float, float, dict[str, torch.Tensor]] | None = None
        for lr in learning_rates:
            self.model.head.load_state_dict(initial)
            self._reset_optimizer(lr)
            torch.manual_seed(seed)
            miou = self.fit_cached(
                gpu_train,
                gpu_val,
                batch_size,
                epochs,
                verbose=verbose,
                early_stopping=early_stopping,
            )
            assert miou is not None
            self.lr_history.append({"lr": lr, "val_miou": miou})
            if best is None or miou > best[1] or (miou == best[1] and lr < best[0]):
                best = (lr, miou, copy.deepcopy(self.model.head.state_dict()))
        assert best is not None
        self.model.head.load_state_dict(best[2])
        return best[0]

    def _reset_optimizer(self, lr: float) -> None:
        """Start a fresh optimizer and loss scaler at the given learning rate."""
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def evaluate_cached(
        self,
        cache: CachedFeaturesDataset,
        batch_size: int = 64,
        *,
        collect_confusions: bool = False,
    ) -> SegMetrics | tuple[SegMetrics, torch.Tensor]:
        """Evaluate segmentation metrics from cached backbone features.

        Transfer the cache once as a :class:`GPUTensorCache` to avoid per-batch copies.

        Args:
            cache: Pre-extracted features (output of
                :meth:`SegmentationProbe.extract_segmentation_features`).
            batch_size: Batch size for iterating over the cache.
            collect_confusions: If True, also return per-image confusion matrices.

        Returns:
            Metrics, optionally followed by per-image confusion matrices.
        """
        gpu_cache = GPUTensorCache.from_cached(cache, self.device)
        return self._evaluate(
            gpu_cache,
            batch_size,
            collect_confusions=collect_confusions,
        )

    def _per_image_confusions(self, predictions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return one confusion matrix per image, excluding ignored labels."""
        predictions = predictions.reshape(len(predictions), -1)
        masks = masks.reshape(len(masks), -1)
        valid = masks != self.ignore_index
        offsets = torch.arange(len(masks), device=masks.device).unsqueeze(1)
        indices = offsets * self.num_classes**2 + masks * self.num_classes + predictions
        indices = indices[valid]
        counts = torch.bincount(indices, minlength=len(masks) * self.num_classes**2)
        return counts.reshape(len(masks), self.num_classes, self.num_classes)

    def _compute_metrics(self) -> "SegMetrics":
        """Compute and return all metrics as a dict."""
        return {
            "mIoU": self.metric.compute().item(),
            "fw_IoU": self.metric_fw_iou.compute().item(),
            "precision": self.metric_precision.compute().item(),
            "recall": self.metric_recall.compute().item(),
            "f1": self.metric_f1.compute().item(),
        }

    def _evaluation_batches(
        self, data: DataLoader | GPUTensorCache, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield logits and masks in input order using the appropriate forward path."""
        if isinstance(data, GPUTensorCache):
            input_hw = (data.masks.shape[-2], data.masks.shape[-1])
            for features, masks in data.ordered_batches(batch_size):
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model.head(features, *input_hw)
                yield logits, masks
            return

        for batch in data:
            if isinstance(batch, dict):
                images, masks = batch["image"], batch["mask"]
            else:
                images, masks = batch[0], batch[1]
            images = images.to(self.device)
            masks = masks.to(self.device)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = masks.long()
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model(images)
            yield logits, masks

    @torch.no_grad()
    def _evaluate(
        self,
        data: DataLoader | GPUTensorCache,
        batch_size: int = 64,
        *,
        collect_confusions: bool = False,
    ) -> SegMetrics | tuple[SegMetrics, torch.Tensor]:
        """Evaluate raw images or device-resident features with the same metrics."""
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        confusion_list: list[torch.Tensor] = []

        for logits, masks in self._evaluation_batches(data, batch_size):
            for m in self._all_metrics:
                m.update(logits, masks)
            if collect_confusions:
                predictions = logits.argmax(dim=1)
                confusion_list.append(self._per_image_confusions(predictions, masks).cpu())

        metrics = self._compute_metrics()
        if collect_confusions:
            return metrics, torch.cat(confusion_list, dim=0)
        return metrics


def build_seg_probe_and_solver(
    model: nn.Module,
    num_classes: int,
    config: SegmentationConfig,
    device: torch.device,
) -> tuple[SegmentationProbe, SegmentationSolver]:
    """Build the frozen-backbone probe and solver from validated segmentation settings."""
    layer_names = config.layers
    if not layer_names:
        raise ValueError(
            "Segmentation evaluation requires segmentation.layers to name "
            "spatial backbone layers. Refusing to probe the global backbone output."
        )
    probe = SegmentationProbe(
        backbone=model,
        layer_names=layer_names,
        num_classes=num_classes,
        head_type=config.head,
        freeze_backbone=True,
        temporal_pool=config.temporal_pool,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    solver = SegmentationSolver(
        model=probe,
        num_classes=num_classes,
        lr=config.learning_rate,
        device=str(device),
        criterion=criterion,
        lr_scheduler=config.scheduler,
        ignore_index=config.ignore_index,
    )
    return probe, solver
