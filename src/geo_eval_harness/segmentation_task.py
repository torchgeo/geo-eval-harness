"""Training and evaluation for segmentation probes."""

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

from .boundary_iou import MulticlassBoundaryIoU
from .config.schema import SegmentationConfig
from .segmentation_lora import apply_lora, has_lora_adapters, set_lora_train_mode
from .segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
)

logger = logging.getLogger(__name__)

SegMetrics = dict[str, float]


class SegmentationSolver:
    """Train and evaluate a segmentation probe.

    See :ref:`segmentation-model-selection` for the selection policy.
    """

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
        backbone_lr: float | None = None,
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
            backbone_lr: Learning rate for trainable backbone parameters, such as LoRA adapters.
                ``None`` trains all parameters at ``lr``.
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler
        self.val_history: list[float] = []
        self.best_epoch: int | None = None
        self.best_val_miou: float | None = None
        self._best_state: list[tuple[torch.Tensor, torch.Tensor]] = []

        self.ignore_index = ignore_index
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if backbone_lr is not None:
            params = [
                {"params": [p for p in self.model.head.parameters() if p.requires_grad]},
                {
                    "params": [p for p in self.model.backbone.parameters() if p.requires_grad],
                    "lr": backbone_lr,
                },
            ]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

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
        self.metric_boundary_iou = MulticlassBoundaryIoU(
            num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        self._all_metrics = [
            self.metric,
            self.metric_fw_iou,
            self.metric_precision,
            self.metric_recall,
            self.metric_f1,
            self.metric_boundary_iou,
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

    def _min_train_batch_size(self) -> int:
        """Return the smallest training batch the head can learn from."""
        return int(getattr(self.model.head, "min_train_batch_size", 1))

    def _check_train_batch_size(self, batch_size: int | None) -> None:
        """Reject a configured training batch size that the head can never train on."""
        min_size = self._min_train_batch_size()
        if batch_size is not None and batch_size < min_size:
            raise ValueError(
                f"Segmentation training batch size {batch_size} is below the "
                f"{type(self.model.head).__name__} minimum of {min_size}. Increase "
                "segmentation.batch_size (cached features) or runtime.batch_size."
            )

    def _require_training_step(self, epoch: int, steps: int) -> None:
        """Fail instead of evaluating a head that an epoch could not train."""
        if steps == 0:
            raise ValueError(
                f"Segmentation epoch {epoch} had no usable training batch; every batch was "
                f"empty or smaller than the head minimum of {self._min_train_batch_size()}."
            )

    def _skip_small_batch(self, size: int, *, skip_logged: bool) -> bool:
        """Return whether a training batch is too small for the head, logging the first skip."""
        min_size = self._min_train_batch_size()
        if size >= min_size:
            return False
        if not skip_logged:
            logger.info(
                "Skipping a training batch of %d samples; the head needs at least %d.",
                size,
                min_size,
            )
        return True

    def _set_train_mode(self) -> None:
        """Train the head; a frozen or LoRA-adapted backbone stays in eval mode except adapters."""
        self.model.train()
        if self.model.freeze_backbone:
            self.model.backbone.eval()
        elif has_lora_adapters(self.model.backbone):
            set_lora_train_mode(self.model.backbone)

    def _reset_selection(self, epochs: int) -> None:
        """Clear per-fit selection state and require completed training epochs."""
        self.val_history = []
        self.best_epoch = None
        self.best_val_miou = None
        self._best_state = []
        if epochs < 1:
            raise ValueError("Segmentation fitting requires at least one training epoch.")

    def _select_epoch(self, epoch: int, miou: float, *, verbose: bool) -> None:
        """Snapshot mutable evaluation state on strict validation improvement."""
        if not math.isfinite(miou):
            raise ValueError(
                f"Segmentation validation mIoU must be finite; epoch {epoch} yielded {miou}."
            )
        self.val_history.append(miou)
        if verbose:
            logger.info("Epoch %d Val mIoU: %.17g", epoch, miou)
        if self.best_val_miou is not None and miou <= self.best_val_miou:
            return
        tensors = [
            *self.model.head.parameters(),
            *(p for p in self.model.backbone.parameters() if p.requires_grad),
            *self.model.buffers(),
        ]
        self._best_state = [(tensor, tensor.detach().to("cpu", copy=True)) for tensor in tensors]
        self.best_epoch, self.best_val_miou = epoch, miou

    @torch.no_grad()
    def _restore_selection(self, *, verbose: bool) -> float | None:
        """Restore into existing tensors without replacing modules or hooks."""
        for tensor, saved in self._best_state:
            tensor.copy_(saved)
        self._best_state = []
        if verbose and self.best_epoch is not None:
            logger.info("Selected epoch %d Val mIoU: %.17g", self.best_epoch, self.best_val_miou)
        return self.best_val_miou

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        *,
        verbose: bool = True,
    ) -> float | None:
        """Train the segmentation probe.

        A frozen backbone stays in eval mode, and a LoRA-adapted backbone keeps only its adapter
        modules in train mode. Training batches smaller than the head's ``min_train_batch_size``
        are skipped. A loader batch size below that minimum, or an epoch without any usable batch,
        raises ``ValueError``.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.

        Returns:
            Selected validation mIoU, or None without validation.
        """
        self._reset_selection(epochs)
        if val_loader is not None and len(val_loader) == 0:
            raise ValueError("Segmentation validation loader must contain at least one batch.")
        self._check_train_batch_size(train_loader.batch_size)
        scheduler = self._make_scheduler(epochs)
        skip_logged = False

        for epoch in range(epochs):
            self._set_train_mode()

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = tqdm(train_loader, desc=desc, disable=not verbose)
            steps = 0
            for batch in batches:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device).long()
                else:
                    images, masks = batch[0].to(self.device), batch[1].to(self.device).long()

                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                if self._skip_small_batch(len(images), skip_logged=skip_logged):
                    skip_logged = True
                    continue

                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss: torch.Tensor = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                steps += 1

            self._require_training_step(epoch + 1, steps)
            if scheduler is not None:
                scheduler.step()

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                assert isinstance(val_metrics, dict)
                self._select_epoch(epoch + 1, val_metrics["mIoU"], verbose=verbose)

        return self._restore_selection(verbose=verbose)

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

    def fit_cached(
        self,
        train_cache: "CachedFeaturesDataset | GPUTensorCache",
        val_cache: "CachedFeaturesDataset | GPUTensorCache | None" = None,
        batch_size: int = 64,
        epochs: int = 10,
        *,
        verbose: bool = True,
    ) -> float | None:
        """Train the segmentation head on cached backbone features.

        Cached features go directly to ``self.model.head``; the backbone is not called. Training
        batches smaller than the head's ``min_train_batch_size`` are skipped. A ``batch_size``
        below that minimum, or an epoch without any usable batch, raises ``ValueError``.

        Keep the full cache on the training device to avoid per-batch transfers and stacking.
        Pass a ``GPUTensorCache`` as ``train_cache`` or ``val_cache`` to reuse it across calls.

        Args:
            train_cache: Pre-extracted training features from
                :meth:`SegmentationProbe.extract_segmentation_features`, or a GPU cache.
            val_cache: Optional validation cache.
            batch_size: Batch size for iterating over cached data.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.

        Returns:
            Selected validation mIoU, or None without validation.
        """
        self._reset_selection(epochs)
        if val_cache is not None and len(val_cache) == 0:
            raise ValueError("Segmentation validation cache must contain at least one sample.")
        self._check_train_batch_size(batch_size)
        gpu_train = (
            train_cache
            if isinstance(train_cache, GPUTensorCache)
            else GPUTensorCache.from_cached(train_cache, self.device)
        )
        gpu_val = (
            GPUTensorCache.from_cached(val_cache, self.device)
            if isinstance(val_cache, CachedFeaturesDataset)
            else val_cache
        )

        scheduler = self._make_scheduler(epochs)

        input_hw: tuple[int, int] = (gpu_train.masks.shape[-2], gpu_train.masks.shape[-1])
        num_batches = math.ceil(len(gpu_train) / batch_size)
        skip_logged = False

        for epoch in range(epochs):
            self._set_train_mode()

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = tqdm(
                gpu_train.shuffled_batches(batch_size),
                total=num_batches,
                desc=desc,
                disable=not verbose,
            )
            steps = 0
            for features, masks in batches:
                if self._skip_small_batch(len(masks), skip_logged=skip_logged):
                    skip_logged = True
                    continue
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model.head(features, *input_hw)
                    loss: torch.Tensor = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                steps += 1

            self._require_training_step(epoch + 1, steps)
            if scheduler is not None:
                scheduler.step()

            if gpu_val is not None:
                val_metrics = self._evaluate(gpu_val, batch_size)
                assert isinstance(val_metrics, dict)
                self._select_epoch(epoch + 1, val_metrics["mIoU"], verbose=verbose)

        return self._restore_selection(verbose=verbose)

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
        """Compute and return all segmentation metrics."""
        return {
            "mIoU": self.metric.compute().item(),
            "fw_IoU": self.metric_fw_iou.compute().item(),
            "precision": self.metric_precision.compute().item(),
            "recall": self.metric_recall.compute().item(),
            "f1": self.metric_f1.compute().item(),
            "boundary_IoU": self.metric_boundary_iou.compute().item(),
            "boundary_IoU_width": float(self.metric_boundary_iou.boundary_width),
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
    """Build the probe and solver from validated segmentation settings.

    The backbone is frozen unless ``config.lora_rank`` adds LoRA adapters, which then train at
    ``config.backbone_learning_rate``.
    """
    layer_names = config.layers
    if not layer_names:
        raise ValueError(
            "Segmentation evaluation requires segmentation.layers to name "
            "spatial backbone layers. Refusing to probe the global backbone output."
        )
    if config.lora_rank is not None:
        assert config.lora_target_modules is not None
        apply_lora(model, config.lora_target_modules, config.lora_rank, config.lora_alpha)
    probe = SegmentationProbe(
        backbone=model,
        layer_names=layer_names,
        num_classes=num_classes,
        head_type=config.head,
        freeze_backbone=config.lora_rank is None,
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
        backbone_lr=config.backbone_learning_rate,
    )
    return probe, solver
