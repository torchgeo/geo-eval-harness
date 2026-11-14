"""Training and evaluation for segmentation probes."""

import logging
import math
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass

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
from .config.schema import Adaptation, SegmentationConfig
from .segmentation_checkpointing import enable_gradient_checkpointing
from .segmentation_lora import apply_lora, has_lora_adapters, merged_lora, set_lora_train_mode
from .segmentation_probe import (
    MAX_NONFINITE_FRACTION,
    CachedFeaturesDataset,
    GPUTensorCache,
    NonFiniteFeaturesError,
    SegmentationProbe,
    check_finite_features,
)

logger = logging.getLogger(__name__)

SegMetrics = dict[str, float]


@dataclass(frozen=True, kw_only=True)
class EpochRecord:
    """Training and validation measurements for one completed segmentation epoch.

    Attributes:
        epoch: Epoch number, starting from one.
        train_loss: Sample-weighted mean training loss over the epoch's optimizer steps.
        val_loss: Validation loss as the criterion's mean over the whole split, or ``None``
            without validation.
        val_miou: Validation dataset-level macro mIoU, or ``None`` without validation.
        head_lr: Head learning rate used during the epoch.
        backbone_lr: Learning rate of trainable backbone parameters, or ``None`` for a frozen
            backbone.
        train_steps: Optimizer steps taken.
        skipped_batches: Training batches skipped as smaller than the head minimum.
        nonfinite_batches: Training batches skipped because the backbone features were not
            finite (for example a float16 overflow); the head never saw them.
        epoch_seconds: Wall time for training and validation.
        peak_gpu_mem_mb: Peak allocated CUDA memory in MiB, or ``None`` off CUDA.
        train_miou: Macro mIoU of the training-mode predictions accumulated over the epoch's
            optimizer steps (weights change during the epoch), using the validation definition:
            ignored labels excluded and classes with an empty union left out.
    """

    epoch: int
    train_loss: float
    val_loss: float | None
    val_miou: float | None
    head_lr: float
    backbone_lr: float | None
    train_steps: int
    skipped_batches: int
    nonfinite_batches: int
    epoch_seconds: float
    peak_gpu_mem_mb: float | None
    train_miou: float


def _macro_miou(confusion: torch.Tensor) -> float:
    """Return macro IoU over classes with a non-empty union, matching torchmetrics."""
    confusion = confusion.double()
    tp = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - tp
    present = union > 0
    return float((tp[present] / union[present]).mean()) if present.any() else float("nan")


class _EpochTracker:
    """Accumulate one epoch's training statistics without per-step device syncs."""

    def __init__(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        num_classes: int,
        ignore_index: int,
    ) -> None:
        self.epoch = epoch
        self.lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.steps = 0
        self.skipped = 0
        self.nonfinite = 0
        self.samples = 0
        self.loss_sum = torch.zeros((), device=device)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
        self.started = time.perf_counter()

    def add_step(
        self, loss: torch.Tensor, batch_size: int, logits: torch.Tensor, masks: torch.Tensor
    ) -> None:
        """Record one optimizer step and its training-mode predictions."""
        self.loss_sum += loss.detach().float() * batch_size
        self.samples += batch_size
        self.steps += 1
        valid = masks != self.ignore_index
        predictions = logits.detach().argmax(dim=1)
        indices = masks[valid] * self.num_classes + predictions[valid]
        self.confusion += torch.bincount(indices, minlength=self.num_classes**2)

    def train_miou(self) -> float:
        """Return the epoch's training mIoU."""
        return _macro_miou(self.confusion.reshape(self.num_classes, self.num_classes))


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
        early_stopping_patience: int | None = None,
        *,
        amp: bool = True,
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
            early_stopping_patience: Stop training once validation mIoU has not improved for
                this many epochs. The LR schedule still spans ``epochs``, so a stopped run equals
                the first epochs of the full run. ``None`` trains all epochs.
            amp: Train and evaluate under CUDA float16 autocast with gradient scaling.
                ``False`` runs in float32.
        """
        self.model = model.to(device)
        self.backbone_eval_mode = False
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler
        self.early_stopping_patience = early_stopping_patience
        self.epoch_history: list[EpochRecord] = []
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

        self.amp = amp
        self.use_amp = amp and device.startswith("cuda") and torch.cuda.is_available()
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

    def _check_nonfinite(self, tracker: "_EpochTracker") -> None:
        """Report non-finite batches and fail if they exceed ``MAX_NONFINITE_FRACTION``."""
        if not tracker.nonfinite:
            return
        total = tracker.steps + tracker.skipped + tracker.nonfinite
        logger.warning(
            "Epoch %d skipped %d of %d training batches with non-finite backbone features.",
            tracker.epoch,
            tracker.nonfinite,
            total,
        )
        if tracker.nonfinite > MAX_NONFINITE_FRACTION * total:
            raise NonFiniteFeaturesError(
                f"Segmentation epoch {tracker.epoch} skipped {tracker.nonfinite} of {total} "
                f"training batches with non-finite backbone features (limit "
                f"{MAX_NONFINITE_FRACTION:.0%}). Set segmentation.amp: off to run in float32."
            )

    def _train_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        tracker: "_EpochTracker",
        *,
        doy: torch.Tensor | None = None,
    ) -> None:
        """Take one optimizer step, or count and skip a batch whose backbone features overflowed.

        The probe raises before its head runs, so head normalisation statistics stay finite.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
            try:
                logits = self.model(images, doy=doy)
            except NonFiniteFeaturesError:  # allow-except: skip the batch before the head
                tracker.nonfinite += 1
                return
            loss: torch.Tensor = self.criterion(logits, masks)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        tracker.add_step(loss, len(images), self._tracked_logits(logits), masks)

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

    def _tracked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, C, H, W)`` training logits whose mIoU the epoch record reports."""
        return logits

    def _set_train_mode(self) -> None:
        """Train the head; a frozen or LoRA-adapted backbone stays in eval mode except adapters.

        With ``backbone_eval_mode`` (full fine-tuning) every backbone weight trains, but dropout,
        stochastic depth, and normalisation statistics behave as at inference, like LoRA.
        """
        self.model.train()
        if self.model.freeze_backbone or self.backbone_eval_mode:
            self.model.backbone.eval()
        elif has_lora_adapters(self.model.backbone):
            set_lora_train_mode(self.model.backbone)

    def _reset_selection(self, epochs: int) -> None:
        """Clear per-fit selection state and require completed training epochs."""
        self.epoch_history = []
        self.best_epoch = None
        self.best_val_miou = None
        self._best_state = []
        if epochs < 1:
            raise ValueError("Segmentation fitting requires at least one training epoch.")

    @property
    def val_history(self) -> list[float]:
        """Return every epoch's validation mIoU in chronological order."""
        return [r.val_miou for r in self.epoch_history if r.val_miou is not None]

    def _begin_epoch(self, epoch: int) -> _EpochTracker:
        """Enter train mode and start measuring one epoch."""
        self._set_train_mode()
        if self._tracks_cuda_memory():
            torch.cuda.reset_peak_memory_stats(self.device)
        return _EpochTracker(
            epoch, self.optimizer, self.device, self.num_classes, self.ignore_index
        )

    def _tracks_cuda_memory(self) -> bool:
        """Return whether peak CUDA memory can be measured on this device."""
        return self.device_type == "cuda" and torch.cuda.is_available()

    def _finish_epoch(
        self,
        tracker: _EpochTracker,
        val_metrics: SegMetrics | None,
        on_epoch: Callable[[EpochRecord], None] | None,
    ) -> None:
        """Record the epoch and report it before selection, so failing epochs stay visible."""
        train_loss = (tracker.loss_sum / tracker.samples).item()
        peak_mem = (
            torch.cuda.max_memory_allocated(self.device) / 2**20
            if self._tracks_cuda_memory()
            else None
        )
        head_lr, backbone_lr = self._epoch_lrs(tracker)
        record = EpochRecord(
            epoch=tracker.epoch,
            train_loss=train_loss,
            val_loss=None if val_metrics is None else val_metrics.get("loss"),
            val_miou=None if val_metrics is None else val_metrics["mIoU"],
            head_lr=head_lr,
            backbone_lr=backbone_lr,
            train_steps=tracker.steps,
            skipped_batches=tracker.skipped,
            nonfinite_batches=tracker.nonfinite,
            epoch_seconds=time.perf_counter() - tracker.started,
            peak_gpu_mem_mb=peak_mem,
            train_miou=tracker.train_miou(),
        )
        self.epoch_history.append(record)
        if on_epoch is not None:
            on_epoch(record)

    def _epoch_lrs(self, tracker: _EpochTracker) -> tuple[float, float | None]:
        """Return the epoch's head and backbone learning rates (parameter groups 0 and 1)."""
        return tracker.lrs[0], tracker.lrs[1] if len(tracker.lrs) > 1 else None

    def _select_epoch(self, epoch: int, miou: float, *, verbose: bool) -> None:
        """Snapshot mutable evaluation state on strict validation improvement."""
        if not math.isfinite(miou):
            raise ValueError(
                f"Segmentation validation mIoU must be finite; epoch {epoch} yielded {miou}."
            )
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

    def _select_and_check_stop(
        self, epoch: int, val_metrics: "SegMetrics | None", *, verbose: bool
    ) -> bool:
        """Select on validation mIoU; return whether early-stopping patience has run out."""
        if val_metrics is None:
            return False
        self._select_epoch(epoch, val_metrics["mIoU"], verbose=verbose)
        patience = self.early_stopping_patience
        if patience is None or self.best_epoch is None or epoch - self.best_epoch < patience:
            return False
        logger.info("Early stopping after epoch %d (best epoch %d).", epoch, self.best_epoch)
        return True

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
        on_epoch: Callable[[EpochRecord], None] | None = None,
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
            on_epoch: Called with each epoch's :class:`EpochRecord` before epoch selection.

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
            tracker = self._begin_epoch(epoch + 1)

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
                if self._skip_small_batch(len(images), skip_logged=skip_logged):
                    skip_logged = True
                    tracker.skipped += 1
                    continue

                self._train_step(
                    images,
                    masks,
                    tracker,
                    doy=batch.get("doy") if isinstance(batch, dict) else None,
                )

            self._check_nonfinite(tracker)
            self._require_training_step(epoch + 1, tracker.steps)
            if scheduler is not None:
                scheduler.step()

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                assert isinstance(val_metrics, dict)
            self._finish_epoch(tracker, val_metrics, on_epoch)
            if self._select_and_check_stop(epoch + 1, val_metrics, verbose=verbose):
                break

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

    def fit_cached(  # noqa: PLR0913 -- Public training options.
        self,
        train_cache: "CachedFeaturesDataset | GPUTensorCache",
        val_cache: "CachedFeaturesDataset | GPUTensorCache | None" = None,
        batch_size: int = 64,
        epochs: int = 10,
        *,
        verbose: bool = True,
        on_epoch: Callable[[EpochRecord], None] | None = None,
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
            on_epoch: Called with each epoch's :class:`EpochRecord` before epoch selection.

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
            else GPUTensorCache.from_cached(train_cache, self.device, amp=self.amp)
        )
        gpu_val = (
            GPUTensorCache.from_cached(val_cache, self.device, amp=self.amp)
            if isinstance(val_cache, CachedFeaturesDataset)
            else val_cache
        )

        scheduler = self._make_scheduler(epochs)

        input_hw: tuple[int, int] = (gpu_train.masks.shape[-2], gpu_train.masks.shape[-1])
        num_batches = math.ceil(len(gpu_train) / batch_size)
        skip_logged = False

        for epoch in range(epochs):
            tracker = self._begin_epoch(epoch + 1)

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = tqdm(
                gpu_train.shuffled_batches(batch_size),
                total=num_batches,
                desc=desc,
                disable=not verbose,
            )
            for features, masks in batches:
                if self._skip_small_batch(len(masks), skip_logged=skip_logged):
                    skip_logged = True
                    tracker.skipped += 1
                    continue
                try:
                    check_finite_features(features)
                except NonFiniteFeaturesError:  # allow-except: skip the batch before the head
                    tracker.nonfinite += 1
                    continue
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model.head(features, *input_hw)
                    loss: torch.Tensor = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                tracker.add_step(loss, len(masks), self._tracked_logits(logits), masks)

            self._check_nonfinite(tracker)
            self._require_training_step(epoch + 1, tracker.steps)
            if scheduler is not None:
                scheduler.step()

            val_metrics = None
            if gpu_val is not None:
                val_metrics = self._evaluate(gpu_val, batch_size)
                assert isinstance(val_metrics, dict)
            self._finish_epoch(tracker, val_metrics, on_epoch)
            if self._select_and_check_stop(epoch + 1, val_metrics, verbose=verbose):
                break

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
        gpu_cache = GPUTensorCache.from_cached(cache, self.device, amp=self.amp)
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

    def _loss_weight(self, masks: torch.Tensor) -> torch.Tensor:
        """Return the denominator of the criterion's batch mean, so batch means combine exactly.

        Mean cross-entropy divides by the summed class weight of the labeled pixels, which is
        their count without class weights.
        """
        labeled = masks != self.ignore_index
        class_weight = getattr(self.criterion, "weight", None)
        if class_weight is None:
            return labeled.sum()
        return (class_weight[torch.where(labeled, masks, 0)] * labeled).sum()

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
        """Yield logits and masks in input order using the appropriate forward path.

        Non-finite features or logits raise instead of scoring a constant prediction.
        """
        for logits, masks in self._raw_evaluation_batches(data, batch_size):
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    "Segmentation evaluation logits are not finite (float16 overflow or "
                    "corrupted normalisation statistics?). Set segmentation.amp: off to run in "
                    "float32."
                )
            yield logits, masks

    def _raw_evaluation_batches(
        self, data: DataLoader | GPUTensorCache, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(data, GPUTensorCache):
            input_hw = (data.masks.shape[-2], data.masks.shape[-1])
            for features, masks in data.ordered_batches(batch_size):
                check_finite_features(features)
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
                logits = self.model(
                    images, doy=batch.get("doy") if isinstance(batch, dict) else None
                )
            yield logits, masks

    @torch.no_grad()
    def _evaluate(
        self,
        data: DataLoader | GPUTensorCache,
        batch_size: int = 64,
        *,
        collect_confusions: bool = False,
    ) -> SegMetrics | tuple[SegMetrics, torch.Tensor]:
        """Evaluate raw images or device-resident features with the same metrics.

        LoRA adapters are merged into the backbone weights for the pass (see
        :func:`merged_lora`), so evaluation costs the same as the plain backbone.
        Every split reports ``loss`` alongside the metrics; training logs only the validation loss.
        """
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        backbone = getattr(self.model, "backbone", None)
        merge = (
            merged_lora(backbone)
            if backbone is not None
            and not isinstance(data, GPUTensorCache)
            and has_lora_adapters(backbone)
            else nullcontext()
        )
        confusion_list: list[torch.Tensor] = []
        loss_sum = torch.zeros((), device=self.device)
        loss_weight: torch.Tensor | int = 0
        with merge:
            for logits, masks in self._evaluation_batches(data, batch_size):
                for m in self._all_metrics:
                    m.update(logits, masks)
                weight = self._loss_weight(masks)
                batch_loss = self.criterion(logits.float(), masks) * weight
                loss_sum += torch.where(weight > 0, batch_loss, 0.0)
                loss_weight = loss_weight + weight
                if collect_confusions:
                    predictions = logits.argmax(dim=1)
                    confusion_list.append(self._per_image_confusions(predictions, masks).cpu())

        metrics = self._compute_metrics()
        total_weight = float(loss_weight)
        metrics["loss"] = (loss_sum / total_weight).item() if total_weight > 0 else float("nan")
        if collect_confusions:
            return metrics, torch.cat(confusion_list, dim=0)
        return metrics


class SweepCrossEntropy(nn.CrossEntropyLoss):
    """Sum of per-head cross-entropy losses for ``(n_heads, B, C, H, W)`` sweep logits.

    Heads share no parameters, so each receives exactly the gradient of its own mean loss.
    ``(B, C, H, W)`` logits of a selected head get the plain cross-entropy.
    """

    def __init__(self, ignore_index: int) -> None:
        super().__init__(ignore_index=ignore_index)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: A002 -- PyTorch loss signature.
        """Return the summed per-head loss."""
        if input.ndim == target.ndim + 1:
            return super().forward(input, target)
        n_heads = input.shape[0]
        targets = target.unsqueeze(0).expand(n_heads, *target.shape).flatten(0, 1)
        loss = super().forward(input.flatten(0, 1).float(), targets)
        return loss * n_heads


class SweepSegmentationSolver(SegmentationSolver):
    """Train a learning-rate x weight-decay grid of linear probes at once and keep the best.

    UniverSat's ``LP_eval.py`` protocol: every head of a ``SweepPatchLinearHead`` has its own
    AdamW learning rate and weight decay and a cosine schedule to zero without warmup. After each
    epoch every head is scored on validation mIoU and keeps a snapshot of its best epoch. Training
    ends by restoring the head with the highest best-epoch validation mIoU (ties: first in grid
    order, learning rate major); test evaluation then uses only that head.
    """

    def __init__(  # noqa: PLR0913 -- Public constructor options.
        self,
        model: SegmentationProbe,
        num_classes: int,
        learning_rates: list[float],
        weight_decays: list[float],
        device: str = "cuda",
        lr_scheduler: str = "cosine",
        ignore_index: int = 255,
        early_stopping_patience: int | None = None,
        *,
        amp: bool = True,
    ) -> None:
        """Configure one AdamW parameter group per sweep head."""
        head = model.sweep_head
        if head is None:
            raise ValueError("SweepSegmentationSolver requires a probe built with sweep_heads.")
        self.grid = [(lr, wd) for lr in learning_rates for wd in weight_decays]
        if head.n_heads != len(self.grid):
            raise ValueError(
                f"The sweep head has {head.n_heads} heads but the grid has {len(self.grid)} cells."
            )
        super().__init__(
            model=model,
            num_classes=num_classes,
            lr=learning_rates[0],
            device=device,
            criterion=SweepCrossEntropy(ignore_index),
            lr_scheduler=lr_scheduler,
            ignore_index=ignore_index,
            early_stopping_patience=early_stopping_patience,
            amp=amp,
        )
        self.sweep_head = head
        groups = [
            {"params": head.head_parameters(i), "lr": lr, "weight_decay": wd}
            for i, (lr, wd) in enumerate(self.grid)
        ]
        grouped = {id(p) for i in range(head.n_heads) for p in head.head_parameters(i)}
        trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        if trainable != grouped:
            raise ValueError("Sweep training expects only the sweep head's parameters to train.")
        self.optimizer = torch.optim.AdamW(groups)
        self.head_val_history: list[list[float]] = []
        self.selected_head: int | None = None
        self._last_head_mious: list[float] = []
        self._head_best: list[tuple[float, int, list[torch.Tensor]] | None] = []

    @property
    def selected_lr(self) -> float | None:
        """Return the selected head's learning rate."""
        return None if self.selected_head is None else self.grid[self.selected_head][0]

    @property
    def selected_weight_decay(self) -> float | None:
        """Return the selected head's weight decay."""
        return None if self.selected_head is None else self.grid[self.selected_head][1]

    def _make_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Return a per-epoch cosine decay of every group's learning rate to zero."""
        if self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
            )
        return super()._make_scheduler(epochs)

    def _reset_selection(self, epochs: int) -> None:
        super()._reset_selection(epochs)
        self.sweep_head.selected = None
        self.selected_head = None
        self.head_val_history = []
        self._last_head_mious = []
        self._head_best = [None] * self.sweep_head.n_heads

    def _leading_head(self) -> int:
        return max(range(len(self._head_best)), key=self._head_score, default=0)

    def _tracked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Report training mIoU of the head that currently leads on validation."""
        if logits.ndim != 5:
            return logits
        return logits[self._leading_head()]

    def _epoch_lrs(self, tracker: _EpochTracker) -> tuple[float, float | None]:
        """Report the learning rate of the head leading on validation; the backbone is frozen."""
        return tracker.lrs[self._leading_head()], None

    def _head_score(self, index: int) -> float:
        best = self._head_best[index]
        return -math.inf if best is None else best[0]

    @torch.no_grad()
    def _evaluate(
        self,
        data: DataLoader | GPUTensorCache,
        batch_size: int = 64,
        *,
        collect_confusions: bool = False,
    ) -> SegMetrics | tuple[SegMetrics, torch.Tensor]:
        """Score every head on ``data`` until a head is selected, then evaluate that head."""
        if self.sweep_head.selected is not None:
            return super()._evaluate(data, batch_size, collect_confusions=collect_confusions)
        if collect_confusions:
            raise ValueError("Per-image confusions need a selected sweep head.")
        self.model.eval()
        n_heads, classes = self.sweep_head.n_heads, self.num_classes
        confusion = torch.zeros(n_heads * classes * classes, dtype=torch.long, device=self.device)
        offsets = torch.arange(n_heads, device=self.device).view(n_heads, 1, 1, 1) * classes**2
        for logits, masks in self._evaluation_batches(data, batch_size):
            predictions = logits.argmax(dim=2)
            valid = (masks != self.ignore_index).unsqueeze(0).expand_as(predictions)
            indices = offsets + masks.unsqueeze(0) * classes + predictions
            confusion += torch.bincount(indices[valid], minlength=n_heads * classes**2)
        confusion = confusion.view(n_heads, classes, classes).cpu()
        self._last_head_mious = [_macro_miou(confusion[i]) for i in range(n_heads)]
        self.head_val_history.append(self._last_head_mious)
        return {"mIoU": max(self._last_head_mious)}

    def _select_epoch(self, epoch: int, miou: float, *, verbose: bool) -> None:
        """Snapshot every head whose validation mIoU strictly improved this epoch."""
        if not all(math.isfinite(m) for m in self._last_head_mious):
            raise ValueError(f"Sweep validation mIoU must be finite; epoch {epoch} was not.")
        for index, head_miou in enumerate(self._last_head_mious):
            if head_miou <= self._head_score(index):
                continue
            state = [
                p.detach().to("cpu", copy=True) for p in self.sweep_head.head_parameters(index)
            ]
            self._head_best[index] = (head_miou, epoch, state)
        if self.best_val_miou is None or miou > self.best_val_miou:
            self.best_epoch, self.best_val_miou = epoch, miou
        if verbose:
            lead = max(range(len(self._last_head_mious)), key=self._last_head_mious.__getitem__)
            lr, wd = self.grid[lead]
            logger.info("Epoch %d sweep val mIoU: best %.17g (lr=%g, wd=%g)", epoch, miou, lr, wd)

    @torch.no_grad()
    def _restore_selection(self, *, verbose: bool) -> float | None:
        """Restore the best head at its best epoch and route evaluation to it."""
        if all(best is None for best in self._head_best):
            return None
        index = max(range(len(self._head_best)), key=self._head_score)
        best = self._head_best[index]
        assert best is not None
        miou, epoch, state = best
        for tensor, saved in zip(self.sweep_head.head_parameters(index), state, strict=True):
            tensor.copy_(saved)
        self._head_best = []
        self.sweep_head.selected = index
        self.selected_head = index
        self.best_epoch, self.best_val_miou = epoch, miou
        if verbose:
            lr, wd = self.grid[index]
            logger.info(
                "Selected sweep head lr=%g wd=%g, epoch %d, val mIoU %.17g", lr, wd, epoch, miou
            )
        return miou


def build_seg_probe_and_solver(
    model: nn.Module,
    num_classes: int,
    config: SegmentationConfig,
    device: torch.device,
) -> tuple[SegmentationProbe, SegmentationSolver]:
    """Build the probe and solver from validated segmentation settings.

    The backbone is frozen unless ``config.lora_rank`` adds LoRA adapters or
    ``config.full_finetune`` trains every backbone weight, both at
    ``config.backbone_learning_rate``.
    """
    layer_names = config.layers
    if not layer_names:
        raise ValueError(
            "Segmentation evaluation requires segmentation.layers to name "
            "spatial backbone layers. Refusing to probe the global backbone output."
        )
    adaptation = config.adaptation
    if adaptation is Adaptation.LORA:
        assert config.lora_rank is not None
        assert config.lora_target_modules is not None
        apply_lora(
            model,
            config.lora_target_modules,
            config.lora_rank,
            config.lora_alpha,
            last_blocks=config.lora_last_blocks,
        )
    if adaptation is Adaptation.FULL:
        # Some wrappers freeze their backbone at construction; full fine-tuning has to undo that.
        enable = getattr(model, "enable_full_finetune", None)
        if callable(enable):
            enable()
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError(
                f"full_finetune: {type(model).__name__} has no trainable backbone parameter; "
                "its wrapper freezes the backbone and does not define enable_full_finetune()."
            )
    if config.gradient_checkpointing:
        if config.lora_rank is None and not config.full_finetune:
            logger.info("gradient_checkpointing has no effect on a frozen backbone; skipped.")
        else:
            enable_gradient_checkpointing(model)
    probe = SegmentationProbe(
        backbone=model,
        layer_names=layer_names,
        num_classes=num_classes,
        head_type=config.head,
        neck=config.neck,
        neck_norm=config.neck_norm,
        upsampler=config.upsampler,
        head_dropout=config.head_dropout,
        freeze_backbone=adaptation is Adaptation.FROZEN,
        temporal_pool=config.temporal_pool,
        sweep_heads=len(config.sweep_grid),
        feature_standardize=config.feature_standardize,
        amp=config.amp == "fp16",
    )
    if config.sweep_grid:
        sweep_solver = SweepSegmentationSolver(
            model=probe,
            num_classes=num_classes,
            learning_rates=config.sweep_learning_rates,
            weight_decays=config.sweep_weight_decays,
            device=str(device),
            lr_scheduler=config.scheduler,
            ignore_index=config.ignore_index,
            early_stopping_patience=config.early_stopping_patience,
            amp=config.amp == "fp16",
        )
        return probe, sweep_solver
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
        weight_decay=config.weight_decay,
        early_stopping_patience=config.early_stopping_patience,
        amp=config.amp == "fp16",
    )
    solver.backbone_eval_mode = adaptation is Adaptation.FULL
    return probe, solver
