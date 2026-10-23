"""Segmentation probe: multi-scale frozen-backbone feature extraction and head training."""

import logging
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from geo_eval_harness.models.segmentation_heads import (
    ConvBlockHead,
    DPTHead,
    FPNHead,
    LinearHead,
    PatchLinearHead,
    UPerNetHead,
)

logger = logging.getLogger(__name__)


def _resolve_num_prefix_tokens(backbone: nn.Module) -> int | None:
    """Find a declared ``num_prefix_tokens`` in the backbone, or return ``None``.

    Search the module tree because timm models may be nested under ``.backbone``.
    Non-timm token models must declare their own count.

    DINOv3 has 1 CLS token and 4 register tokens, giving ``(B, 261, 1024)`` at 256 px.
    Removing only CLS leaves no square patch grid.
    """
    for module in backbone.modules():
        n = getattr(module, "num_prefix_tokens", None)
        if isinstance(n, int) and n >= 0:
            return n
    return None


class CachedFeaturesDataset(Dataset):
    """Backbone features and masks cached in RAM.

    ``layer_tensors[li]`` holds contiguous ``(N, C, H, W)`` features, float16 by default.
    ``masks`` is an ``(N, H, W)`` long tensor.
    :meth:`GPUTensorCache.from_cached` transfers whole layers, avoiding per-sample Python work.

    Each ``__getitem__`` returns a ``(features, mask)`` tuple.
    """

    def __init__(
        self,
        layer_tensors: list[torch.Tensor],
        masks: torch.Tensor,
    ) -> None:
        self.layer_tensors = layer_tensors
        self.masks = masks

    def __len__(self) -> int:
        return self.masks.shape[0]

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        return [t[index] for t in self.layer_tensors], self.masks[index]


def _estimate_cache_bytes(cache: "CachedFeaturesDataset") -> int:
    if not cache.layer_tensors:
        return 0
    return (
        sum(t.numel() * t.element_size() for t in cache.layer_tensors)
        + cache.masks.numel() * cache.masks.element_size()
    )


class GPUTensorCache:
    """Cached features and masks kept on the target device.

    Build with :meth:`from_cached` to avoid copying or stacking features each batch.
    Use :meth:`shuffled_batches` for training and :meth:`ordered_batches` for evaluation.

    Args:
        layer_tensors: Contiguous ``(N, C, H, W)`` tensors, one per hooked layer on ``device``.
            ``from_cached`` uses float16 on CUDA and float32 on CPU.
        masks: ``(N, H, W)`` long tensor on the target device.
        device: The device these tensors live on.
    """

    def __init__(
        self,
        layer_tensors: list[torch.Tensor],
        masks: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        self.layer_tensors = layer_tensors
        self.masks = masks
        self.device = device

    def __len__(self) -> int:
        return self.masks.shape[0]

    @classmethod
    def from_cached(
        cls,
        cache: "CachedFeaturesDataset",
        device: torch.device | str,
    ) -> "GPUTensorCache":
        """Move cached features and masks to ``device``.

        Args:
            cache: CPU-resident cached features.
            device: Target device.

        Returns:
            A :class:`GPUTensorCache` with all data on *device*.
        """
        target_device = torch.device(device)
        # CPU heads need float32 without autocast; CUDA uses float16 for mixed precision.
        dtype = torch.float16 if target_device.type == "cuda" else torch.float32
        layer_tensors = [t.to(target_device, dtype=dtype) for t in cache.layer_tensors]
        masks = cache.masks.to(target_device, dtype=torch.long)
        return cls(layer_tensors, masks, target_device)

    def shuffled_batches(
        self, batch_size: int
    ) -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        """Yield *(features, masks)* mini-batches in random order."""
        idx = torch.randperm(len(self), device=self.device)
        for start in range(0, len(self), batch_size):
            b = idx[start : start + batch_size]
            yield [t[b] for t in self.layer_tensors], self.masks[b]

    def ordered_batches(self, batch_size: int) -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        """Yield *(features, masks)* mini-batches in sequential order."""
        for start in range(0, len(self), batch_size):
            s = slice(start, start + batch_size)
            yield [t[s] for t in self.layer_tensors], self.masks[s]


class SegmentationProbe(nn.Module):
    """Predict per-pixel class logits from selected backbone feature layers.

    All heads, including ``DPTHead``, expect coarse-to-fine layers, deepest/lowest resolution first.
    For ResNet, use ``["layer4", "layer3", "layer2", "layer1"]``.

    Args:
        backbone: Feature extractor. May be a raw backbone or a ``BenchModel``
            wrapper (``backbone.*`` prefixes are stripped automatically).
        layer_names: Ordered list of layer names to hook (coarse-to-fine).
        num_classes: Number of segmentation output classes.
        freeze_backbone: If ``True`` (default), backbone parameters are frozen
            and the backbone runs in eval mode during inference.
        head_type: Decoder architecture — one of ``"linear"``, ``"conv_block"``,
            ``"fpn"``, ``"dpt"``, ``"patch_linear"``, ``"upernet"``.
        hidden_dim: Hidden channel dimension for ``conv_block``, ``fpn``,
            ``dpt``, and ``upernet`` heads (default 256).
    """

    def __init__(  # noqa: PLR0913 -- Public constructor options.
        self,
        backbone: nn.Module,
        layer_names: list[str],
        num_classes: int,
        *,
        freeze_backbone: bool = True,
        head_type: str = "linear",
        hidden_dim: int | None = None,
        temporal_pool: str = "mean",
    ) -> None:
        super().__init__()
        if temporal_pool not in ("mean", "max"):
            raise ValueError(f"temporal_pool must be 'mean' or 'max', got {temporal_pool!r}")
        self.temporal_pool = temporal_pool
        self.backbone = backbone
        self.layer_names = layer_names
        self.freeze_backbone = freeze_backbone
        self.head_type = head_type
        self.effective_classes = num_classes

        self._features: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

        self.register_hooks()

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.channels_list = self._dry_run_channels()
        hdim = hidden_dim or 256

        if head_type == "linear":
            self.head = LinearHead(self.channels_list, num_classes)
        elif head_type == "conv_block":
            self.head = ConvBlockHead(self.channels_list, num_classes, hidden_dim=hdim)
        elif head_type == "fpn":
            self.head = FPNHead(self.channels_list, num_classes, hidden_dim=hdim)
        elif head_type == "dpt":
            self.head = DPTHead(self.channels_list, num_classes, hidden_dim=hdim)
        elif head_type == "upernet":
            self.head = UPerNetHead(self.channels_list, num_classes, hidden_dim=hdim)
        elif head_type == "patch_linear":
            self.head = PatchLinearHead(self.channels_list, num_classes).to(self._backbone_device())
            dry_run_features = [
                torch.zeros(
                    (1, channels, height, width),
                    device=self._backbone_device(),
                )
                for channels, (height, width) in zip(
                    self.channels_list, self.feature_hw_list, strict=True
                )
            ]
            with torch.no_grad():
                _ = self.head(dry_run_features, *self.dry_run_input_hw)
        else:
            raise ValueError(
                "Unknown head_type: "
                f"{head_type!r}. Choose from: linear, conv_block, fpn, dpt, patch_linear, upernet"
            )

    def register_hooks(self) -> None:
        """Register feature hooks and reject missing or duplicate layer names."""
        duplicate_layers = {name for name in self.layer_names if self.layer_names.count(name) > 1}
        if duplicate_layers:
            raise ValueError(
                f"Segmentation probe layers must be unique; duplicates: {sorted(duplicate_layers)}."
            )

        found_layers = set()
        for name, module in self.backbone.named_modules():
            if name.startswith("backbone."):
                name = name.replace("backbone.", "", 1)
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
                found_layers.add(name)

        missing_layers = set(self.layer_names) - found_layers
        if missing_layers:
            available = [
                name.replace("backbone.", "", 1) if name.startswith("backbone.") else name
                for name, _ in self.backbone.named_modules()
            ]
            raise ValueError(
                f"Segmentation layers not found in backbone: {sorted(missing_layers)}. "
                f"Set eval.segmentation.layers for this model to names it exposes, e.g. "
                f"{[n for n in available if n][:8]}."
            )

    def _hook_fn(self, name: str):
        """Return a forward hook that captures the output of the named layer."""

        def hook(module, _input, output):  # noqa: ARG001
            self._features[name] = output

        return hook

    def _backbone_device(self) -> torch.device:
        """Use a backbone parameter or buffer's device, defaulting to CPU."""
        p = next(self.backbone.parameters(), None)
        if p is not None:
            return p.device
        b = next(self.backbone.buffers(), None)
        if b is not None:
            return b.device
        return torch.device("cpu")

    def _dry_run_channels(self) -> list[int]:
        device = self._backbone_device()
        in_channels = int(getattr(self.backbone, "num_channels", 3))
        dummy = torch.randn(1, in_channels, 224, 224, device=device)
        self.dry_run_input_hw = (224, 224)
        if not self.layer_names:
            self.layer_names = ["backbone_output"]
            self.hooks.append(self.backbone.register_forward_hook(self._hook_fn("backbone_output")))

        was_training = self.backbone.training
        self.backbone.eval()
        self._features.clear()
        with torch.no_grad():
            self.backbone(dummy)

        channels = []
        self.feature_hw_list: list[tuple[int, int]] = []
        for name in self.layer_names:
            feat = self._process_feature(self._features[name])
            channels.append(feat.shape[1])
            self.feature_hw_list.append((feat.shape[-2], feat.shape[-1]))
        self.backbone.train(was_training)
        return channels

    def _process_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 2:
            return feat.view(feat.shape[0], feat.shape[1], 1, 1)
        if feat.ndim == 3:
            return self.reshape_tokens(feat)
        # Infer NHWC when H == W and channels exceed the spatial dimensions; otherwise keep NCHW.
        if feat.ndim == 4:
            _, d1, d2, d3 = feat.shape
            if d1 == d2 and d3 > d1:
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def reshape_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """Reshape token features into a spatial grid, accounting for prefix tokens.

        A backbone that declares ``tokens_to_grid`` maps its own token layout instead.
        """
        to_grid: Callable[[torch.Tensor], torch.Tensor] | None = getattr(
            self.backbone, "tokens_to_grid", None
        )
        if to_grid is not None:
            return to_grid(feat)
        bsz, d1, d2 = feat.shape
        n_prefix = _resolve_num_prefix_tokens(self.backbone)

        # Try the declared prefix count before the 0/1-token fallbacks.
        for drop in dict.fromkeys(([n_prefix] if n_prefix else []) + [0, 1]):
            if drop >= d1:
                continue
            side = math.isqrt(d1 - drop)
            if side * side == d1 - drop:
                return feat[:, drop:, :].permute(0, 2, 1).reshape(bsz, d2, side, side)

        # A declared prefix count fixes the token axis; never reinterpret channel width.
        # Otherwise, a square channel width can be mistaken for the patch grid.
        if n_prefix is None and d1 < d2:
            side = math.isqrt(d2)
            if side * side == d2:
                return feat.reshape(bsz, d1, side, side)
            side_no_cls = math.isqrt(d2 - 1) if d2 > 1 else 0
            if side_no_cls * side_no_cls == d2 - 1:
                return feat[:, :, 1:].reshape(bsz, d1, side_no_cls, side_no_cls)

        raise ValueError(
            "Could not reshape 3D feature map to 2D grid. "
            f"Got shape={tuple(feat.shape)}, num_prefix_tokens={n_prefix}. "
            "Expected tokens with L=s^2 after dropping prefix tokens."
        )

    @torch.no_grad()
    def extract_segmentation_features(
        self,
        dataloader: "torch.utils.data.DataLoader",
        cache_dtype: torch.dtype = torch.float16,
    ) -> "CachedFeaturesDataset":
        """Run the frozen backbone once over *dataloader* and cache features.

        Args:
            dataloader: DataLoader that yields ``dict`` or ``(image, mask)`` batches.
            cache_dtype: Storage dtype for cached feature tensors. Use
                ``torch.float16`` (default) to halve RAM, or ``torch.float32``
                for full precision.

        Returns:
            A :class:`CachedFeaturesDataset` with one entry per sample, pooling
            temporal features across dates as in :meth:`forward`.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            batches_per_layer: list[list[torch.Tensor]] = [[] for _ in self.layer_names]
            all_masks: list[torch.Tensor] = []
            device = self._backbone_device()

            for batch in dataloader:
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                    masks = batch["mask"]
                else:
                    images, masks = batch[0].to(device), batch[1]

                steps = 0
                if images.ndim == 5:
                    steps = images.shape[1]
                    images = images.flatten(0, 1)

                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                masks = masks.long()
                self._features.clear()
                _ = self.backbone(images)

                for li, n in enumerate(self.layer_names):
                    feat = self._process_feature(self._features[n])
                    if steps:
                        feat = self._pool_time(feat, steps)
                    batches_per_layer[li].append(feat.to(dtype=cache_dtype, device="cpu"))
                all_masks.append(masks.cpu())
        finally:
            self.backbone.train(was_training)

        layer_tensors = [torch.cat(batches) for batches in batches_per_layer]
        masks_tensor = torch.cat(all_masks)
        logger.info("Cached features for %s samples.", masks_tensor.shape[0])
        return CachedFeaturesDataset(layer_tensors, masks_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute segmentation logits from input images.

        Args:
            x: ``(B, C, H, W)`` images or a ``(B, T, C, H, W)`` time series.
                Each time step is encoded as an image before pooling features over ``T``.
                This gives the head one map per sample.

        Returns:
            Logits tensor of shape ``(B, num_classes, H, W)``.
        """
        input_h, input_w = x.shape[-2:]
        steps = 0
        if x.ndim == 5:
            batch, steps = x.shape[0], x.shape[1]
            x = x.reshape(batch * steps, *x.shape[2:])

        if self.freeze_backbone:
            self.backbone.eval()
            use_amp = x.device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=use_amp):
                _ = self.backbone(x)
        else:
            _ = self.backbone(x)

        features = [self._process_feature(self._features[n]) for n in self.layer_names]
        if steps:
            features = [self._pool_time(f, steps) for f in features]
        return self.head(features, input_h, input_w)

    def _pool_time(self, feat: torch.Tensor, steps: int) -> torch.Tensor:
        """Reduce ``(B*T, C, H, W)`` back to ``(B, C, H, W)`` over time."""
        c, h, w = feat.shape[1:]
        feat = feat.view(-1, steps, c, h, w)
        return feat.mean(dim=1) if self.temporal_pool == "mean" else feat.amax(dim=1)
