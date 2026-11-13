"""Segmentation probe: multi-scale frozen-backbone feature extraction and head training."""

import logging
import math
from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from geo_eval_harness.config.schema import NECK_HEADS
from geo_eval_harness.models import anyup
from geo_eval_harness.models.segmentation_heads import (
    ConvBlockHead,
    DPTHead,
    FeatureStandardizer,
    FPNHead,
    LinearHead,
    NeckedHead,
    PatchLinearHead,
    StandardizedHead,
    SweepPatchLinearHead,
    UNetHead,
    UPerNetHead,
    ViTNeck,
    feature_moments,
)

logger = logging.getLogger(__name__)

#: Largest share of an epoch's training batches, or of the samples that fit the feature
#: standardizer, that may be skipped for non-finite backbone features.
MAX_NONFINITE_FRACTION = 0.05


class NonFiniteFeaturesError(FloatingPointError):
    """Backbone features contain ``inf`` or ``NaN``, for example after a float16 overflow.

    Raised before the head runs, so the head's normalisation statistics stay untouched.
    """


def check_finite_features(features: list[torch.Tensor]) -> None:
    """Raise :class:`NonFiniteFeaturesError` if any feature map has a non-finite value."""
    if not all(bool(torch.isfinite(f).all()) for f in features):
        raise NonFiniteFeaturesError(
            "Backbone features are not finite (float16 overflow?). Set segmentation.amp: off "
            "to run in float32."
        )


def _finite_samples(features: list[torch.Tensor]) -> torch.Tensor:
    """Return a ``(N,)`` mask of the samples whose features are finite in every layer."""
    return torch.stack([torch.isfinite(f).flatten(1).all(dim=1) for f in features]).all(dim=0)


#: Heads built as ``Head(channels_list, num_classes, hidden_dim=...)``.
_HIDDEN_DIM_HEADS: dict[str, type[nn.Module]] = {
    "conv_block": ConvBlockHead,
    "fpn": FPNHead,
    "dpt": DPTHead,
    "upernet": UPerNetHead,
    "unet": UNetHead,
}


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
        *,
        amp: bool = True,
    ) -> "GPUTensorCache":
        """Move cached features and masks to ``device``.

        Args:
            cache: CPU-resident cached features.
            device: Target device.
            amp: Store float16 features on CUDA for mixed precision; ``False`` keeps float32.

        Returns:
            A :class:`GPUTensorCache` with all data on *device*.
        """
        target_device = torch.device(device)
        # CPU heads need float32 without autocast; CUDA uses float16 for mixed precision.
        dtype = torch.float16 if amp and target_device.type == "cuda" else torch.float32
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
            ``"fpn"``, ``"dpt"``, ``"patch_linear"``, ``"upernet"``, ``"unet"``.
        hidden_dim: Hidden channel dimension for ``conv_block``, ``fpn``,
            ``dpt``, ``upernet``, and ``unet`` heads (default 256).
        neck: ``"none"`` (default), ``"vit"`` or ``"vit_fine"``. ``"vit"`` puts a
            :class:`~geo_eval_harness.models.ViTNeck` before an ``fpn``, ``upernet``, or ``unet``
            head; it needs exactly four layers at one resolution. ``"vit_fine"`` takes the finest
            level from the last layer at its own resolution, so only the first three must share one.
        neck_norm: Per-level input normalisation of the ViT neck, ``"layernorm"`` (default) or
            ``"none"``.
        head_dropout: Channel dropout (``nn.Dropout2d``, as mmsegmentation's UPerHead) before the
            UPerNet classifier; ``0`` (default) adds none.
        upsampler: ``"none"`` (default) or ``"anyup"``. ``"anyup"`` (UPerNet only) puts frozen
            pretrained AnyUp between the fused features and the classifier, guided by the input's
            RGB bands (see :class:`~geo_eval_harness.models.anyup.AnyUpClassifier`).
        sweep_heads: With ``head_type="patch_linear"``, train this many independent probes at
            once as a ``SweepPatchLinearHead`` (one per learning-rate x weight-decay cell).
            ``0`` (default) builds a single ``PatchLinearHead``.
        feature_standardize: Z-score each hooked feature channel with statistics fit on the
            training split (:meth:`fit_standardizer`) before the head. Frozen backbones only.
        amp: Run a frozen backbone under CUDA float16 autocast. ``False`` keeps it in float32.
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
        neck: str = "none",
        neck_norm: str = "layernorm",
        upsampler: str = "none",
        head_dropout: float = 0.0,
        sweep_heads: int = 0,
        feature_standardize: bool = False,
        amp: bool = True,
    ) -> None:
        super().__init__()
        self.amp = amp
        if temporal_pool not in ("mean", "max"):
            raise ValueError(f"temporal_pool must be 'mean' or 'max', got {temporal_pool!r}")
        self.temporal_pool = temporal_pool
        self.backbone = backbone
        self.layer_names = layer_names
        self.freeze_backbone = freeze_backbone
        self.head_type = head_type

        self._features: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

        self.register_hooks()

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.channels_list = self._dry_run_channels()
        hdim = hidden_dim or 256

        self.head: nn.Module
        if head_type == "linear":
            single_map = neck == "vit" and len(self.channels_list) == 1
            self.head = LinearHead(
                self.channels_list * 4 if single_map else self.channels_list, num_classes
            )
        elif head_type in _HIDDEN_DIM_HEADS:
            # A single-layer ViT neck builds four pyramid levels from that one map.
            single_map = neck == "vit" and len(self.channels_list) == 1
            self.head = _HIDDEN_DIM_HEADS[head_type](
                self.channels_list * 4 if single_map else self.channels_list,
                num_classes,
                hidden_dim=hdim,
            )
        elif head_type == "patch_linear":
            self.head = (
                SweepPatchLinearHead(self.channels_list, num_classes, sweep_heads)
                if sweep_heads
                else PatchLinearHead(self.channels_list, num_classes)
            ).to(self._backbone_device())
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
                f"{head_type!r}. Choose from: linear, conv_block, fpn, dpt, patch_linear, upernet, unet"
            )
        if sweep_heads and head_type != "patch_linear":
            raise ValueError(f"sweep_heads requires the patch_linear head, got {head_type!r}.")
        self._attach_neck(neck, head_type, neck_norm)
        self._attach_head_dropout(head_dropout, head_type)
        self._anyup = None
        self._attach_upsampler(upsampler, head_type)
        self.standardizer: FeatureStandardizer | None = None
        if feature_standardize:
            if not self.freeze_backbone:
                raise ValueError("feature_standardize requires a frozen backbone.")
            self.standardizer = FeatureStandardizer(self.channels_list).to(self._backbone_device())
            self.head = StandardizedHead(self.standardizer, self.head)

    @property
    def sweep_head(self) -> SweepPatchLinearHead | None:
        """Return the ``SweepPatchLinearHead``, if the probe trains a sweep."""
        head = self.head.head if isinstance(self.head, StandardizedHead) else self.head
        return head if isinstance(head, SweepPatchLinearHead) else None

    def _require_standardizer(self) -> FeatureStandardizer:
        if self.standardizer is None:
            raise RuntimeError("This probe was built without feature_standardize.")
        return self.standardizer

    @torch.no_grad()
    def fit_standardizer(
        self,
        train: "CachedFeaturesDataset | torch.utils.data.DataLoader",
        chunk_size: int = 256,
    ) -> None:
        """Fit the feature standardizer on every position of the training split.

        ``train`` is either the cached training features or the training loader; a loader costs
        one extra backbone pass. Both feed the same exact float64 moment accumulation.
        Samples with non-finite features are left out, like non-finite training batches; more
        than ``MAX_NONFINITE_FRACTION`` of them raises :class:`NonFiniteFeaturesError`.
        """
        standardizer = self._require_standardizer()
        sums: list[torch.Tensor] = []
        sq_sums: list[torch.Tensor] = []
        counts: list[int] = []
        samples = dropped = 0
        for chunk in self._standardizer_features(train, chunk_size):
            finite = _finite_samples(chunk)
            samples += len(finite)
            features = chunk
            if not bool(finite.all()):
                dropped += int((~finite).sum())
                features = [layer[finite] for layer in chunk]
            if not sums:
                sums = [
                    torch.zeros(f.shape[1], dtype=torch.float64, device=f.device) for f in features
                ]
                sq_sums = [torch.zeros_like(total) for total in sums]
                counts = [0] * len(features)
            for index, layer in enumerate(features):
                total, sq_total, count = feature_moments(layer)
                sums[index] += total
                sq_sums[index] += sq_total
                counts[index] += count
        if dropped:
            logger.warning(
                "Feature standardization left out %d of %d samples with non-finite features.",
                dropped,
                samples,
            )
            if dropped > MAX_NONFINITE_FRACTION * samples:
                raise NonFiniteFeaturesError(
                    f"Feature standardization left out {dropped} of {samples} samples with "
                    f"non-finite backbone features (limit {MAX_NONFINITE_FRACTION:.0%}). Set "
                    "segmentation.amp: off to run in float32."
                )
        if not sums:
            raise ValueError("Feature standardization needs a non-empty training split.")
        standardizer.set_moments(sums, sq_sums, counts)
        logger.info("Feature standardization fit on %d training positions.", counts[0])

    def _standardizer_features(
        self, train: "CachedFeaturesDataset | torch.utils.data.DataLoader", chunk_size: int
    ) -> Iterator[list[torch.Tensor]]:
        """Yield per-layer training features from the cache in chunks, or from the backbone."""
        device = self._backbone_device()
        if isinstance(train, CachedFeaturesDataset):
            for start in range(0, len(train), chunk_size):
                yield [
                    layer[start : start + chunk_size].to(device) for layer in train.layer_tensors
                ]
            return
        was_training = self.backbone.training
        try:
            for batch in train:
                images = (batch["image"] if isinstance(batch, dict) else batch[0]).to(device)
                yield self._backbone_features(images)
        finally:
            self.backbone.train(was_training)

    def _backbone_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the backbone on images or a time series and return the hooked feature maps."""
        steps = 0
        if x.ndim == 5:
            batch, steps = x.shape[0], x.shape[1]
            x = x.reshape(batch * steps, *x.shape[2:])
        self._features.clear()
        self._run_backbone(x)
        features = [self._process_feature(self._features[n]) for n in self.layer_names]
        if steps:
            features = [self._pool_time(f, steps) for f in features]
        return features

    def _run_backbone(self, x: torch.Tensor) -> None:
        """Run the backbone to fill the hooks; a frozen one runs under CUDA autocast if ``amp``."""
        if self.freeze_backbone:
            self.backbone.eval()
            use_amp = self.amp and x.device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=use_amp):
                _ = self.backbone(x)
        else:
            _ = self.backbone(x)

    def _upernet_decoder(self) -> nn.Module:
        upernet = self.head.head if isinstance(self.head, NeckedHead) else self.head
        return cast(nn.Module, upernet.decoder)

    def _attach_head_dropout(self, head_dropout: float, head_type: str) -> None:
        """Put channel dropout before the UPerNet classifier when requested."""
        if not head_dropout:
            return
        if head_type != "upernet":
            raise ValueError(f"head_dropout requires the upernet head, got {head_type!r}.")
        decoder = self._upernet_decoder()
        classifier = cast(nn.Module, decoder.classifier)
        decoder.classifier = nn.Sequential(nn.Dropout2d(head_dropout), classifier)

    def _attach_upsampler(self, upsampler: str, head_type: str) -> None:
        """Wrap the UPerNet classifier (after any head dropout) with frozen AnyUp when requested."""
        if upsampler == "none":
            return
        if upsampler != "anyup":
            raise ValueError(f"Unknown upsampler: {upsampler!r}. Choose from: none, anyup")
        if head_type != "upernet":
            raise ValueError(f"upsampler 'anyup' requires the upernet head, got {head_type!r}.")
        decoder = self._upernet_decoder()
        self._anyup = anyup.AnyUpClassifier(
            decoder.get_submodule("classifier"), anyup.load_anyup().to(self._backbone_device())
        )
        decoder.classifier = self._anyup
        self._anyup_rgb = anyup.rgb_guidance_indices(list(getattr(self.backbone, "bands", [])))
        logger.info("AnyUp before the classifier, guidance channels %s", self._anyup_rgb)

    def _attach_neck(self, neck: str, head_type: str, neck_norm: str = "layernorm") -> None:
        """Put the requested neck before the head after checking the hooked feature maps."""
        if neck == "none":
            if head_type == "unet" and len(set(self.feature_hw_list)) == 1:
                raise ValueError(
                    "head 'unet' needs a feature pyramid; the hooked layers share one resolution "
                    f"{self.feature_hw_list[0]}. Use neck 'vit' for ViT blocks."
                )
            return
        if neck not in ("vit", "vit_fine"):
            raise ValueError(f"Unknown neck: {neck!r}. Choose from: none, vit, vit_fine")
        if head_type not in NECK_HEADS:
            raise ValueError(
                f"neck {neck!r} requires head in {sorted(NECK_HEADS)}, got {head_type!r}."
            )
        trunk_hw = self.feature_hw_list[:3] if neck == "vit_fine" else self.feature_hw_list
        single = neck == "vit" and len(self.layer_names) == 1
        if not single and (len(self.layer_names) != 4 or len(set(trunk_hw)) != 1):
            raise ValueError(
                f"neck {neck!r} needs exactly 4 layers sharing one resolution (ViT blocks; with "
                f"'vit_fine' only the first three must; 'vit' also takes a single layer); got "
                f"{len(self.layer_names)} layers with sizes {self.feature_hw_list}. Pyramid "
                "backbones already provide multi-scale features."
            )
        self.head = NeckedHead(
            ViTNeck(self.channels_list, input_norm=neck_norm, native_finest=neck == "vit_fine"),
            self.head,
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
        # Backbones built for one fixed input size declare it; everything else accepts 224.
        side = int(getattr(self.backbone, "probe_input_size", None) or 224)
        dummy = torch.randn(1, in_channels, side, side, device=device)
        self.dry_run_input_hw = (side, side)
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
        # Backbones that declare NHWC stage maps skip the shape guess, which fails once the grid
        # is at least as wide as the channels (Swin stage 0 at 512). Undeclared backbones are
        # read as NHWC when H == W and channels exceed the spatial dimensions, else NCHW.
        if feat.ndim == 4:
            if getattr(self.backbone, "feature_layout", None) == "nhwc":
                return feat.permute(0, 3, 1, 2).contiguous()
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

        self._run_backbone(x)

        features = [self._process_feature(self._features[n]) for n in self.layer_names]
        if steps:
            features = [self._pool_time(f, steps) for f in features]
        check_finite_features(features)
        if self._anyup is not None:
            if steps:
                raise ValueError("upsampler 'anyup' does not support time series inputs.")
            self._anyup.guidance = anyup.anyup_guidance(x, self._anyup_rgb)
            try:
                return self.head(features, input_h, input_w)
            finally:
                self._anyup.guidance = None
        return self.head(features, input_h, input_w)

    def _pool_time(self, feat: torch.Tensor, steps: int) -> torch.Tensor:
        """Reduce ``(B*T, C, H, W)`` back to ``(B, C, H, W)`` over time."""
        c, h, w = feat.shape[1:]
        feat = feat.view(-1, steps, c, h, w)
        return feat.mean(dim=1) if self.temporal_pool == "mean" else feat.amax(dim=1)
