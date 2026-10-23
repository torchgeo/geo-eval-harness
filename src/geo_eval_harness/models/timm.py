"""Timm backbone wrapper for patch-level feature extraction."""

import logging
from dataclasses import dataclass
from typing import Any

import timm
import torch
import torch.nn.functional as F

from geo_eval_harness.datasets.base import BandSpec

from ._normalization import InputUnit, NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

_VALID_INPUT_NORMALIZATIONS = ("bands_zscore", "imagenet", "timm_default", "none")

# Wavelength windows (micrometres) identify RGB despite names such as "red" and "b04".
_BLUE_UM = (0.45, 0.515)
_GREEN_UM = (0.515, 0.60)
_RED_UM = (0.60, 0.70)


def _rgb_first_permutation(bands: list[BandSpec]) -> list[int] | None:
    """Return a channel permutation with red/green/blue moved to the front.

    ``None`` if the band list doesn't contain exactly one band in each of
    the red/green/blue wavelength windows -- e.g. a SAR-only or otherwise
    non-optical band set, where there is nothing to reorder.
    """
    windows = {"red": _RED_UM, "green": _GREEN_UM, "blue": _BLUE_UM}
    matches: dict[str, list[int]] = {"red": [], "green": [], "blue": []}
    for i, band in enumerate(bands):
        if band.wavelength_um is None:
            continue
        for key, (lo, hi) in windows.items():
            if lo <= band.wavelength_um < hi:
                matches[key].append(i)

    if any(len(idx) != 1 for idx in matches.values()):
        return None

    rgb_idx = [matches["red"][0], matches["green"][0], matches["blue"][0]]
    rest = [i for i in range(len(bands)) if i not in rgb_idx]
    return rgb_idx + rest


class TimmPatchBenchModel(BenchModel):
    """BenchModel wrapper for any timm backbone.

    Args:
        bands: Ordered :class:`BandSpec` list (channel count = ``len(bands)``).
        model_name: Any timm model name (e.g. ``"resnet50"``,
            ``"convnext_small"``, ``"vit_base_patch16_224"``).
        pretrained: Load pretrained weights when available.
        normalize: If ``True``, L2-normalize the output embedding.
        global_pool: Global pooling strategy for timm headless models.
        use_cls_token: For ViT-family models, use the CLS token instead of
            averaging spatial tokens.
        auto_resize: If ``True``, bilinearly resize each batch to ``target_size``.
        target_size: Square target size; auto-inferred from
            ``backbone.default_cfg["input_size"]`` when not given.
        input_normalization: One of ``"bands_zscore"`` (default; per-channel
            z-score using :class:`BandSpec` statistics — correct for raw
            remote-sensing data of any channel count), ``"imagenet"``
            (per-channel rescale to ``[0, 1]`` using ``BandSpec.{min, max}``,
            then ``(x - mean) / std`` with ImageNet RGB stats; refuses to
            instantiate when ``len(bands) != 3``), ``"timm_default"`` (same
            shape as ``"imagenet"`` but reads ``backbone.default_cfg``'s
            ``mean``/``std``; also requires ``len(bands) == 3``), or
            ``"none"`` (identity).
    """

    _rgb_mean: torch.Tensor
    _rgb_std: torch.Tensor
    _band_min: torch.Tensor
    _band_range: torch.Tensor

    #: Constructor kwargs come from user YAML, and __init__ ignores unknown
    #: keys, so validate them instead of silently dropping a typo.
    validated_settings: "type[TimmModelSettings]"

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        model_name: str,
        pretrained: bool = True,
        normalize: bool = False,
        global_pool: str | None = "avg",
        auto_resize: bool = False,
        target_size: int | None = None,
        use_cls_token: bool = False,
        input_normalization: str = "bands_zscore",
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_kwargs: object,
    ) -> None:
        # Set pretraining stats before BenchModel builds its model_native normalizer.
        self.set_pretrain_stats(bands, model_name)

        super().__init__(bands=bands, normalization=normalization, **_kwargs)

        if input_normalization not in _VALID_INPUT_NORMALIZATIONS:
            raise ValueError(
                f"input_normalization must be one of {_VALID_INPUT_NORMALIZATIONS}, "
                f"got {input_normalization!r}."
            )

        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self.auto_resize = auto_resize
        self.use_cls_token = use_cls_token
        self.input_normalization = input_normalization

        if self.use_cls_token and global_pool != "":
            global_pool = ""

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=self.num_channels,
            num_classes=0,
            global_pool=global_pool,
        )

        # timm.models._manipulate.adapt_input_conv tiles [R, G, B] weights for extra channels.
        # Put physical RGB first to match the weights; native S2 order is blue/green/red.
        self._channel_perm: list[int] | None = None
        if self.pretrained and self.num_channels != 3:
            self._channel_perm = _rgb_first_permutation(self.bands)

        default_cfg = getattr(self.backbone, "default_cfg", {}) or {}
        cfg_input_size = default_cfg.get("input_size", None)
        inferred_size: int | None = None
        if isinstance(cfg_input_size, (list, tuple)) and len(cfg_input_size) == 3:
            inferred_size = int(cfg_input_size[1])
        self.target_size = int(target_size) if target_size is not None else inferred_size

        if self.auto_resize and self.target_size is None:
            logger.warning(
                "auto_resize=True but target_size could not be inferred for %s; "
                "disabling auto-resize.",
                model_name,
            )
            self.auto_resize = False

        self.register_rgb_normalization(default_cfg)

    def set_pretrain_stats(self, bands: list[BandSpec], model_name: str) -> None:
        """Set RGB statistics before BenchModel creates its normalizer."""
        if len(bands) == 3:
            cfg = timm.get_pretrained_cfg(model_name)
            if cfg is None:
                raise RuntimeError(
                    f"timm has no pretrained cfg for {model_name!r}; check the "
                    f"model name (was the wrapper called with a fake/scratch model?)."
                )
            if cfg.mean is not None and cfg.std is not None:
                self.expected_input_unit = InputUnit.REFLECTANCE_0_1
                self.pretrain_mean = list(cfg.mean)
                self.pretrain_std = list(cfg.std)

    def register_rgb_normalization(self, default_cfg: dict[str, Any]) -> None:
        """Register fixed RGB statistics and dataset ranges as buffers."""
        if self.input_normalization in ("imagenet", "timm_default"):
            if self.num_channels != 3:
                raise ValueError(
                    f"input_normalization={self.input_normalization!r} requires 3 input "
                    f"channels (the ImageNet RGB stats are 3-D); got {self.num_channels}. "
                    "Use 'bands_zscore' for multispectral data."
                )

            if self.input_normalization == "imagenet":
                cfg_mean = (0.485, 0.456, 0.406)
                cfg_std = (0.229, 0.224, 0.225)
            else:
                cfg_mean = default_cfg.get("mean")
                cfg_std = default_cfg.get("std")
                if cfg_mean is None or cfg_std is None:
                    raise ValueError(
                        f"input_normalization='timm_default' but timm '{self.model_name}' has no "
                        "default_cfg mean/std — pick a different normalization mode."
                    )

            mean = torch.tensor(cfg_mean, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(cfg_std, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("_rgb_mean", mean)
            self.register_buffer("_rgb_std", std)

            spec_min = torch.tensor([b.min for b in self.bands], dtype=torch.float32).view(
                1, self.num_channels, 1, 1
            )
            spec_max = torch.tensor([b.max for b in self.bands], dtype=torch.float32).view(
                1, self.num_channels, 1, 1
            )
            spec_range = (spec_max - spec_min).clamp_min(1e-8)
            self.register_buffer("_band_min", spec_min)
            self.register_buffer("_band_range", spec_range)

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalization policy."""
        mode = self.input_normalization
        if mode == "bands_zscore":
            return super().normalize_inputs(images)
        if mode == "none":
            return images
        # ImageNet/timm statistics expect [0, 1] inputs, so rescale raw bands first.
        band_min = self._band_min.to(dtype=images.dtype)
        band_range = self._band_range.to(dtype=images.dtype)
        scaled = (images - band_min) / band_range
        mean = self._rgb_mean.to(dtype=images.dtype)
        std = self._rgb_std.to(dtype=images.dtype)
        return (scaled - mean) / std

    def _permute_channels(self, images: torch.Tensor) -> torch.Tensor:
        """Reorder input channels so red/green/blue lead, if identifiable."""
        if self._channel_perm is None:
            return images
        return images[:, self._channel_perm, :, :]

    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return pooled patch embeddings of shape ``(B, K)`` from normalized inputs."""
        images = self._permute_channels(images)
        if self.auto_resize and self.target_size is not None:
            h, w = images.shape[-2], images.shape[-1]
            if h != self.target_size or w != self.target_size:
                images = F.interpolate(
                    images,
                    size=(self.target_size, self.target_size),
                    mode="bilinear",
                    align_corners=False,
                )
        x = self.backbone(images)

        # timm returns (B, C) for pooled models, (B, C, h, w) or (B, N, C) otherwise.
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.ndim == 3:
            if self.use_cls_token and self._has_cls_token_like():
                x = x[:, 0, :]
            elif x.shape[1] > 1:
                x = x[:, 1:, :] if self._has_cls_token_like() else x
                x = x.mean(dim=1)
            else:
                x = x.squeeze(1)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def _has_cls_token_like(self) -> bool:
        """Cheap heuristic for ViT/DeiT/MAE-style backbones with a CLS token."""
        return any(hasattr(self.backbone, attr) for attr in ("cls_token", "dist_token"))


@dataclass(frozen=True)
class TimmModelSettings:
    """Validated constructor settings for :class:`TimmPatchBenchModel`."""

    model_name: str
    pretrained: bool = True
    global_pool: str | None = "avg"
    input_normalization: str = "bands_zscore"
    normalize: bool = False
    auto_resize: bool = False
    target_size: int | None = None
    use_cls_token: bool = False

    def __post_init__(self) -> None:
        """Validate settings before importing or constructing a backbone."""
        for name in ("pretrained", "normalize", "auto_resize", "use_cls_token"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.target_size is not None:
            if not isinstance(self.target_size, int):
                raise TypeError("target_size must be an integer")
            if self.target_size <= 0:
                raise ValueError("target_size must be positive")
        if self.global_pool not in (None, "", "avg", "max", "avgmax", "catavgmax"):
            raise ValueError(f"unsupported global_pool: {self.global_pool!r}")
        if self.input_normalization not in _VALID_INPUT_NORMALIZATIONS:
            raise ValueError(f"unsupported input_normalization: {self.input_normalization!r}")

    def build(
        self, bands: list[BandSpec], *, normalization: str = "bandspec_zscore"
    ) -> TimmPatchBenchModel:
        """Construct the wrapper these settings describe."""
        return TimmPatchBenchModel(bands=bands, normalization=normalization, **vars(self))


TimmPatchBenchModel.validated_settings = TimmModelSettings
