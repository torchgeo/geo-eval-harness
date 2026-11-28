"""SatMAE++ fMoW-RGB and fMoW-Sentinel encoders (Noman et al., CVPR 2024)."""

import argparse
from pathlib import Path

import torch
from timm.layers import PatchEmbed, resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision.transforms.functional import normalize

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._band_mapping import resolve_src_indices
from ._input_units import InputUnit, detect_input_unit, to_rgb_display_scale, to_s2_dn
from ._normalization import NormalizationStrategy
from ._pooling import VALID_MODES, pool_tokens
from .interface import BenchModel

_VIT_L = {"embed_dim": 1024, "depth": 24, "num_heads": 16}
_PRETRAINING_PREFIXES = ("decoder_", "mask_token", "proj_up_", "up_block", "head.", "fc_norm.")

# fMoW-Sentinel groups 10 m BGR + NIR, 20 m red-edge + narrow NIR, and 20 m SWIR.
_S2_GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))

# Training DN statistics in the checkpoint's band order (techmn/satmae_pp, util/datasets.py).
_S2_DN_STATS = {
    "blue": (1184.3824625, 650.2842772),
    "green": (1120.77120066, 712.12507725),
    "red": (1136.26026392, 965.23119807),
    "rededge1": (1263.73947144, 948.9819932),
    "rededge2": (1645.40315151, 1108.06650639),
    "rededge3": (1846.87040806, 1258.36394548),
    "nir": (1762.59530783, 1233.1492281),
    "nir_narrow": (1972.62420416, 1364.38688993),
    "swir1": (1732.16362238, 1310.36996126),
    "swir2": (1247.91870117, 1087.6020813),
}
_S2_DN_LOW = [mean - 2 * std for mean, std in _S2_DN_STATS.values()]
_S2_DN_SPAN = [4 * std for _, std in _S2_DN_STATS.values()]


class SatMAEPPBench(BenchModel):
    """SatMAE++ ViT-L/16. The released fMoW-RGB checkpoint expects OpenCV BGR channels."""

    wants_resolved_image_size = True
    wants_native_image_size = True
    band_order: tuple[str, ...] = ("blue", "green", "red")
    native_units: tuple[InputUnit, ...] | None = None
    patch_size = 16
    default_image_size = 224
    checkpoint = "checkpoint_ViT-L_pretrain_fmow_rgb"
    revision = "8f4c97f767fb633c635e0c23f6e1441be9487f0f"

    def __init__(  # noqa: PLR0913 - benchmark constructor options
        self,
        bands: list[BandSpec],
        *,
        image_size: int | None = None,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        pool: str = "mean",
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
    ) -> None:
        if pool not in VALID_MODES:
            raise ValueError(f"pool must be one of {VALID_MODES}, got {pool!r}.")
        self.handles_own_normalization = normalization == NormalizationStrategy.MODEL_NATIVE
        super().__init__(bands, normalization=normalization)
        self.pool = pool
        size = self.default_image_size if image_size is None else image_size
        if size <= 0 or size % self.patch_size:
            raise ValueError(
                f"image_size must be a positive multiple of {self.patch_size}, got {size}."
            )
        self.probe_input_size = size
        self.grid_size = size // self.patch_size
        source = resolve_src_indices(bands, preferred_sensors=("s2",))
        missing = [name for name in self.band_order if name not in source]
        if missing:
            raise BandCompatibilityError(
                f"SatMAE++ requires {missing}; available bands: {sorted(source)}."
            )
        self._indices = [source[name] for name in self.band_order]
        self._unit = (
            detect_input_unit([bands[i] for i in self._indices])
            if self.handles_own_normalization
            else None
        )
        if self._unit is not None and self.native_units and self._unit not in self.native_units:
            expected = ", ".join(self.native_units)
            raise UnsupportedNormalizationError(
                f"{type(self).__name__} model normalization expects {expected}, got {self._unit}. "
                "Use --normalization dataset."
            )
        self.backbone = self._build_encoder(size)
        if pretrained:
            filename = f"{self.checkpoint}.pth"
            path = fetch_pinned_file(
                f"mubashir04/{self.checkpoint}",
                filename,
                self.revision,
                load_checksums("satmaepp.sha256")[filename],
                local_path=checkpoint_path,
            )
            self._load_checkpoint(path)

    def _build_encoder(self, size: int) -> VisionTransformer:
        """Construct the fMoW-RGB ViT at the resolved image size."""
        encoder = VisionTransformer(
            img_size=size, patch_size=self.patch_size, num_classes=0, **_VIT_L
        )
        assert encoder.pos_embed is not None
        encoder.pos_embed.requires_grad_(False)
        return encoder

    def _load_checkpoint(self, path: Path) -> None:
        """Load encoder tensors strictly, discarding pretraining modules."""
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = {
            k: v
            for k, v in checkpoint.get("model", checkpoint).items()
            if not k.startswith(_PRETRAINING_PREFIXES)
        }
        state["pos_embed"] = resample_abs_pos_embed(
            state["pos_embed"], [self.grid_size, self.grid_size], antialias=False
        )
        self.backbone.load_state_dict(state, strict=True)

    def _native_pixels(self, images: torch.Tensor, unit: InputUnit) -> torch.Tensor:
        """Apply the checkpoint's display scale and fMoW BGR channel statistics."""
        return normalize(
            to_rgb_display_scale(images, unit),
            [0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
            [0.28774282336235046, 0.27541765570640564, 0.2764017581939697],
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Select checkpoint bands and apply the requested normalization."""
        images = super().normalize_inputs(images)[:, self._indices]
        return images if self._unit is None else self._native_pixels(images, self._unit)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return pool_tokens(self.backbone.forward_features(images), self.pool, num_prefix_tokens=1)


class _GroupedPatchEmbed(nn.ModuleList):
    """Apply one checkpoint patch projection per Sentinel-2 spectral group."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return group-major patch tokens ``(B, G, L, D)``."""
        return torch.stack(
            [proj(images[:, group]) for group, proj in zip(_S2_GROUPS, self, strict=True)], dim=1
        )


class _GroupedViT(VisionTransformer):
    """ViT-L/8 with separate spatial and spectral group position embeddings."""

    def __init__(self, size: int, patch_size: int) -> None:
        super().__init__(img_size=size, patch_size=patch_size, num_classes=0, **_VIT_L)
        self.patch_embed = _GroupedPatchEmbed(
            PatchEmbed(size, patch_size, len(group), self.embed_dim) for group in _S2_GROUPS
        )
        channel_dim = self.embed_dim // 4
        assert self.pos_embed is not None
        self.pos_embed = nn.Parameter(
            self.pos_embed[:, :, :-channel_dim].detach().clone(), requires_grad=False
        )
        self.channel_embed = nn.Parameter(
            torch.randn(1, len(_S2_GROUPS), channel_dim) * 0.02, requires_grad=False
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Add spatial and spectral positions before prepending the class token."""
        batch, groups, count, _ = x.shape
        assert self.pos_embed is not None
        spatial = self.pos_embed[:, None, 1:].expand(-1, groups, -1, -1)
        spectral = self.channel_embed[:, :, None].expand(-1, -1, count, -1)
        tokens = x + torch.cat((spatial, spectral), dim=-1)
        assert self.cls_token is not None
        return torch.cat((self.cls_token.expand(batch, -1, -1), tokens.flatten(1, 2)), dim=1)


class SatMAEPPSentinelBench(SatMAEPPBench):
    """SatMAE++ grouped ViT-L/8 for the ten fMoW-Sentinel bands."""

    band_order = tuple(_S2_DN_STATS)
    native_units = (InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1)
    patch_size = 8
    num_patch_groups = len(_S2_GROUPS)
    default_image_size = 96
    checkpoint = "checkpoint_ViT-L_pretrain_fmow_sentinel"
    revision = "e6792ae99987b96569993306b396c232f27ba8cc"

    def _build_encoder(self, size: int) -> VisionTransformer:
        """Construct the grouped fMoW-Sentinel ViT at the resolved image size."""
        return _GroupedViT(size, self.patch_size)

    def _native_pixels(self, images: torch.Tensor, unit: InputUnit) -> torch.Tensor:
        """Apply fMoW-Sentinel's per-band two-sigma clipping and 8-bit quantization."""
        scaled = normalize(to_s2_dn(images, unit), _S2_DN_LOW, _S2_DN_SPAN)
        return scaled.clamp_(0, 1).mul_(255).floor_().div_(255)

    def tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Average corresponding spatial patches across the three spectral groups."""
        patches = tokens[:, 1:].reshape(
            tokens.shape[0], self.num_patch_groups, self.grid_size, self.grid_size, -1
        )
        return patches.mean(dim=1).permute(0, 3, 1, 2)
