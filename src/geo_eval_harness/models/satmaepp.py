"""SatMAE++ fMoW-RGB encoder (Noman et al., CVPR 2024)."""

import argparse
from pathlib import Path

import torch
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms.functional import normalize

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._band_mapping import resolve_src_indices
from ._input_units import InputUnit, detect_input_unit, to_rgb_display_scale
from ._normalization import NormalizationStrategy
from ._pooling import VALID_MODES, pool_tokens
from .interface import BenchModel
from .smarties import _sincos_pos_embed

_VIT_L = {"embed_dim": 1024, "depth": 24, "num_heads": 16}
_PRETRAINING_HEADS = ("decoder_", "mask_token", "proj_up_", "up_block", "head.", "fc_norm.")


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
        checkpoint_path: str | None = None,
        pool: str = "mean",
        allow_missing: bool = False,
        normalization: str = "bandspec_zscore",
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
        if missing and not allow_missing:
            raise BandCompatibilityError(
                f"SatMAE++ requires {missing}; available bands: {sorted(source)}."
            )
        self._indices = [source.get(name, len(bands)) for name in self.band_order]
        self._pad_missing = bool(missing)
        self._unit = None
        if self.handles_own_normalization:
            selected = [bands[i] for i in self._indices if i < len(bands)]
            if not selected:
                raise BandCompatibilityError(
                    "SatMAE++ model normalization needs at least one matching band."
                )
            self._unit = detect_input_unit(selected)
            if self.native_units and self._unit not in self.native_units:
                raise UnsupportedNormalizationError(
                    f"{self.checkpoint} does not support model normalization for {self._unit}. "
                    "Use --normalization dataset."
                )
        self.backbone = self._build_encoder(size)
        if pretrained:
            filename = self.checkpoint + ".pth"
            path = fetch_pinned_file(
                f"mubashir04/{self.checkpoint}",
                filename,
                self.revision,
                load_checksums("satmaepp.sha256")[filename],
                local_path=checkpoint_path,
            )
            self._load_checkpoint(path)

    def _build_encoder(self, size: int) -> VisionTransformer:
        """Construct the fMoW-RGB ViT with the MAE spatial encoding."""
        encoder = VisionTransformer(
            img_size=size, patch_size=self.patch_size, num_classes=0, **_VIT_L
        )
        encoder.pos_embed = torch.nn.Parameter(
            _sincos_pos_embed(encoder.embed_dim, self.grid_size), requires_grad=False
        )
        return encoder

    def _load_checkpoint(self, path: Path) -> None:
        """Load encoder tensors strictly, discarding the pretraining heads."""
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = {
            k: v
            for k, v in checkpoint.get("model", checkpoint).items()
            if not k.startswith(_PRETRAINING_HEADS)
        }
        state["pos_embed"] = resample_abs_pos_embed(
            state["pos_embed"], [self.grid_size, self.grid_size], antialias=False
        )
        self.backbone.load_state_dict(state, strict=True)

    def _native_pixels(self, images: torch.Tensor, unit: InputUnit) -> torch.Tensor:
        """Apply the checkpoint's display scale and fMoW channel statistics."""
        return normalize(
            to_rgb_display_scale(images, unit),
            [0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
            [0.28774282336235046, 0.27541765570640564, 0.2764017581939697],
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Select checkpoint bands and apply the requested normalization."""
        images = super().normalize_inputs(images)
        if self._pad_missing:
            images = torch.nn.functional.pad(images, (0, 0, 0, 0, 0, 1))
        images = images[:, self._indices]
        return images if self._unit is None else self._native_pixels(images, self._unit)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return pool_tokens(self.backbone.forward_features(images), self.pool, num_prefix_tokens=1)
