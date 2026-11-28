"""SatMAE++ fMoW-RGB encoder (Noman et al., CVPR 2024)."""

import argparse
from pathlib import Path

import torch
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms.functional import normalize

from geo_eval_harness.bands import BandCompatibilityError, BandSpec
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._band_mapping import resolve_src_indices
from ._input_units import InputUnit, detect_input_unit, to_rgb_display_scale
from ._normalization import NormalizationStrategy
from ._pooling import VALID_MODES, pool_tokens
from .interface import BenchModel

_VIT_L = {"embed_dim": 1024, "depth": 24, "num_heads": 16}
_PRETRAINING_PREFIXES = ("decoder_", "mask_token", "proj_up_", "up_block", "head.", "fc_norm.")


class SatMAEPPBench(BenchModel):
    """SatMAE++ ViT-L/16. The released fMoW-RGB checkpoint expects OpenCV BGR channels."""

    wants_resolved_image_size = True
    wants_native_image_size = True
    band_order: tuple[str, ...] = ("blue", "green", "red")
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
