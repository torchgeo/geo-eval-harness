"""SatMAE++ encoder wrapper for geo-eval-harness (Noman et al., CVPR 2024).

Implements inference-only encoders matching ``techmn/satmae_pp`` (Apache-2.0):

* ``rgb_vit_l``: ViT-L/16 pretrained on fMoW-RGB at 224 px. Upstream pretrained and
  evaluated fMoW-RGB on OpenCV ``cv2.imread`` images in **BGR** channel order
  (``blue``, ``green``, ``red``; see ``techmn/satmae_pp`` README and issue #2). This
  wrapper maps input bands to BGR automatically by canonical band name.
* ``s2_vit_l``: Grouped-channel ViT-L/8 pretrained on 10-band fMoW-Sentinel at 96 px
  (dropping B01 coastal, B09 water vapor, and B10 cirrus). Bands are grouped into
  10 m RGB+NIR ``(B02, B03, B04, B08)``, 20 m Red-Edge ``(B05, B06, B07, B8A)``,
  and 20 m SWIR ``(B11, B12)``, each with its own patch projection and 256-d group
  embedding concatenated onto the 768-d 2-D sin-cos spatial embedding.

Official Apache-2.0 checkpoints are downloaded from pinned Hugging Face commits and
verified against ``checksums/satmaepp.sha256``.
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from geo_eval_harness.bands import BandSpec
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._band_mapping import BandMappingPolicy, map_to_model_bands
from ._input_units import InputUnit, detect_input_unit, to_rgb_display_scale, to_s2_dn
from ._normalization import NormalizationStrategy, build_normalizer
from ._pooling import VALID_MODES
from .interface import BenchModel
from .torchgeo_models import _auto_resize

logger = logging.getLogger(__name__)

#: fMoW-RGB channel order expected by the released SatMAE++ RGB weights (OpenCV BGR).
SATMAEPP_RGB_BANDS: tuple[str, ...] = ("blue", "green", "red")

#: 10 Sentinel-2 bands retained by fMoW-Sentinel (13-band S2 minus B01, B09, B10).
SATMAEPP_S2_BANDS: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "swir1",
    "swir2",
)

#: Spectral channel groups over :data:`SATMAEPP_S2_BANDS`:
#: 10 m RGB+NIR, 20 m Red-Edge, and 20 m SWIR.
SATMAEPP_S2_CHANNEL_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 6),
    (3, 4, 5, 7),
    (8, 9),
)

#: Per-slot fMoW-RGB mean/std from ``util/datasets.py`` applied to ``[0, 1]`` BGR inputs.
SATMAEPP_RGB_MEAN: tuple[float, ...] = (
    0.4182007312774658,
    0.4214799106121063,
    0.3991275727748871,
)
SATMAEPP_RGB_STD: tuple[float, ...] = (
    0.28774282336235046,
    0.27541765570640564,
    0.2764017581939697,
)

#: fMoW-Sentinel DN mean/std for :data:`SATMAEPP_S2_BANDS` (indices 1..8, 11..12 of S2 L1C).
SATMAEPP_S2_MEAN: tuple[float, ...] = (
    1184.3824625,
    1120.77120066,
    1136.26026392,
    1263.73947144,
    1645.40315151,
    1846.87040806,
    1762.59530783,
    1972.62420416,
    1732.16362238,
    1247.91870117,
)
SATMAEPP_S2_STD: tuple[float, ...] = (
    650.2842772,
    712.12507725,
    965.23119807,
    948.9819932,
    1108.06650639,
    1258.36394548,
    1233.1492281,
    1364.38688993,
    1310.36996126,
    1087.6020813,
)

#: Non-encoder keys shipped in SatMAE++ MAE pretraining checkpoints (decoder + upsample blocks).
_IGNORED_CHECKPOINT_PREFIXES: tuple[str, ...] = (
    "decoder_",
    "mask_token",
    "proj_up_",
    "up_block",
    "head.",
    "fc_norm.",
    "channel_cls_embed",
)


@dataclass(frozen=True)
class SatMAEPPVariant:
    """Architecture and pinned Hugging Face weights of one SatMAE++ checkpoint."""

    modality: Literal["rgb", "s2"]
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int
    default_img_size: int
    channel_embed: int
    channel_groups: tuple[tuple[int, ...], ...] | None
    repo: str
    filename: str
    revision: str


SATMAEPP_VARIANTS: dict[str, SatMAEPPVariant] = {
    "rgb_vit_l": SatMAEPPVariant(
        modality="rgb",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        default_img_size=224,
        channel_embed=0,
        channel_groups=None,
        repo="mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb",
        filename="checkpoint_ViT-L_pretrain_fmow_rgb.pth",
        revision="8f4c97f767fb633c635e0c23f6e1441be9487f0f",
    ),
    "s2_vit_l": SatMAEPPVariant(
        modality="s2",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=8,
        default_img_size=96,
        channel_embed=256,
        channel_groups=SATMAEPP_S2_CHANNEL_GROUPS,
        repo="mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel",
        filename="checkpoint_ViT-L_pretrain_fmow_sentinel.pth",
        revision="e6792ae99987b96569993306b396c232f27ba8cc",
    ),
}


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Build 1-D sine-cosine positional embeddings of shape ``(M, embed_dim)``."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}.")
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, *, cls_token: bool = True
) -> np.ndarray:
    """Build 2-D sine-cosine positional embeddings matching ``techmn/satmae_pp``."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}.")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # Upstream meshgrids (w, h), so the first half encodes column indices.
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_w, emb_h], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def interpolate_pos_embed(pos_embed: torch.Tensor, new_grid_size: int) -> torch.Tensor:
    """Bicubically interpolate spatial ``pos_embed`` tokens while keeping the CLS slot.

    Works identically for standard ViT embeddings ``(1, 1 + L, D)`` and grouped-channel
    spatial embeddings ``(1, 1 + L, D - channel_embed)``.
    """
    extra_tokens = pos_embed[:, :1, :]
    spatial = pos_embed[:, 1:, :]
    num_tokens = spatial.shape[1]
    orig_size = round(num_tokens**0.5)
    if orig_size * orig_size != num_tokens:
        raise ValueError(f"pos_embed spatial token count {num_tokens} is not a perfect square.")
    if orig_size == new_grid_size:
        return pos_embed
    dim = spatial.shape[-1]
    spatial_2d = spatial.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    resized = F.interpolate(
        spatial_2d.float(),
        size=(new_grid_size, new_grid_size),
        mode="bicubic",
        align_corners=False,
    ).to(pos_embed.dtype)
    return torch.cat((extra_tokens, resized.permute(0, 2, 3, 1).flatten(1, 2)), dim=1)


def _validate_img_size(img_size: int, patch_size: int) -> None:
    if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size <= 0:
        raise ValueError(f"SatMAE++ img_size must be a positive integer, got {img_size!r}.")
    if img_size % patch_size != 0:
        raise ValueError(
            f"SatMAE++ img_size must be a multiple of patch_size {patch_size}, got {img_size}."
        )


class _LayerNorm(nn.LayerNorm):
    """LayerNorm with ``eps=1e-6`` matching SatMAE++'s ViT blocks."""

    def __init__(self, dim: int, **kwargs: Any) -> None:
        super().__init__(dim, eps=1e-6, **kwargs)


def _build_model_normalizer(
    bands: list[BandSpec], modality: Literal["rgb", "s2"]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build SatMAE++ pretraining normalization for BGR RGB or 10-band Sentinel-2 inputs."""
    src_unit = detect_input_unit(bands)
    if modality == "rgb":
        mean = torch.tensor(SATMAEPP_RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(SATMAEPP_RGB_STD, dtype=torch.float32).view(1, 3, 1, 1)

        def normalize_rgb(images: torch.Tensor) -> torch.Tensor:
            x = to_rgb_display_scale(images, src_unit)
            return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

        return normalize_rgb

    if src_unit not in (InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1):
        raise UnsupportedNormalizationError(
            f"SatMAE++ Sentinel-2 model normalization expects Sentinel-2 DN or reflectance, "
            f"got {src_unit}. Use --normalization dataset."
        )
    mean_s2 = torch.tensor(SATMAEPP_S2_MEAN, dtype=torch.float32).view(1, 10, 1, 1)
    std_s2 = torch.tensor(SATMAEPP_S2_STD, dtype=torch.float32).view(1, 10, 1, 1)
    low = mean_s2 - 2.0 * std_s2
    span = 4.0 * std_s2

    def normalize_s2(images: torch.Tensor) -> torch.Tensor:
        x = to_s2_dn(images, src_unit)
        return ((x - low.to(x.device, x.dtype)) / span.to(x.device, x.dtype)).clamp(0.0, 1.0)

    return normalize_s2


class SatMAEPPBench(BenchModel):
    """Frozen SatMAE++ ViT-L encoder (fMoW-RGB and 10-band fMoW-Sentinel).

    * ``variant="rgb_vit_l"`` expects red, green, and blue bands and reorders them to
      OpenCV **BGR** (``blue``, ``green``, ``red``), matching the released fMoW-RGB
      pretraining checkpoint.
    * ``variant="s2_vit_l"`` selects the 10 Sentinel-2 bands ``B02..B08, B8A, B11, B12``
      (dropping B01, B09, B10) and encodes them through three spectral channel groups
      ``(B02, B03, B04, B08)``, ``(B05, B06, B07, B8A)``, and ``(B11, B12)``.

    For segmentation, ``taps.<i>`` exposes block ``i``'s patch tokens as ``(B, D, H, W)``
    (averaging across spectral groups for ``s2_vit_l``). For ``rgb_vit_l``, ``blocks.<i>``
    also exposes standard ``(B, 1 + H*W, D)`` ViT tokens.

    Args:
        bands: Ordered :class:`BandSpec` list describing the input channels.
        variant: ``"rgb_vit_l"`` (ViT-L/16, 224 px) or ``"s2_vit_l"`` (grouped ViT-L/8, 96 px).
        pretrained: Download and strictly load the pinned Apache-2.0 checkpoint.
        checkpoint_path: Optional local checkpoint path (still SHA-256 verified when
            ``pretrained=True``).
        pool: Token pooling mode: ``"mean"`` (patch mean, default), ``"cls"``, or ``"both"``.
        img_size: Optional square input size divisible by ``patch_size``. Defaults to the
            variant's pretraining size (224 for RGB, 96 for Sentinel-2). When different from
            the checkpoint grid, ``pos_embed`` is bicubically interpolated on load.
        allow_missing: Zero-fill missing target bands instead of raising ``BandCompatibilityError``.
        **kwargs: Forwarded to :class:`BenchModel` (e.g. ``normalization``).
    """

    handles_own_normalization = True
    num_prefix_tokens: int = 1
    feature_layout: str | None = "nchw"

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        variant: Literal["rgb_vit_l", "s2_vit_l"] = "rgb_vit_l",
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        pool: str = "mean",
        img_size: int | None = None,
        allow_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        if variant not in SATMAEPP_VARIANTS:
            raise ValueError(
                f"Unknown SatMAE++ variant {variant!r}; choose {sorted(SATMAEPP_VARIANTS)}."
            )
        if pool not in VALID_MODES:
            raise ValueError(f"SatMAE++ pool mode {pool!r} not in {VALID_MODES}.")
        spec = SATMAEPP_VARIANTS[variant]
        if img_size is not None:
            _validate_img_size(img_size, spec.patch_size)
        super().__init__(bands=bands, **kwargs)

        self.variant = variant
        self.spec = spec
        self.pool = pool
        self.img_size = img_size
        self.input_size = img_size or spec.default_img_size
        self.probe_input_size = self.input_size
        self.patch_size = spec.patch_size
        self.embed_dim = spec.embed_dim
        self.channel_groups = spec.channel_groups
        self.target_bands = list(
            SATMAEPP_RGB_BANDS if spec.modality == "rgb" else SATMAEPP_S2_BANDS
        )
        self._band_policy = BandMappingPolicy(allow_missing=allow_missing, band_fallbacks={})
        # Validate band compatibility eagerly so bad band sets fail before loading weights.
        dummy = torch.zeros(1, len(self.bands), 1, 1)
        map_to_model_bands(dummy, self.bands, self.target_bands, policy=self._band_policy)

        if self.normalization is NormalizationStrategy.MODEL_NATIVE:
            self._model_normalizer: Callable[[torch.Tensor], torch.Tensor] | None = (
                _build_model_normalizer(self.bands, spec.modality)
            )
            self._normalizer = lambda x: x
        else:
            self._model_normalizer = None
            self._normalizer = build_normalizer(self.normalization, bands=self.bands)

        self._build_encoder(spec)
        if pretrained:
            self.load_checkpoint(self._download_checkpoint(variant, checkpoint_path))

    def _build_encoder(self, spec: SatMAEPPVariant) -> None:
        grid = self.input_size // spec.patch_size
        self.patch_embed: PatchEmbed | nn.ModuleList
        if spec.channel_groups is None:
            self.patch_embed = PatchEmbed(self.input_size, spec.patch_size, 3, spec.embed_dim)
            self.channel_embed: nn.Parameter | None = None
        else:
            self.patch_embed = nn.ModuleList(
                PatchEmbed(self.input_size, spec.patch_size, len(group), spec.embed_dim)
                for group in spec.channel_groups
            )
            num_groups = len(spec.channel_groups)
            chan_init = get_1d_sincos_pos_embed_from_grid(
                spec.channel_embed, np.arange(num_groups, dtype=np.float32)
            )
            self.channel_embed = nn.Parameter(
                torch.from_numpy(chan_init).float().unsqueeze(0), requires_grad=False
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, spec.embed_dim))
        pos_dim = spec.embed_dim - spec.channel_embed
        pos_init = get_2d_sincos_pos_embed(pos_dim, grid, cls_token=True)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_init).float().unsqueeze(0), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            Block(
                spec.embed_dim, spec.num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=_LayerNorm
            )
            for _ in range(spec.depth)
        )
        self.norm = _LayerNorm(spec.embed_dim)
        self.taps = nn.ModuleList(nn.Identity() for _ in range(spec.depth))
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def _download_checkpoint(variant: str, checkpoint_path: str | None = None) -> Path:
        spec = SATMAEPP_VARIANTS[variant]
        digest = load_checksums("satmaepp.sha256")[spec.filename]
        return fetch_pinned_file(
            spec.repo, spec.filename, spec.revision, digest, local_path=checkpoint_path
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a SatMAE++ pretraining checkpoint strictly into the encoder.

        Accepts either a full pretraining checkpoint dict (``{"model": state_dict, ...}``)
        or a flat ``state_dict``. Interpolates ``pos_embed`` if the checkpoint grid differs
        from ``input_size``.

        Raises:
            ValueError: Unexpected, missing, or mis-shaped encoder weights.
        """
        with torch.serialization.safe_globals([argparse.Namespace]):
            raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported SatMAE++ checkpoint format in {path}.")
        self.load_encoder_state_dict(state, source=str(path))

    def load_encoder_state_dict(
        self, state: dict[str, torch.Tensor], *, source: str = "state_dict"
    ) -> None:
        """Validate, interpolate ``pos_embed`` if needed, and load encoder weights strictly."""
        state = dict(state)
        if "pos_embed" in state:
            state["pos_embed"] = interpolate_pos_embed(
                state["pos_embed"], self.input_size // self.patch_size
            )
        expected = self.state_dict()
        unexpected = sorted(
            k for k in state if k not in expected and not k.startswith(_IGNORED_CHECKPOINT_PREFIXES)
        )
        encoder = {k: v for k, v in state.items() if k in expected}
        missing = sorted(k for k in expected if k not in encoder)
        mismatched = sorted(k for k, v in encoder.items() if v.shape != expected[k].shape)
        if unexpected or missing or mismatched:
            raise ValueError(
                f"Incompatible SatMAE++ {self.variant} checkpoint {source}: "
                f"unexpected={unexpected[:5]}, missing={missing[:5]}, mismatched={mismatched[:5]}"
            )
        self.load_state_dict(encoder, strict=True)
        logger.info("Loaded SatMAE++ %s weights from %s", self.variant, source)

    def _prepare_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Reorder bands into model order (BGR for RGB, 10-band S2 for S2) and resize."""
        mapped, _ = map_to_model_bands(
            images, self.bands, self.target_bands, policy=self._band_policy
        )
        if self._model_normalizer is not None:
            mapped = self._model_normalizer(mapped)
        return _auto_resize(mapped, self.input_size)

    def _embed_tokens(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        if self.channel_groups is None:
            patch_embed = cast(PatchEmbed, self.patch_embed)
            tokens = patch_embed(images) + self.pos_embed[:, 1:, :]
            cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(batch, -1, -1)
            return torch.cat((cls_tokens, tokens), dim=1)

        group_embeds = cast(nn.ModuleList, self.patch_embed)
        channel_embed = cast(nn.Parameter, self.channel_embed)
        embedded = [
            group_embeds[i](images[:, list(group), :, :])
            for i, group in enumerate(self.channel_groups)
        ]
        x = torch.stack(embedded, dim=1)  # (B, G, L, D)
        num_groups, num_patches = x.shape[1], x.shape[2]
        pos = self.pos_embed[:, 1:, :].unsqueeze(1).expand(-1, num_groups, -1, -1)
        chan = channel_embed.unsqueeze(2).expand(-1, -1, num_patches, -1)
        x = (x + torch.cat((pos, chan), dim=-1)).reshape(batch, num_groups * num_patches, -1)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        return torch.cat((cls_tokens, x), dim=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor, grid: int) -> torch.Tensor:
        """Reshape patch tokens ``(B, 1 + G*L, D)`` into an NCHW feature map ``(B, D, H, W)``."""
        batch, _, dim = tokens.shape
        patches = tokens[:, 1:, :]
        if self.channel_groups is None:
            return patches.transpose(1, 2).reshape(batch, dim, grid, grid)
        num_groups = len(self.channel_groups)
        grouped = patches.reshape(batch, num_groups, grid, grid, dim)
        return grouped.mean(dim=1).permute(0, 3, 1, 2).contiguous()

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return normed ``(B, 1 + G*L, D)`` tokens and emit spatial maps through ``taps``."""
        x = self._prepare_inputs(images)
        x = self._embed_tokens(x)
        grid = self.input_size // self.patch_size
        for i, block in enumerate(self.blocks):
            x = block(x)
            self.taps[i](self._tokens_to_spatial(x, grid))
        return self.norm(x)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return tokens[:, 0, :]
        patch_mean = tokens[:, 1:, :].mean(dim=1)
        if self.pool == "mean":
            return patch_mean
        return torch.cat((tokens[:, 0, :], patch_mean), dim=-1)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._pool(self.forward_tokens(images))
