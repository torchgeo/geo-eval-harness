"""SatMAE++ grouped-channel encoder for fMoW-Sentinel (Noman et al., CVPR 2024)."""

import torch
from timm.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from . import satmaepp
from ._input_units import InputUnit, to_s2_dn
from ._thor_encoder import sincos_1d
from .satmaepp import SatMAEPPBench
from .smarties import _sincos_pos_embed

# fMoW-Sentinel groups 10 m BGR + NIR, 20 m red-edge + narrow NIR, and 20 m SWIR.
_GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))


# Training DN statistics in the checkpoint's band order (techmn/satmae_pp, util/datasets.py).
_DN_STATS = {
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


class _GroupPatches(nn.ModuleList):
    """Apply one checkpoint patch projection per spectral group."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return group-major patch tokens (B, G, L, D)."""
        return torch.stack(
            [project(images[:, group]) for group, project in zip(_GROUPS, self, strict=True)], dim=1
        )


class _GroupedViT(VisionTransformer):
    """ViT-L/8 with separate spatial and spectral position embeddings."""

    def __init__(self, size: int, patch_size: int) -> None:
        super().__init__(img_size=size, patch_size=patch_size, num_classes=0, **satmaepp._VIT_L)
        self.patch_embed = _GroupPatches(
            PatchEmbed(size, patch_size, len(group), self.embed_dim) for group in _GROUPS
        )
        channel_dim = self.embed_dim // 4
        self.pos_embed = nn.Parameter(
            _sincos_pos_embed(self.embed_dim - channel_dim, size // patch_size), requires_grad=False
        )
        self.channel_embed = nn.Parameter(
            sincos_1d(channel_dim, torch.arange(len(_GROUPS))).unsqueeze(0), requires_grad=False
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate spectral and spatial positions before prepending the class token."""
        batch, groups, count, _ = x.shape
        assert self.pos_embed is not None
        spatial = self.pos_embed[:, None, 1:].expand(-1, groups, -1, -1)
        spectral = self.channel_embed[:, :, None].expand(-1, -1, count, -1)
        tokens = x + torch.cat((spatial, spectral), dim=-1)
        assert self.cls_token is not None
        return torch.cat((self.cls_token.expand(batch, -1, -1), tokens.flatten(1, 2)), dim=1)


class SatMAEPPSentinelBench(SatMAEPPBench):
    """Grouped ViT-L/8 for the ten fMoW-Sentinel bands."""

    dn_low: torch.Tensor
    dn_span: torch.Tensor
    band_order = tuple(_DN_STATS)
    patch_size = 8
    num_patch_groups = len(_GROUPS)
    default_image_size = 96
    checkpoint = "checkpoint_ViT-L_pretrain_fmow_sentinel"
    revision = "e6792ae99987b96569993306b396c232f27ba8cc"
    native_units = (InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1)

    def _build_encoder(self, size: int) -> VisionTransformer:
        """Build the checkpoint's grouped ViT at the resolved image size."""
        mean, std = torch.tensor(list(_DN_STATS.values())).T.reshape(2, 1, -1, 1, 1)
        self.register_buffer("dn_low", mean - 2 * std, persistent=False)
        self.register_buffer("dn_span", 4 * std, persistent=False)
        return _GroupedViT(size, self.patch_size)

    def _native_pixels(self, images: torch.Tensor, unit: InputUnit) -> torch.Tensor:
        """Reproduce fMoW-Sentinel's per-band two-sigma clipping and 8-bit quantization."""
        dn = to_s2_dn(images, unit)
        return ((dn - self.dn_low) / self.dn_span).clamp_(0, 1).mul_(255).floor_().div_(255)

    def tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Average corresponding spatial patches across the three spectral groups."""
        patches = tokens[:, 1:].reshape(
            tokens.shape[0], self.num_patch_groups, self.grid_size, self.grid_size, -1
        )
        return patches.mean(dim=1).permute(0, 3, 1, 2)
