"""Inference-only RAMEN encoder (Houdré et al., arXiv:2512.05025).

Reimplements ``RAMEN_Encoder_MonoTemporal`` from https://github.com/nicolashoudre/RAMEN at commit
ca52286 (``pangaea-bench/pangaea/encoders/ramen_encoder.py``, MIT). Upstream ships the encoder
inside a fork of PANGAEA and a Lightning-Hydra pretraining project, neither of which is
installable as a library. Parameter names follow the released checkpoint, so it loads strictly
once the unused temporal (``ltae``) and elevation (``dem_projector``) modules are dropped.

RAMEN has no patch embedding. Each pixel's channel vector is projected to the token width by a
``(C, D)`` matrix generated from the channels' wavelengths (optical) or polarisation and orbit
(SAR). The resulting per-pixel feature map is resampled from the input GSD to a target GSD, and
each pixel of the resampled map becomes one token.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import nn
from torchgeo.models.scale_mae import (
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed_with_resolution,
)

#: SAR channel keys of :class:`RadarProjector`, as ``<orbit>_<polarisation>``.
RADAR_CHANNELS: tuple[str, ...] = tuple(
    f"{orbit}_{pol}" for orbit in ("asc", "des") for pol in ("vv", "vh", "hv", "hh")
)
#: Checkpoint modules the mono-temporal optical/SAR encoder does not use.
UNUSED_PREFIXES: tuple[str, ...] = ("ltae.", "dem_projector.")


@dataclass(frozen=True)
class ModalityInput:
    """One RAMEN modality: ``(B, C, H, W)`` pixels and one channel key per channel.

    Optical channels are keyed by centre wavelength in nm, SAR channels by a
    :data:`RADAR_CHANNELS` key.
    """

    pixels: torch.Tensor
    channels: tuple[float, ...] | tuple[str, ...]


def _projection_mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.LayerNorm(dim)
    )


class SpectralProjector(nn.Module):
    """Channel embeddings from centre wavelengths in nm (unnormalized sin-cos encoding)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = _projection_mlp(dim)

    def forward(self, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        """Map ``(C,)`` wavelengths to ``(C, D)`` channel embeddings."""
        return self.mlp(get_1d_sincos_pos_embed_from_grid_torch(self.dim, wavelengths_nm))


class RadarProjector(nn.Module):
    """Channel embeddings from one learned vector per orbit direction and polarisation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        for key in RADAR_CHANNELS:
            self.register_parameter(f"{key}_param", nn.Parameter(torch.zeros(dim)))
        self.mlp = _projection_mlp(dim)

    def forward(self, channels: tuple[str, ...]) -> torch.Tensor:
        """Map ``C`` :data:`RADAR_CHANNELS` keys to ``(C, D)`` channel embeddings."""
        return self.mlp(torch.stack([getattr(self, f"{key}_param") for key in channels]))


class ScaleResampler(nn.Module):
    """Bilinear resampling refined by a mixture of 1x1 convolutions conditioned on the scale."""

    def __init__(self, dim: int, num_experts: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.experts = nn.ModuleList(nn.Conv2d(dim, dim, kernel_size=1) for _ in range(num_experts))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, num_experts)
        )
        self.temperature = nn.Parameter(torch.tensor(0.7))

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Resample ``(B, D, H, W)`` by ``scale`` (input GSD / target GSD)."""
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        log_scale = torch.tensor([scale], device=x.device, dtype=x.dtype).log()
        scale_embed = get_1d_sincos_pos_embed_from_grid_torch(self.dim, log_scale)
        weights = (self.mlp(scale_embed) / self.temperature.clamp(min=1e-3)).softmax(dim=-1)[0]
        out = torch.zeros_like(x)
        for weight, expert in zip(weights, self.experts, strict=True):
            out = out + weight * expert(x)
        return x + out


class RamenEncoder(nn.Module):
    """RAMEN ViT encoder for single-date optical and SAR inputs.

    Args:
        embed_dim: Token width.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
    """

    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spectral_projector = SpectralProjector(embed_dim)
        self.radar_projector = RadarProjector(embed_dim)
        self.resampler = ScaleResampler(embed_dim)
        self.in_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def channel_embeddings(self, channels: tuple[float, ...] | tuple[str, ...]) -> torch.Tensor:
        """Return the ``(C, D)`` projection matrix of one modality's channels."""
        if all(isinstance(c, str) for c in channels):
            return self.radar_projector(channels)
        wavelengths = torch.tensor(channels, dtype=torch.float32, device=self.cls_token.device)
        return self.spectral_projector(wavelengths)

    def _modality_tokens(self, modality: ModalityInput, scale: float, side: int) -> torch.Tensor:
        """Project pixels, resample to a ``side`` x ``side`` grid, and flatten to tokens."""
        weight = self.channel_embeddings(modality.channels).to(modality.pixels.dtype)
        x = torch.einsum("bchw,cd->bdhw", modality.pixels, weight)
        x = self.resampler(x, scale)
        if x.shape[-2:] != (side, side):
            x = F.interpolate(x, size=(side, side), mode="bilinear")
        return self.in_norm(x.flatten(2).transpose(1, 2))

    def forward(
        self,
        modalities: list[ModalityInput],
        input_gsd: float,
        target_gsd: float,
        on_block: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Encode square modalities into ``(B, 1 + M * side², D)`` tokens, class token first.

        ``side`` is ``int(H * input_gsd / target_gsd)``, as upstream. ``on_block(i, x)`` receives
        each block's output; only the last block's output is normed, as in upstream's
        ``output_layers``.
        """
        scale = input_gsd / target_gsd
        side = int(modalities[0].pixels.shape[-1] * scale)
        tokens = torch.cat([self._modality_tokens(m, scale, side) for m in modalities], dim=1)
        res = torch.tensor([target_gsd], dtype=torch.float32, device=tokens.device)
        pos = get_2d_sincos_pos_embed_with_resolution(self.embed_dim, side, res, cls_token=True)
        pos = pos.to(tokens.dtype)
        tokens = tokens + pos[:, 1:].repeat(1, len(modalities), 1)
        cls = (self.cls_token + pos[:, :1]).expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        last = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == last:
                x = self.norm(x)
            if on_block is not None:
                on_block(i, x)
        return x
