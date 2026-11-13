"""Inference-only THOR ViT encoder (Forgaard et al., arXiv:2601.16011).

Reimplements the encoder of https://github.com/FM4CS/THOR at commit 7ded3ce
(``thor/models/thor_vit.py`` and ``thor/utils/patch_embed.py``, MIT) for its evaluation path:
a fixed patch size, no class token, group-mean aggregation, a patch-size embedding, and GSD-aware
2D ALiBi. Parameter names follow the released checkpoints, so they load strictly. Upstream patches
``timm``'s ``Attention``/``Block`` globally to take the ALiBi bias; these blocks take it directly.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

#: Band groups in THOR's token order (``_default_input_params["groups"]`` upstream). Bands of a
#: group share a GSD and a patch size, and their patch embeddings are averaged into one token grid.
THOR_GROUPS: tuple[tuple[str, ...], ...] = (
    ("S2:Red", "S2:Green", "S2:Blue", "S2:NIR"),
    ("S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"),
    ("S2:CoastAerosal", "S2:WaterVapor"),
    ("S1:IW-VH", "S1:IW-VV", "S1:EW-VH", "S1:EW-VV"),
    ("S1:IW-HV", "S1:IW-HH", "S1:EW-HV", "S1:EW-HH"),
    tuple(f"S3:Oa{i:02d}_reflectance" for i in range(1, 8)),
    tuple(f"S3:Oa{i:02d}_reflectance" for i in range(8, 15)),
    tuple(f"S3:Oa{i:02d}_reflectance" for i in range(15, 22)),
    tuple(f"S3:S{i}_reflectance_an" for i in range(1, 7)),
    tuple(f"S3:S{i}_BT_in" for i in range(7, 10)),
)
#: Pretraining GSD (m) of each group; Sentinel-3 is resampled to multiples of 6 upstream.
THOR_GROUP_GSD: tuple[int, ...] = (10, 20, 60, 10, 10, 240, 240, 240, 480, 960)
#: Size of the stored per-band patch-embedding kernels; other patch sizes are PI-resized.
STORED_PATCH_SIZE = 16
#: Patch-size embeddings are relative to the smallest patch (4 px) at the finest GSD (10 m).
_REF_PATCH_SIZE = 4
_MIN_GSD = 10


def patch_embed_name(channel: str) -> str:
    """Return the patch-embedding key of a THOR channel (IW and EW SAR share one kernel)."""
    sensor, band = channel.split(":")
    if sensor == "S1":
        return f"S1:{band.split('-')[1]}"
    return channel


def alibi_slopes(num_heads: int) -> list[float]:
    """ALiBi head slopes (Press et al., 2022), as ``get_slopes`` upstream."""

    def power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return power_of_2(num_heads)
    closest = 2 ** math.floor(math.log2(num_heads))
    return power_of_2(closest) + alibi_slopes(2 * closest)[0::2][: num_heads - closest]


def sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """MAE 1D sin-cos embedding of ``positions`` (M,) into float32 ``(M, dim)``."""
    omega = torch.arange(dim // 2, dtype=torch.float32, device=positions.device) / (dim / 2.0)
    out = positions.float().reshape(-1, 1) * (1.0 / 10000**omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


@cache
def _pinv_resize_matrix(old: int, new: int) -> torch.Tensor:
    """Pseudo-inverse ``(new², old²)`` of bicubic kernel resizing (FlexiViT, Beyer et al., 2023)."""
    basis = torch.eye(old * old).reshape(-1, 1, old, old)
    resized = F.interpolate(basis, (new, new), mode="bicubic", antialias=True)
    return torch.linalg.pinv(resized.reshape(old * old, new * new))


def resize_patch_kernel(weight: torch.Tensor, new: int) -> torch.Tensor:
    """PI-resize a ``(D, 1, p, p)`` patch-embedding kernel to ``(D, 1, new, new)``."""
    old = weight.shape[-1]
    if old == new:
        return weight
    pinv = _pinv_resize_matrix(old, new).to(weight.device)
    flat = weight.reshape(weight.shape[0], -1).float() @ pinv.T
    return flat.reshape(weight.shape[0], 1, new, new).to(weight.dtype)


@dataclass(frozen=True)
class GroupInput:
    """One THOR band group: its index in :data:`THOR_GROUPS` and ``{channel: (B, 1, S, S)}``."""

    index: int
    bands: dict[str, torch.Tensor]


class _Attention(nn.Module):
    """timm-layout multi-head self-attention taking an additive bias."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Attend over ``(B, N, D)`` tokens with a ``(1, H, N, N)`` bias."""
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transform."""
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    """Pre-norm transformer block with the parameter names of ``timm``'s ``Block``."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, 4 * dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Run one residual attention and MLP stage."""
        x = x + self.attn(self.norm1(x), bias)
        return x + self.mlp(self.norm2(x))


class _BandPatchEmbed(nn.Module):
    """One single-channel patch-embedding kernel per band (``IndFlexiPatchEmbed`` upstream)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        names = dict.fromkeys(patch_embed_name(c) for group in THOR_GROUPS for c in group)
        self.patch_embed = nn.ModuleDict(
            {
                name: nn.Conv2d(1, embed_dim, STORED_PATCH_SIZE, stride=STORED_PATCH_SIZE)
                for name in names
            }
        )

    def forward(self, channel: str, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Embed ``(B, 1, S, S)`` into ``(B, (S / patch_size)², D)``."""
        conv = cast(nn.Conv2d, self.patch_embed[patch_embed_name(channel)])
        weight = resize_patch_kernel(conv.weight, patch_size)
        return F.conv2d(x, weight, conv.bias, stride=patch_size).flatten(2).transpose(1, 2)


class ThorEncoder(nn.Module):
    """THOR ViT encoder for a fixed patch size.

    Args:
        embed_dim: Token width.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        band_embed_dim: Width of the group embedding; the remaining ``embed_dim - band_embed_dim``
            channels carry the patch-size embedding.
    """

    encoder_slopes: torch.Tensor

    def __init__(self, embed_dim: int, depth: int, num_heads: int, band_embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.band_embed_dim = band_embed_dim
        self.ind_patch_embed = _BandPatchEmbed(embed_dim)
        groups = torch.arange(len(THOR_GROUPS))
        self.band_embed = nn.ParameterDict(
            {
                f"group{i}": nn.Parameter(row.float(), requires_grad=False)
                for i, row in enumerate(sincos_1d(band_embed_dim, groups))
            }
        )
        self.register_buffer("encoder_slopes", torch.tensor(alibi_slopes(num_heads)))
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _group_tokens(self, group: GroupInput, patch_size: int) -> torch.Tensor:
        """Average the group's band embeddings and add its patch-size and group embeddings."""
        embeds = [
            self.ind_patch_embed(channel, x, patch_size) for channel, x in group.bands.items()
        ]
        tokens = torch.stack(embeds, dim=-1).mean(dim=-1)
        relative_size = torch.tensor(
            [patch_size * THOR_GROUP_GSD[group.index] / (_REF_PATCH_SIZE * _MIN_GSD)],
            device=tokens.device,
        )
        size_embed = sincos_1d(self.embed_dim - self.band_embed_dim, relative_size)
        group_embed = self.band_embed[f"group{group.index}"][None]
        return tokens + torch.cat([size_embed, group_embed], dim=-1).to(tokens.dtype)

    def _alibi(self, sides: list[int], gsds: list[int], patch_size: int) -> torch.Tensor:
        """GSD-aware 2D ALiBi bias ``(1, H, N, N)`` over the concatenated token grids."""
        points = []
        for side, gsd in zip(sides, gsds, strict=True):
            line = (torch.arange(side, dtype=torch.float32) * patch_size + patch_size / 2) * gsd
            points.append(torch.cartesian_prod(line, line))
        coords = torch.cat(points).to(self.encoder_slopes.device)
        # As in upstream evaluation, distances are in units of the largest patch footprint.
        coords = coords / max(patch_size * gsd for gsd in gsds)
        distances = torch.cdist(coords, coords)
        return (-self.encoder_slopes[:, None, None] * distances)[None]

    def forward(
        self,
        groups: list[GroupInput],
        patch_size: int,
        on_block: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Encode band groups (in :data:`THOR_GROUPS` order) into normed ``(B, N, D)`` tokens.

        ``on_block(i, x)`` receives each block's un-normed output.
        """
        tokens = [self._group_tokens(group, patch_size) for group in groups]
        sides = [math.isqrt(t.shape[1]) for t in tokens]
        bias = self._alibi(sides, [THOR_GROUP_GSD[g.index] for g in groups], patch_size)
        x = torch.cat(tokens, dim=1)
        for i, block in enumerate(self.blocks):
            x = block(x, bias)
            if on_block is not None:
                on_block(i, x)
        return self.norm(x)
