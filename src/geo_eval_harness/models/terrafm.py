"""Checkpoint-compatible TerraFM encoder (arXiv:2506.06281).

Architecture reference: https://github.com/mbzuai-oryx/TerraFM.
Load local weights from https://huggingface.co/MBZUAI/TerraFM.
Parameter names follow the released checkpoint for strict loading.
"""

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn

from geo_eval_harness.bands import BandSpec

from ._band_mapping import BandMappingPolicy, map_to_model_bands, resolve_src_indices
from ._pooling import pool_tokens
from .interface import BenchModel
from .torchgeo_models import _auto_resize

logger = logging.getLogger(__name__)

# Sentinel-2 L2A band order is undocumented upstream.  TerraFM pretrains on
# Major-TOM Core-S2L2A, whose 12 bands are the standard S2 sequence with the
# cirrus band B10 dropped; this matches the order CROMA uses in this repo.
TERRAFM_S2_12: tuple[str, ...] = (
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "watervapor",
    "swir1",
    "swir2",
)
TERRAFM_S1_2: tuple[str, ...] = ("vv", "vh")
TERRAFM_INPUT_SIZE = 224
TERRAFM_PATCH_SIZE = 16


class _TerraFMAttention(nn.Module):
    """Multi-head self-attention with the released ``qkv``/``proj`` layout."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over ``(B, N, D)`` tokens."""
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, dim))


class _TerraFMMlp(nn.Module):
    """Two-layer GELU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transform to ``(B, N, D)`` tokens."""
        return self.fc2(self.act(self.fc1(x)))


class _TerraFMBlock(nn.Module):
    """Pre-norm transformer block (``norm1`` -> attention -> ``norm2`` -> MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _TerraFMAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _TerraFMMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one residual attention/MLP stage over ``(B, N, D)`` tokens."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _TerraFMTokenProjection(nn.Module):
    """Project wide per-modality tokens down to the transformer width.

    Upstream keeps this as the single-modality ("Case 1") path of a
    cross-attention fusion whose learnable query bank is pretraining-only and
    absent from the released checkpoint.
    """

    def __init__(self, embed_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.proj1 = nn.Linear(attn_dim, attn_dim, bias=False)
        self.norm_input = nn.LayerNorm(attn_dim)
        self.proj2 = nn.Linear(attn_dim, attn_dim)
        self.proj3 = nn.Linear(attn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, N, 3D)`` tokens to ``(B, N, D)``."""
        return self.proj3(self.proj2(self.norm_input(self.proj1(x))))


class _TerraFMPatchEmbed(nn.Module):
    """Per-modality patch stem with one Conv2d per sensor and product level.

    The ``3 * embed_dim`` stem width follows Panopticon, as credited upstream.
    """

    def __init__(
        self,
        img_size: int,
        embed_dim: int,
        patch_size: int,
        in_chans_s1: int,
        in_chans_s2: int,
    ) -> None:
        super().__init__()
        attn_dim = embed_dim * 3
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.attn_dim = attn_dim
        self.conv2d_s2_l2a = nn.Conv2d(in_chans_s2, attn_dim, patch_size, stride=patch_size)
        self.conv2d_s2_l1c = nn.Conv2d(in_chans_s2, attn_dim, patch_size, stride=patch_size)
        self.conv2d_s1 = nn.Conv2d(in_chans_s1, attn_dim, patch_size, stride=patch_size)
        self.projection = _TerraFMTokenProjection(embed_dim=embed_dim, attn_dim=attn_dim)
        self.s2_l2a_embed = nn.Parameter(torch.zeros(1, attn_dim))
        self.s2_l1c_embed = nn.Parameter(torch.zeros(1, attn_dim))
        self.s1_embed = nn.Parameter(torch.zeros(1, attn_dim))

    def forward(self, images: torch.Tensor, *, is_l2a: bool = False) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, N, D)``, routing on channel count."""
        if images.shape[1] == 2:
            x = self.conv2d_s1(images).flatten(2).transpose(1, 2) + self.s1_embed
        elif is_l2a:
            x = self.conv2d_s2_l2a(images).flatten(2).transpose(1, 2) + self.s2_l2a_embed
        else:
            x = self.conv2d_s2_l1c(images).flatten(2).transpose(1, 2) + self.s2_l1c_embed
        return self.projection(x)


class TerraFMBench(BenchModel):
    """Frozen TerraFM ViT encoder loading official weights from a local file.

    TerraFM is a single-modality-at-a-time encoder downstream: the paper's
    cross-sensor fusion is a pretraining construct with no released weights, so
    each config feeds one band stack (12-band Sentinel-2, or Sentinel-1 VV/VH).

    Normalization deviates from the published model in one respect: no
    pretraining mean/std statistics are published anywhere upstream, so this
    wrapper uses the framework default ``bandspec_zscore`` rather than
    declaring ``model_native``.

    Args:
        bands: Ordered :class:`BandSpec` list describing the dataset channels.
        variant: ``"base"`` or ``"large"``.  Only base weights are published.
        modality: ``"s2"`` for the 12-band optical stem, ``"s1"`` for VV/VH.
        is_l2a: Select the L2A rather than the L1C Sentinel-2 stem.  Ignored
            for ``modality="s1"``.
        checkpoint_path: Local path to ``TerraFM-B.pth``.
        checkpoint_md5: Optional MD5 to verify the checkpoint against.
        pretrained: Load ``checkpoint_path``; ``True`` requires it to be set.
        pool: Token pooling mode passed to :func:`pool_tokens`.
        auto_resize: Bilinearly resize inputs to ``target_size`` when needed.
        target_size: Spatial size fed to the backbone.
    """

    #: Width, depth, heads.  ``large`` is advertised upstream but unpublished.
    _VARIANTS: ClassVar[dict[str, tuple[int, int, int]]] = {
        "base": (768, 12, 12),
        "large": (1024, 24, 16),
    }
    num_prefix_tokens: int = 1

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        variant: Literal["base", "large"] = "base",
        modality: Literal["s2", "s1"] = "s2",
        is_l2a: bool = True,
        checkpoint_path: str | Path | None = None,
        checkpoint_md5: str | None = None,
        pretrained: bool = True,
        pool: str = "cls",
        auto_resize: bool = True,
        target_size: int = TERRAFM_INPUT_SIZE,
        **kwargs: Any,
    ) -> None:
        if variant not in self._VARIANTS:
            raise ValueError(f"Unknown TerraFM variant {variant!r}; choose base or large.")
        if modality not in ("s2", "s1"):
            raise ValueError(f"Unknown TerraFM modality {modality!r}; choose s2 or s1.")
        super().__init__(bands=bands, **kwargs)
        self.variant = variant
        self.modality = modality
        self.is_l2a = is_l2a
        self.pool = pool
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.model_bands = list(TERRAFM_S2_12 if modality == "s2" else TERRAFM_S1_2)
        if modality == "s1":
            self.s1_indices = self._resolve_s1_indices(bands)

        width, depth, heads = self._VARIANTS[variant]
        self.embed_dim = width
        self.patch_embed = _TerraFMPatchEmbed(
            img_size=target_size,
            embed_dim=width,
            patch_size=TERRAFM_PATCH_SIZE,
            in_chans_s1=len(TERRAFM_S1_2),
            in_chans_s2=len(TERRAFM_S2_12),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, width))
        self.blocks = nn.ModuleList([_TerraFMBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self._initialize_weights()
        if pretrained:
            if checkpoint_path is None:
                raise ValueError(
                    "TerraFM pretrained=True requires an explicit local checkpoint_path."
                )
            self.load_checkpoint(checkpoint_path, checkpoint_md5)

    @staticmethod
    def _resolve_s1_indices(bands: list[BandSpec]) -> list[int]:
        src_index = resolve_src_indices(bands, preferred_sensors=("sar", "s1"))
        missing = [name for name in TERRAFM_S1_2 if name not in src_index]
        if missing:
            available = sorted(src_index)
            raise ValueError(
                f"TerraFM modality='s1' requires bands {list(TERRAFM_S1_2)}; missing: {missing}. "
                f"Available canonical bands: {available}."
            )
        return [src_index[name] for name in TERRAFM_S1_2]

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    @staticmethod
    def _verify_md5(path: Path, expected: str | None) -> None:
        if expected is None:
            return
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected.lower():
            raise ValueError(f"TerraFM checkpoint MD5 mismatch for {path}.")

    def load_checkpoint(
        self, checkpoint_path: str | Path, checkpoint_md5: str | None = None
    ) -> None:
        """Load released TerraFM weights strictly, rejecting any incompatibility.

        The released checkpoint is a flat unprefixed state dict with no
        classifier head, so it must load under ``strict=True``; upstream's
        published ``strict=False`` would silently leave a mismatched variant
        randomly initialized.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"TerraFM checkpoint does not exist: {path}")
        self._verify_md5(path, checkpoint_md5)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError(
                "TerraFM checkpoint must be a state dict or contain a 'model' state dict."
            )
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        expected = self.state_dict()
        # The released weights carry num_classes=0, so `head` is nn.Identity and
        # contributes no entries; anything else unexpected is a real mismatch.
        unexpected = sorted(key for key in state if key not in expected)
        encoder = {key: value for key, value in state.items() if key in expected}
        missing = sorted(key for key in expected if key not in encoder)
        mismatched = sorted(
            key for key, value in encoder.items() if value.shape != expected[key].shape
        )
        if unexpected or missing or mismatched:
            raise ValueError(
                f"Incompatible TerraFM checkpoint: unexpected={unexpected[:5]}, "
                f"missing={missing[:5]}, mismatched={mismatched[:5]}"
            )
        self.load_state_dict(encoder, strict=True)
        logger.info("Loaded TerraFM %s weights from %s", self.variant, path)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Bicubically resize the learned position embedding to the token grid."""
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # The 0.1 offset guards the bicubic scale factor against floating point
        # error; see https://github.com/facebookresearch/dino/issues/8.
        w0 = w // self.patch_embed.patch_size + 0.1
        h0 = h // self.patch_embed.patch_size + 0.1
        side = int(math.sqrt(n))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / side, h0 / side),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize:
            images = _auto_resize(images, self.target_size)
        if self.modality == "s1":
            return images[:, self.s1_indices]
        mapped, _ = map_to_model_bands(
            images,
            self.bands,
            self.model_bands,
            policy=BandMappingPolicy(preferred_sensors=("s2",)),
        )
        return mapped

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return the normed ``(B, 1 + N, D)`` token sequence for prepared inputs."""
        x = self._prepare_input(images)
        _, _, w, h = x.shape
        # Pass is_l2a explicitly: upstream's prepare_tokens drops the flag, so
        # its forward() can only ever reach the L1C stem.
        x = self.patch_embed(x, is_l2a=self.is_l2a)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return pool_tokens(self.forward_tokens(images), mode=self.pool)
