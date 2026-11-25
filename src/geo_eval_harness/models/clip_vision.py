"""CLIP vision-tower wrapper for remote-sensing vision-language models.

Loads vision encoders from pinned Hugging Face checkpoints verified against
``checksums/clip_vision.sha256`` into ``timm`` CLIP backbones without
requiring ``open_clip``. Supports SkyCLIP (MIT), RemoteCLIP (Apache-2.0),
and GeoRSCLIP (MIT/CC).

All variants expect 3-channel RGB imagery normalized with standard OpenAI
CLIP statistics (``mean=(0.48145466, 0.4578275, 0.40821073)``,
``std=(0.26862954, 0.26130258, 0.27577711)``) under ``--normalization model``. Sentinel-2 DN
inputs are first stretched to display scale like ESA's TCI (DN 3558 maps to 1).
"""

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import timm
import torch
import torch.nn.functional as F

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.hub import fetch_pinned_file, load_checksums

from ._input_units import InputUnit, detect_input_unit, to_rgb_display_scale
from ._normalization import NormalizationStrategy
from ._pooling import VALID_MODES as _VIT_TOKEN_POOLS
from ._pooling import pool_tokens
from .interface import BenchModel

logger = logging.getLogger(__name__)

#: Standard OpenAI CLIP RGB normalization constants in ``[0, 1]`` reflectance space.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
_CLIP_MEAN_LIST = list(CLIP_MEAN)
_CLIP_STD_LIST = list(CLIP_STD)

_CHECKSUM_MANIFEST = "clip_vision.sha256"

_VIT_POOLS = ("proj", *_VIT_TOKEN_POOLS)
_RESNET_POOLS = ("proj", "avg")

_OPENCLIP_VIT_SWAPS: tuple[tuple[str, str], ...] = (
    ("conv1", "patch_embed.proj"),
    ("positional_embedding", "pos_embed"),
    ("transformer.resblocks.", "blocks."),
    ("ln_pre", "norm_pre"),
    ("ln_post", "norm"),
    ("ln_", "norm"),
    ("in_proj_", "qkv."),
    ("out_proj", "proj"),
    ("mlp.c_fc", "mlp.fc1"),
    ("mlp.c_proj", "mlp.fc2"),
)

_RESNET_CONV_RE = re.compile(r"^conv([0-9])")
_RESNET_BN_RE = re.compile(r"^bn([0-9])")
_RESNET_STAGE_RE = re.compile(r"^layer([0-9])\.([0-9]+)\.([a-z]+)([0-9])")
_RESNET_DOWN_RE = re.compile(r"^layer([0-9])\.([0-9]+)\.downsample\.([0-9])")
_RESNET_ID_MAP: dict[int, str] = {1: "conv1_1x1.", 2: "conv2_kxk.", 3: "conv3_1x1."}


@dataclass(frozen=True)
class CLIPVariantSpec:
    """Pinned Hugging Face source and ``timm`` architecture for one CLIP vision tower."""

    repo_id: str
    revision: str
    filename: str
    timm_name: str
    proj_dim: int
    checkpoint_format: Literal["timm", "openclip_vit", "openclip_resnet"]


#: Authoritative registry of supported remote-sensing CLIP vision-tower checkpoints.
CLIP_VARIANTS: dict[str, CLIPVariantSpec] = {
    "skyclip_vit_b32": CLIPVariantSpec(
        repo_id="torchgeo/vit_base_patch32_224_skyclip_50pct",
        revision="7f82439a2d0f79e1965ae71cdba8b911ed2f15cf",
        filename="vit_base_patch32_224_skyclip_50pct-5e3a3c68.pth",
        timm_name="vit_base_patch32_clip_224",
        proj_dim=512,
        checkpoint_format="timm",
    ),
    "skyclip_vit_l14": CLIPVariantSpec(
        repo_id="torchgeo/vit_large_patch14_224_skyclip_50pct",
        revision="aa6b40850d1dbd643b11716c2a4b7c392d4dec69",
        filename="vit_large_patch14_224_skyclip_50pct-fc4be2dd.pth",
        timm_name="vit_large_patch14_clip_quickgelu_224",
        proj_dim=768,
        checkpoint_format="timm",
    ),
    "remoteclip_rn50": CLIPVariantSpec(
        repo_id="chendelong/RemoteCLIP",
        revision="bf1d8a3ccf2ddbf7c875705e46373bfe542bce38",
        filename="RemoteCLIP-RN50.pt",
        timm_name="resnet50_clip",
        proj_dim=1024,
        checkpoint_format="openclip_resnet",
    ),
    "remoteclip_vit_b32": CLIPVariantSpec(
        repo_id="chendelong/RemoteCLIP",
        revision="bf1d8a3ccf2ddbf7c875705e46373bfe542bce38",
        filename="RemoteCLIP-ViT-B-32.pt",
        timm_name="vit_base_patch32_clip_224",
        proj_dim=512,
        checkpoint_format="openclip_vit",
    ),
    "remoteclip_vit_l14": CLIPVariantSpec(
        repo_id="chendelong/RemoteCLIP",
        revision="bf1d8a3ccf2ddbf7c875705e46373bfe542bce38",
        filename="RemoteCLIP-ViT-L-14.pt",
        timm_name="vit_large_patch14_clip_224",
        proj_dim=768,
        checkpoint_format="openclip_vit",
    ),
    "georsclip_vit_b32": CLIPVariantSpec(
        repo_id="Zilun/GeoRSCLIP",
        revision="4920188e6eba4e711ef9848cfd7cb77e874ee33f",
        filename="ckpt/RS5M_ViT-B-32.pt",
        timm_name="vit_base_patch32_clip_quickgelu_224",
        proj_dim=512,
        checkpoint_format="openclip_vit",
    ),
    "georsclip_vit_l14": CLIPVariantSpec(
        repo_id="Zilun/GeoRSCLIP",
        revision="4920188e6eba4e711ef9848cfd7cb77e874ee33f",
        filename="ckpt/RS5M_ViT-L-14.pt",
        timm_name="vit_large_patch14_clip_quickgelu_224",
        proj_dim=768,
        checkpoint_format="openclip_vit",
    ),
    "georsclip_vit_h14": CLIPVariantSpec(
        repo_id="Zilun/GeoRSCLIP",
        revision="4920188e6eba4e711ef9848cfd7cb77e874ee33f",
        filename="ckpt/RS5M_ViT-H-14.pt",
        timm_name="vit_huge_patch14_clip_224",
        proj_dim=1024,
        checkpoint_format="openclip_vit",
    ),
}


def _unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract a plain tensor mapping from a loaded checkpoint object."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint).__name__}.")
    raw = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected state_dict mapping, got {type(raw).__name__}.")
    return {
        key.removeprefix("module."): value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }


def convert_openclip_vit_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, prefix: str = "visual."
) -> dict[str, torch.Tensor]:
    """Map OpenCLIP/OpenAI CLIP ``visual.*`` ViT keys to a ``timm`` CLIP ViT state dict."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        name = key.removeprefix(prefix)
        for old, new in _OPENCLIP_VIT_SWAPS:
            name = name.replace(old, new)
        if name == "proj":
            out["head.weight"] = value.transpose(0, 1)
            out["head.bias"] = torch.zeros(value.shape[1], dtype=value.dtype)
        elif name == "class_embedding":
            out["cls_token"] = value.unsqueeze(0).unsqueeze(1)
        elif name == "pos_embed":
            out["pos_embed"] = value.unsqueeze(0)
        else:
            out[name] = value
    if not out:
        raise ValueError(f"No OpenCLIP ViT keys found with prefix {prefix!r}.")
    return out


def _resnet_stage_sub(match: re.Match[str]) -> str:
    stage_idx = int(match.group(1)) - 1
    layer_idx = int(match.group(2))
    layer_type = match.group(3)
    layer_id = int(match.group(4))
    return f"stages.{stage_idx}.{layer_idx}.{_RESNET_ID_MAP[layer_id]}{layer_type}"


def _resnet_down_sub(match: re.Match[str]) -> str:
    stage_idx = int(match.group(1)) - 1
    layer_idx = int(match.group(2))
    layer_id = int(match.group(3))
    suffix = "conv.conv" if layer_id == 0 else "conv.bn"
    return f"stages.{stage_idx}.{layer_idx}.shortcut.{suffix}"


def convert_openclip_resnet_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, prefix: str = "visual."
) -> dict[str, torch.Tensor]:
    """Map OpenCLIP/OpenAI CLIP ``visual.*`` ResNet keys to ``timm`` ``resnet50_clip``."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        name = key.removeprefix(prefix)
        name = _RESNET_CONV_RE.sub(r"stem.conv\1.conv", name)
        name = _RESNET_BN_RE.sub(r"stem.conv\1.bn", name)
        name = _RESNET_STAGE_RE.sub(_resnet_stage_sub, name)
        name = _RESNET_DOWN_RE.sub(_resnet_down_sub, name)
        if name.startswith("attnpool"):
            name = (
                name.replace("attnpool", "head", 1)
                .replace("positional_embedding", "pos_embed")
                .replace("q_proj", "q")
                .replace("k_proj", "k")
                .replace("v_proj", "v")
                .replace("c_proj", "proj")
            )
        out[name] = value
    if not out:
        raise ValueError(f"No OpenCLIP ResNet keys found with prefix {prefix!r}.")
    return out


def _convert_checkpoint(checkpoint: object, spec: CLIPVariantSpec) -> dict[str, torch.Tensor]:
    """Convert a loaded checkpoint into the ``timm`` state dict expected by ``spec``."""
    state_dict = _unwrap_state_dict(checkpoint)
    if spec.checkpoint_format == "timm":
        return state_dict
    if spec.checkpoint_format == "openclip_vit":
        return convert_openclip_vit_state_dict(state_dict)
    return convert_openclip_resnet_state_dict(state_dict)


def _build_model_normalizer(bands: list[BandSpec]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the CLIP RGB normalizer, stretching Sentinel-2 DN to display scale first."""
    src = detect_input_unit(bands)
    mean = torch.tensor(_CLIP_MEAN_LIST, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD_LIST, dtype=torch.float32).view(1, 3, 1, 1)

    def normalize(images: torch.Tensor) -> torch.Tensor:
        x = to_rgb_display_scale(images, src)
        return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

    return normalize


class CLIPVisionBench(BenchModel):
    """Frozen remote-sensing CLIP vision tower backed by ``timm``.

    Supports SkyCLIP (ViT-B/32, ViT-L/14), RemoteCLIP (RN50, ViT-B/32,
    ViT-L/14), and GeoRSCLIP (ViT-B/32, ViT-L/14, ViT-H/14) without
    requiring ``open_clip``.

    Args:
        bands: Ordered :class:`BandSpec` list. Must have exactly 3 RGB bands.
        variant: Key in ``CLIP_VARIANTS`` (e.g. ``"skyclip_vit_b32"``,
            ``"remoteclip_vit_l14"``, ``"georsclip_vit_h14"``).
        pretrained: Download and load the pinned SHA-256-verified checkpoint.
        pool: Embedding readout mode. ``"proj"`` (default) returns the CLIP
            image projection vector; ViT variants also accept ``"cls"``,
            ``"mean"``, and ``"both"`` on pre-projection tokens; ResNet
            variants also accept ``"avg"`` on the final stage feature map.
        normalize: If ``True``, L2-normalize the returned embedding.
        dynamic_img_size: Allow ViT backbones to interpolate positional
            embeddings for non-224 square or rectangular inputs.
        checkpoint_path: Optional local checkpoint file verified against the
            variant's packaged SHA-256 digest instead of downloading.
        normalization: Input normalization strategy (defaults to
            ``model_native``, applying standard CLIP RGB statistics).
    """

    expected_input_unit = InputUnit.REFLECTANCE_0_1
    pretrain_mean = _CLIP_MEAN_LIST
    pretrain_std = _CLIP_STD_LIST

    def __init__(  # noqa: PLR0913 - explicit YAML constructor options
        self,
        bands: list[BandSpec],
        *,
        variant: str,
        pretrained: bool = True,
        pool: str = "proj",
        normalize: bool = False,
        dynamic_img_size: bool = False,
        checkpoint_path: str | None = None,
        normalization: NormalizationStrategy | str = NormalizationStrategy.MODEL_NATIVE,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization=normalization, **_kwargs)
        if self.normalization is NormalizationStrategy.MODEL_NATIVE:
            self._normalizer = _build_model_normalizer(bands)
        if self.num_channels != 3:
            raise ValueError(
                f"CLIPVisionBench only supports 3-channel RGB input, got {self.num_channels}. "
                "Run with --bands red,green,blue."
            )
        if variant not in CLIP_VARIANTS:
            raise ValueError(
                f"Unknown CLIP variant {variant!r}; expected one of {sorted(CLIP_VARIANTS)}."
            )
        spec = CLIP_VARIANTS[variant]
        self._is_resnet = spec.checkpoint_format == "openclip_resnet"
        valid_pools = _RESNET_POOLS if self._is_resnet else _VIT_POOLS
        if pool not in valid_pools:
            raise ValueError(
                f"Unsupported pool={pool!r} for {variant!r}; expected one of {valid_pools}."
            )

        self.variant = variant
        self.spec = spec
        self.pretrained = pretrained
        self.pool = pool
        self.normalize = normalize
        self.dynamic_img_size = dynamic_img_size

        extra_kwargs: dict[str, Any] = (
            {"dynamic_img_size": True} if not self._is_resnet and dynamic_img_size else {}
        )
        self.backbone = timm.create_model(
            spec.timm_name, pretrained=False, num_classes=spec.proj_dim, **extra_kwargs
        )

        if pretrained:
            digest = load_checksums(_CHECKSUM_MANIFEST)[spec.filename]
            path = fetch_pinned_file(
                spec.repo_id,
                spec.filename,
                spec.revision,
                digest,
                local_path=checkpoint_path,
            )
            logger.info("Loading %s weights from %s", variant, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = _convert_checkpoint(checkpoint, spec)
            self.backbone.load_state_dict(state_dict, strict=True)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return image embeddings of shape ``(B, D)`` from normalized RGB inputs."""
        backbone: Any = self.backbone
        if self.pool == "proj":
            features = cast(torch.Tensor, backbone(images))
        elif self._is_resnet:
            feature_map = cast(torch.Tensor, backbone.forward_features(images))
            features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        else:
            tokens = cast(torch.Tensor, backbone.forward_features(images))
            features = pool_tokens(tokens, mode=self.pool)

        if self.normalize:
            features = F.normalize(features, p=2, dim=-1)
        return features
