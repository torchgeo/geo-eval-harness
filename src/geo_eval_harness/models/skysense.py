"""SkySense (CVPR 2024) backbones for geo-eval-harness.

SkySense (https://github.com/Jack-bo1220/SkySense) releases three single-modality backbones: a
Swin Transformer V2-Huge for high-resolution RGB imagery and two ViT-Large/4 encoders for
Sentinel-2 and Sentinel-1. The official checkpoints are distributed through a Notion page for
non-commercial research use only, so this wrapper loads the safetensors conversion
``BiliSakura/SkySense-transformers`` and its ``trust_remote_code`` model definitions at a pinned
commit.

Layer naming for SegmentationProbe:
    ``hr``: ``stages.{0..3}.blocks.<last>`` emit ``(B, N, C)`` tokens at strides 4, 8, 16, 32.
    ``s2`` / ``s1``: ``layers.{0..23}`` emit ``(B, 1 + N, 1024)`` tokens at stride 4.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

from geo_eval_harness.bands import BandSpec

from ._band_mapping import resolve_src_indices
from ._input_units import InputUnit
from .interface import BenchModel

logger = logging.getLogger(__name__)

SKYSENSE_REPO = "BiliSakura/SkySense-transformers"
SKYSENSE_REVISION = "eb50219bbfcdf563f55a86ba9b0e2268dd151f04"

# Upstream iSAID config: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375] on 0..255.
_IMAGENET_STATS: dict[str, tuple[float, float]] = {
    "red": (0.485, 0.229),
    "green": (0.456, 0.224),
    "blue": (0.406, 0.225),
}


@dataclass(frozen=True)
class _Backbone:
    subfolder: str
    bands: tuple[str, ...]
    preferred_sensors: tuple[str, ...]
    embed_dim: int
    num_prefix_tokens: int


_BACKBONES: dict[str, _Backbone] = {
    "hr": _Backbone("skysense-swinv2-huge-rgb", ("red", "green", "blue"), (), 2816, 0),
    "s2": _Backbone(
        "skysense-vit-large-s2",
        (
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
        ),
        ("s2",),
        1024,
        1,
    ),
    "s1": _Backbone("skysense-vit-large-s1", ("vv", "vh"), ("sar", "s1"), 1024, 1),
}


class SkySenseBench(BenchModel):
    """Frozen SkySense backbone for one input modality.

    Embeddings mean-pool the final feature map. Only the ``hr`` backbone has published
    pretraining statistics (ImageNet on 0..255 RGB), so ``s2`` and ``s1`` support dataset
    statistics only.

    Args:
        bands: Ordered :class:`BandSpec` list describing the dataset channels.
        modality: ``"hr"`` for the RGB Swin V2-Huge, ``"s2"`` for the 10-band Sentinel-2
            ViT-L (B2-B8, B8A, B11, B12), or ``"s1"`` for the VV/VH Sentinel-1 ViT-L.
        repo: Hugging Face repository holding the converted checkpoints.
        revision: Full commit hash of ``repo`` for both weights and remote code.
        **kwargs: Forwarded to :class:`BenchModel`.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        modality: Literal["hr", "s2", "s1"] = "hr",
        repo: str = SKYSENSE_REPO,
        revision: str = SKYSENSE_REVISION,
        **kwargs: Any,
    ) -> None:
        if modality not in _BACKBONES:
            raise ValueError(f"Unknown SkySense modality {modality!r}; choose hr, s2, or s1.")
        spec = _BACKBONES[modality]
        src_index = resolve_src_indices(bands, preferred_sensors=spec.preferred_sensors)
        missing = [name for name in spec.bands if name not in src_index]
        if missing:
            raise ValueError(
                f"SkySense modality={modality!r} requires bands {list(spec.bands)}; "
                f"missing: {missing}. Available canonical bands: {sorted(src_index)}."
            )
        self.band_indices = [src_index[name] for name in spec.bands]
        if modality == "hr":
            # model_native normalizes dataset channels before band selection drops extras.
            mean = [0.0] * len(bands)
            std = [1.0] * len(bands)
            for name, idx in zip(spec.bands, self.band_indices, strict=True):
                mean[idx], std[idx] = _IMAGENET_STATS[name]
            self.expected_input_unit = InputUnit.UINT8
            self.pretrain_mean, self.pretrain_std = mean, std
        super().__init__(bands=bands, **kwargs)

        self.modality = modality
        self.embed_dim = spec.embed_dim
        self.num_prefix_tokens = spec.num_prefix_tokens
        logger.info("Loading SkySense %s from %s@%s", modality, repo, revision)
        # transformers resolves remote code at the repository root and ignores ``subfolder``.
        root = snapshot_download(repo, revision=revision, allow_patterns=[f"{spec.subfolder}/*"])
        self.backbone = AutoModel.from_pretrained(
            Path(root) / spec.subfolder, trust_remote_code=True, local_files_only=True
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, embed_dim)`` by mean-pooling the last feature map."""
        out = self.backbone(pixel_values=images[:, self.band_indices])
        return out.last_hidden_state.mean(dim=(-2, -1))
