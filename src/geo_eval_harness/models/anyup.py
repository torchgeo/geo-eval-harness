"""Frozen pretrained AnyUp feature upsampling before a segmentation classifier.

AnyUp (Wimmer et al.; https://github.com/wimmerth/anyup, CC BY 4.0) upsamples a feature map to the
resolution of a guidance image with attention conditioned on that image. Here it sits between the
UPerNet fused features and the 1x1 classifier, so class logits are predicted at input resolution
from image-guided features instead of bilinearly upsampled ones. The AnyUp weights stay frozen.

The guidance image is built from the input's red, green, and blue bands, found by wavelength
(or by name); inputs without them repeat the first band. Each image gets a 2-98 percentile stretch
and ImageNet normalisation, so the raw input scale does not matter.
"""

import hashlib
import logging
from pathlib import Path
from typing import override

import torch
import torch.nn as nn

from geo_eval_harness.datasets.base import BandSpec

logger = logging.getLogger(__name__)

ANYUP_REPO = "wimmerth/anyup"
ANYUP_REF = "351807a9c4287368732cc247f26c7c81c9139af4"
ANYUP_ENTRYPOINT = "anyup_multi_backbone"
#: sha256 of the multi-backbone release checkpoint (``checkpoint_v2/anyup_multi_backbone.pth``).
ANYUP_CHECKPOINT_SHA256 = "b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Visible band centres (micrometres) accepted as red, green, blue.
_RGB_RANGES_UM = ((0.62, 0.70), (0.52, 0.60), (0.44, 0.52))
_RGB_NAMES = (("red", "r", "b04"), ("green", "g", "b03"), ("blue", "b", "b02"))


def rgb_guidance_indices(bands: list[BandSpec]) -> tuple[int, int, int]:
    """Return channel indices for red, green, and blue, or the first band three times."""
    found: list[int] = []
    for (low, high), names in zip(_RGB_RANGES_UM, _RGB_NAMES, strict=True):
        index = next(
            (
                i
                for i, b in enumerate(bands)
                if b.wavelength_um is not None and low <= b.wavelength_um < high
            ),
            None,
        )
        if index is None:
            index = next((i for i, b in enumerate(bands) if b.name.lower() in names), None)
        if index is None:
            return (0, 0, 0)
        found.append(index)
    return found[0], found[1], found[2]


def anyup_guidance(images: torch.Tensor, rgb: tuple[int, int, int]) -> torch.Tensor:
    """Return ImageNet-normalised guidance from a per-image 2-98 percentile stretch of ``rgb``."""
    with torch.no_grad():
        guide = images[:, list(rgb)].float()
        flat = guide.flatten(1)
        low = torch.quantile(flat, 0.02, dim=1).view(-1, 1, 1, 1)
        high = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
        guide = ((guide - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        mean = guide.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = guide.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (guide - mean) / std


def _checkpoint_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / "anyup_multi_backbone.pth"


def load_anyup() -> nn.Module:
    """Load the pinned multi-backbone AnyUp, verify its checkpoint, and freeze it."""
    model = torch.hub.load(
        f"{ANYUP_REPO}:{ANYUP_REF}",
        ANYUP_ENTRYPOINT,
        trust_repo=True,
        skip_validation=True,
    )
    checkpoint = _checkpoint_path()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != ANYUP_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"AnyUp checkpoint {checkpoint} has sha256 {digest}, expected {ANYUP_CHECKPOINT_SHA256}."
        )
    for param in model.parameters():
        param.requires_grad = False
    return model.eval()


class AnyUpClassifier(nn.Module):
    """Upsample fused features to the guidance resolution with frozen AnyUp, then classify.

    The caller sets :attr:`guidance` before each forward. AnyUp stays in eval mode.
    """

    def __init__(self, classifier: nn.Module, upsampler: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier
        self.upsampler = upsampler
        self.guidance: torch.Tensor | None = None

    @override
    def train(self, mode: bool = True) -> "AnyUpClassifier":
        """Set train mode on the classifier only."""
        super().train(mode)
        self.upsampler.eval()
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return classifier logits at the guidance resolution."""
        if self.guidance is None:
            raise RuntimeError("AnyUp guidance image was not set before the head forward.")
        guidance = self.guidance.to(dtype=features.dtype)
        upsampled = self.upsampler(guidance, features, output_size=tuple(guidance.shape[-2:]))
        return self.classifier(upsampled)
