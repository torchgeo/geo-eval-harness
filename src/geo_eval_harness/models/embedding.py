"""Precomputed embedding products evaluated as models."""

from pathlib import Path

import torch
import torch.nn as nn

from geo_eval_harness.bands import BandSpec
from geo_eval_harness.embeddings.collection import read_metadata

from ._normalization import NormalizationStrategy
from .interface import BenchModel

__all__ = ["PrecomputedEmbedding", "embedding_bands"]


def embedding_bands(directory: Path) -> list[BandSpec]:
    """Describe the embedding channels of the collection in *directory* as bands."""
    channels = read_metadata(directory)["arrays"]["embedding"]["shape"][0]
    return [
        BandSpec("embedding", f"e{i:03d}", f"e{i:03d}", mean=0.0, std=1.0, min=-1.0, max=1.0)
        for i in range(channels)
    ]


class _FillMissing(nn.Module):
    """Replace missing (NaN) embedding cells with zeros."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0)


class PrecomputedEmbedding(BenchModel):
    """Pass stored embeddings through unchanged; the inputs already are the features.

    ``feature_map`` exposes the ``(B, C, H, W)`` map to segmentation heads.
    Classification uses the mean over cells without missing values.

    Args:
        bands: One :class:`BandSpec` per embedding channel.
        normalization: Must be ``"identity"``; stored embeddings are not normalized.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        normalization: NormalizationStrategy | str = NormalizationStrategy.IDENTITY,
    ) -> None:
        if NormalizationStrategy(normalization) is not NormalizationStrategy.IDENTITY:
            raise ValueError("Stored embeddings are not normalized; use --normalization none")
        super().__init__(bands, normalization=normalization)
        self.feature_map = _FillMissing()

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_map(images)
        valid = torch.isfinite(images).all(dim=1, keepdim=True)
        total = (features * valid).sum(dim=(2, 3))
        return total / valid.sum(dim=(2, 3)).clamp(min=1)
