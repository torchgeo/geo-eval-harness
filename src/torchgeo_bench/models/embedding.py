"""Precomputed embedding products evaluated as models."""

import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

__all__ = ["PrecomputedEmbedding"]


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
        product: Stored collection name, e.g. ``"aef"`` or ``"aef-pca16"``.
    """

    handles_own_normalization = True

    def __init__(self, bands: list[BandSpec], product: str, **kwargs: object) -> None:
        kwargs.pop("normalization", None)
        super().__init__(bands, normalization="identity", **kwargs)
        self.product = product
        self.feature_map = _FillMissing()

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_map(images)
        valid = torch.isfinite(images).all(dim=1, keepdim=True)
        total = (features * valid).sum(dim=(2, 3))
        return total / valid.sum(dim=(2, 3)).clamp(min=1)
