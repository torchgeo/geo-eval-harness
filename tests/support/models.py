"""Small model inputs and fixtures shared by model contract tests."""

import torch
from torch import nn

from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.models.interface import BenchModel


def bands(n: int = 2) -> list[BandSpec]:
    """Return distinct per-channel statistics without loading a dataset."""
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=float(10 * (i + 1)),
            std=float(2 * (i + 1)),
            min=0.0,
            max=255.0,
        )
        for i in range(n)
    ]


class DatedBenchModel(BenchModel):
    """Expose dates in a spatial map without pretrained weights or model dependencies."""

    def __init__(self) -> None:
        super().__init__(bands(1), normalization="identity")
        self.spatial_features = nn.Identity()

    def _encode(self, images: torch.Tensor, doy: torch.Tensor | None) -> torch.Tensor:
        dates = images.new_zeros(images.shape[0]) if doy is None else doy.to(images)
        dense = self.spatial_features(images + dates[:, None, None, None])
        return dense.mean(dim=(-2, -1))

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._encode(images, None)

    def forward_batch(
        self, images: torch.Tensor, *, doy: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Include dates only through the batch-aware interface."""
        return self._encode(self.normalize_inputs(images), doy)
