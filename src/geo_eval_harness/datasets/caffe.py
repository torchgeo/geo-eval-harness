"""CaFFe (GeoBench V2) benchmark dataset."""

from typing import ClassVar

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset

# Statistics of the single grayscale (SAR amplitude) channel, shared by its RGB copies.
_GRAY_STATS = {"mean": 68.4868, "std": 82.7774, "min": 0, "max": 255}


class CaFFe(_V2Dataset):
    """Grayscale calving-front segmentation (4 classes), served as a 3-channel RGB-like image.

    The imagery is single-channel SAR amplitude. Every model gets the same input by repeating the
    gray channel into ``red``/``green``/``blue`` band specs without wavelengths, like the RGB-only
    SpaceNet7, so each model uses its standard RGB setup (timm stems, DOFA/Panopticon default RGB
    wavelengths, TerraMind's RGB modality, OlmoEarth's aerial RGB path). Pseudo-RGB from SAR is a
    modelling choice; selecting fewer bands gives fewer copies of the same channel.
    """

    name = "caffe"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", name, "gray", **_GRAY_STATS) for name in ("red", "green", "blue")
    ]

    _copies: int = 3

    def build_band_order(self, bands: tuple[str, ...] | None) -> object:
        """Load the gray channel once; :meth:`canonicalize_sample` repeats it per selected band."""
        self._copies = len(self.select_band_specs(bands))
        return ["gray"]

    def canonicalize_sample(self, sample: dict) -> dict:
        """Repeat the ``(1, H, W)`` gray image into one identical channel per selected band."""
        image = torch.as_tensor(sample["image"])
        sample["image"] = image.repeat(self._copies, *([1] * (image.ndim - 1)))
        return sample
