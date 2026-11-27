"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from dataclasses import replace
from typing import ClassVar

from torch.utils.data import Dataset

from .base import BandSpec, Footprint
from .geobench_v2 import _OffsetMaskV2Dataset


class SpaceNet7(_OffsetMaskV2Dataset):
    """Planet building footprint segmentation (2 classes).

    One Planet RGB image and mask per sample: segmentation, not change detection.

    Restore upstream ``{1, 2}`` masks to native ``{0, 1}`` building labels.
    """

    name = "spacenet7"
    task = "segmentation"
    num_classes = 2
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 3500, "val": 652, "test": 1152}
    footprint_asset = "images"

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("planet", "red", "red", mean=117.85, std=61.9829, min=0, max=255),
        BandSpec("planet", "green", "green", mean=104.531, std=49.7879, min=0, max=255),
        BandSpec("planet", "blue", "blue", mean=77.561, std=46.01, min=0, max=255),
    ]
    # fmt: on

    def footprint(self, dataset: Dataset, index: int) -> Footprint:
        """Flip the header's positive y spacing: rows run north to south from the origin."""
        footprint = super().footprint(dataset, index)
        a, b, c, d, e, f = footprint.transform
        return replace(footprint, transform=(a, b, c, d, -abs(e), f))
