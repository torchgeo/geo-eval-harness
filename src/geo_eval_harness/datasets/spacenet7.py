"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
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

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("planet", "red", "red", mean=117.85, std=61.9829, min=0, max=255),
        BandSpec("planet", "green", "green", mean=104.531, std=49.7879, min=0, max=255),
        BandSpec("planet", "blue", "blue", mean=77.561, std=46.01, min=0, max=255),
    ]
    # fmt: on
