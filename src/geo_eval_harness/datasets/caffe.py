"""CaFFe (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class CaFFe(_V2Dataset):
    """Aerial grayscale calving-front segmentation (4 classes)."""

    name = "caffe"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["gray"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", "gray", "gray", mean=68.4868, std=82.7774, min=0, max=255),
    ]
    # fmt: on
