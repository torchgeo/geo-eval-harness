"""Forestnet (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class Forestnet(_V2Dataset):
    """Sentinel-2 forest-change classification (12 classes).

    GeoBench V2 version with 6 Sentinel-2 spectral bands.
    """

    name = "forestnet"
    task = "classification"
    num_classes = 12
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 6464, "val": 989, "test": 993}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b02", "B02", mean=72.3759, std=16.2839, min=0, max=255, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=83.1816, std=15.3587, min=0, max=255, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=77.0861, std=16.6665, min=0, max=255, wavelength_um=0.665),
        BandSpec("s2", "b8a", "B8A", mean=123.543, std=16.9485, min=0, max=250, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=91.0484, std=14.2801, min=0, max=255, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=74.3097, std=13.2854, min=0, max=255, wavelength_um=2.19),
    ]
    # fmt: on
