"""MForestnet (GeoBench V1) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MForestnet(_V1Dataset):
    """Landsat forest-change classification (12 classes).

    Based on the ForestNet dataset with 6 Landsat spectral bands.
    """

    name = "m-forestnet"
    task = "classification"
    num_classes = 12
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 6464, "val": 989, "test": 993}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("landsat", "blue", "02 - Blue", mean=72.376, std=16.2839, min=0, max=255, wavelength_um=0.49),
        BandSpec("landsat", "green", "03 - Green", mean=83.1816, std=15.3587, min=0, max=255, wavelength_um=0.56),
        BandSpec("landsat", "red", "04 - Red", mean=77.0862, std=16.6665, min=0, max=255, wavelength_um=0.665),
        BandSpec("landsat", "nir", "05 - NIR", mean=123.543, std=16.9485, min=0, max=250),
        BandSpec("landsat", "swir_1", "06 - SWIR1", mean=91.0484, std=14.2801, min=0, max=255),
        BandSpec("landsat", "swir_2", "07 - SWIR2", mean=74.3097, std=13.2854, min=0, max=255),
    ]
    # fmt: on
