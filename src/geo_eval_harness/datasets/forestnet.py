"""Forestnet (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class Forestnet(_V2Dataset):
    """Landsat-8 forest-change classification (12 classes).

    GeoBench V2 repackages the ``m-forestnet`` chips and split under Sentinel-2 band codes.
    """

    name = "forestnet"
    task = "classification"
    num_classes = 12
    class_names: ClassVar[tuple[str, ...]] = (
        "Oil palm plantation",
        "Timber plantation",
        "Other large-scale plantations",
        "Grassland/shrubland",
        "Small-scale agriculture",
        "Small-scale mixed plantation",
        "Small-scale oil palm plantation",
        "Mining",
        "Fish pond",
        "Logging road",
        "Secondary forest",
        "Other",
    )
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 6464, "val": 989, "test": 993}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("landsat", "blue", "B02", mean=72.3759, std=16.2839, min=0, max=255, wavelength_um=0.49),
        BandSpec("landsat", "green", "B03", mean=83.1816, std=15.3587, min=0, max=255, wavelength_um=0.56),
        BandSpec("landsat", "red", "B04", mean=77.0861, std=16.6665, min=0, max=255, wavelength_um=0.665),
        BandSpec("landsat", "nir", "B8A", mean=123.543, std=16.9485, min=0, max=250, wavelength_um=0.865),
        BandSpec("landsat", "swir_1", "B11", mean=91.0484, std=14.2801, min=0, max=255, wavelength_um=1.61),
        BandSpec("landsat", "swir_2", "B12", mean=74.3097, std=13.2854, min=0, max=255, wavelength_um=2.19),
    ]
    # fmt: on
