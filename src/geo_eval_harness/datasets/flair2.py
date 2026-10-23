"""FLAIR2 (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class FLAIR2(_V2Dataset):
    """French aerial land-cover segmentation (13 classes).

    RGB, NIR, and elevation channels share one image tensor and a flat ``band_order`` list.
    """

    name = "flair2"
    task = "segmentation"
    num_classes = 13
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4049, "val": 1022, "test": 3022}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        # IGN BD ORTHO centre wavelengths (R/G/B/NIR); elevation is non-spectral.
        BandSpec("aerial", "red", "red", mean=111.395, std=51.2846, min=0, max=255, wavelength_um=0.66),
        BandSpec("aerial", "green", "green", mean=115.788, std=45.1, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "blue", mean=106.896, std=44.4006, min=0, max=255, wavelength_um=0.48),
        BandSpec("aerial", "nir", "nir", mean=104.085, std=39.566, min=0, max=255, wavelength_um=0.83),
        BandSpec("elevation", "elevation", "elevation", mean=17.7749, std=30.34, min=0, max=255),
    ]
    # fmt: on
