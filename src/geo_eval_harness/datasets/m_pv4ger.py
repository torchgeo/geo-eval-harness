"""MPv4ger (GeoBench V1) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MPv4ger(_V1Dataset):
    """Aerial solar panel detection (2 classes).

    Based on the PV4GER dataset with 3 aerial RGB bands.
    """

    name = "m-pv4ger"
    task = "classification"
    num_classes = 2
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 11814, "val": 999, "test": 999}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        # NAIP centre wavelengths (4-band sensor; this dataset only ships RGB).
        BandSpec("aerial", "blue", "Blue", mean=116.316, std=44.5176, min=2, max=254, wavelength_um=0.45),
        BandSpec("aerial", "green", "Green", mean=119.375, std=48.1189, min=7, max=254, wavelength_um=0.55),
        BandSpec("aerial", "red", "Red", mean=113.102, std=54.0881, min=1, max=254, wavelength_um=0.65),
    ]
    # fmt: on
