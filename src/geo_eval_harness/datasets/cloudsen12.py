"""CloudSEN12 (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class CloudSEN12(_V2Dataset):
    """Sentinel-2 cloud segmentation (4 classes)."""

    name = "cloudsen12"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 535, "test": 975}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b01", "B01", mean=1973.24, std=2704.21, min=0, max=26044, wavelength_um=0.443),
        BandSpec("s2", "b02", "B02", mean=2011.85, std=2650.55, min=0, max=25520, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=2148.43, std=2500.88, min=0, max=22800, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=2182.13, std=2500.84, min=0, max=20816, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=2526.47, std=2472.63, min=0, max=19952, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=3055.09, std=2212.44, min=0, max=19140, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=3239.42, std=2124.23, min=0, max=18724, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=3293.5, std=2151.72, min=0, max=18416, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=3347.9, std=2049.76, min=0, max=18096, wavelength_um=0.865),
        BandSpec("s2", "b09", "B09", mean=4046.78, std=3147.31, min=0, max=17844, wavelength_um=0.945),
        BandSpec("s2", "b11", "B11", mean=2443.39, std=1583.54, min=0, max=15948, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=1894.92, std=1464.28, min=0, max=16124, wavelength_um=2.19),
    ]
    # fmt: on
