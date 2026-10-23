"""So2Sat (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class So2Sat(_V2Dataset):
    """Sentinel-2 + SAR local climate zone classification (17 classes).

    GeoBench V2 version with 10 Sentinel-2 and 2 SAR bands.
    """

    band_order_strategy = "by_sensor"

    name = "so2sat"
    task = "classification"
    num_classes = 17
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 19992, "val": 986, "test": 986}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b02", "B02", mean=0.1295, std=0.0414, min=0.0001, max=2.8, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=0.1172, std=0.052, min=0.0001, max=2.8, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=0.1138, std=0.0733, min=0.0001, max=2.8, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=0.1272, std=0.0693, min=0.0001, max=2.8001, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=0.1707, std=0.075, min=0.0001, max=2.8002, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=0.1928, std=0.0856, min=0.0001, max=2.8002, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=0.1855, std=0.0865, min=0.0001, max=2.8001, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=0.2073, std=0.094, min=0.0001, max=2.8001, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=0.1768, std=0.1024, min=0.0001, max=2.8, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=0.1285, std=0.0923, min=0.0001, max=2.8, wavelength_um=2.19),
        BandSpec("s1", "vv", "VV", mean=0, std=0.2156, min=-107.636, max=45.3088),
        BandSpec("s1", "vh", "VH", mean=0, std=0.5442, min=-107.636, max=108.199),
    ]
    # fmt: on
