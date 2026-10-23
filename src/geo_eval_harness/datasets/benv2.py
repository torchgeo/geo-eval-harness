"""BENV2 (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class BENV2(_V2Dataset):
    """Sentinel-2 + SAR multi-label classification (19 classes).

    BigEarthNet V2 with 12 Sentinel-2 optical bands and 2 SAR bands.
    """

    band_order_strategy = "by_sensor"

    name = "benv2"
    task = "classification"
    num_classes = 19
    multilabel = True
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 20000, "val": 4000, "test": 4000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b01", "B01", mean=356.468, std=551.961, min=1, max=11450, wavelength_um=0.443),
        BandSpec("s2", "b02", "B02", mean=434.675, std=594.998, min=1, max=20592, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=612.172, std=595.693, min=1, max=18640, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=587.062, std=677.559, min=1, max=17280, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=942.387, std=720.559, min=1, max=16706, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=1769.64, std=1083.08, min=1, max=16408, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=2049.96, std=1258.08, min=1, max=16285, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=2192.89, std=1365.99, min=1, max=16280, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=2236.25, std=1338.89, min=1, max=16071, wavelength_um=0.865),
        BandSpec("s2", "b09", "B09", mean=2242.3, std=1294.24, min=1, max=13792, wavelength_um=0.945),
        BandSpec("s2", "b11", "B11", mean=1576.53, std=1064.65, min=1, max=15351, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=1004.48, std=809.08, min=1, max=15213, wavelength_um=2.19),
        BandSpec("s1", "vv", "VV", mean=-19.361, std=5.6033, min=-66.5271, max=24.3281),
        BandSpec("s1", "vh", "VH", mean=-12.6317, std=5.094, min=-65.3005, max=33.5545),
    ]
    # fmt: on
