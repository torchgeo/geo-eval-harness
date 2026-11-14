"""Dynamic EarthNet (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class DynamicEarthNet(_V2Dataset):
    """Planet + Sentinel-2 land-cover change segmentation (7 classes)."""

    band_order_strategy = "by_sensor"

    name = "dynamic_earthnet"
    task = "segmentation"
    num_classes = 7
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["r", "g", "b"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 700, "val": 100, "test": 200}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("planet", "b", "b", mean=664.423, std=639.946, min=10, max=10051),
        BandSpec("planet", "g", "g", mean=929.265, std=805.98, min=17, max=10039),
        BandSpec("planet", "r", "r", mean=1031.28, std=1072.23, min=9, max=10057),
        BandSpec("planet", "nir", "nir", mean=2605.98, std=1182.39, min=14, max=9998),
        BandSpec("s2", "b01", "B01", mean=1142.08, std=1588.17, min=0, max=15535, wavelength_um=0.443),
        BandSpec("s2", "b02", "B02", mean=1399.03, std=1516.62, min=1, max=15621, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=1429.55, std=1606.99, min=1, max=15541, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=2782.08, std=1497.91, min=0, max=15434, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=1799.14, std=1601.56, min=1, max=16588, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=2486.87, std=1470.5, min=1, max=16479, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=2749.07, std=1499.74, min=1, max=16621, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=2899.32, std=1506.38, min=1, max=16560, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=2369.53, std=1487.68, min=1, max=11401, wavelength_um=0.865),
        BandSpec("s2", "b10", "B10", mean=1732.49, std=1393.75, min=1, max=15547, wavelength_um=1.375),
        BandSpec("s2", "b11", "B11", mean=1049.82, std=1654.71, min=0, max=15633, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=3192, std=2299.15, min=0, max=16726, wavelength_um=2.19),
    ]
    # fmt: on
