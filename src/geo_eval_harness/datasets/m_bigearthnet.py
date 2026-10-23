"""MBigEarthNet (GeoBench V1) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MBigEarthNet(_V1Dataset):
    """Sentinel-2 multi-label land-cover classification (43 classes).

    Based on the BigEarthNet dataset with 12 Sentinel-2 spectral bands.
    Uses multi-hot label encoding.
    """

    name = "m-bigearthnet"
    task = "classification"
    num_classes = 43
    multilabel = True
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 20000, "val": 1000, "test": 1000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "coastal_aerosol", "01 - Coastal aerosol", mean=378.402, std=462.463, min=1, max=18268, wavelength_um=0.443),
        BandSpec("s2", "blue", "02 - Blue", mean=482.274, std=519.331, min=0, max=20545, wavelength_um=0.49),
        BandSpec("s2", "green", "03 - Green", mean=706.537, std=552.357, min=0, max=18989, wavelength_um=0.56),
        BandSpec("s2", "red", "04 - Red", mean=720.926, std=680.972, min=0, max=17881, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", mean=1100.67, std=690.282, min=0, max=16186, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", mean=1909.29, std=982.218, min=0, max=16039, wavelength_um=0.74),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", mean=2191.7, std=1143.42, min=0, max=15956, wavelength_um=0.783),
        BandSpec("s2", "nir", "08 - NIR", mean=2336.86, std=1248.04, min=0, max=16708, wavelength_um=0.842),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", mean=2394.74, std=1223.65, min=0, max=15825, wavelength_um=0.865),
        BandSpec("s2", "water_vapour", "09 - Water vapour", mean=2368.32, std=1166.83, min=1, max=15593, wavelength_um=0.945),
        BandSpec("s2", "swir_1", "11 - SWIR", mean=1875.26, std=1092.42, min=0, max=15422, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "12 - SWIR", mean=1229.38, std=862.716, min=0, max=15258, wavelength_um=2.19),
    ]
    # fmt: on
