"""MEurosat (GeoBench V1) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MEurosat(_V1Dataset):
    """Sentinel-2 land-use classification (10 classes).

    Based on the EuroSAT dataset with 13 Sentinel-2 spectral bands.
    """

    name = "m-eurosat"
    task = "classification"
    num_classes = 10
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 2000, "val": 1000, "test": 1000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "coastal_aerosol", "01 - Coastal aerosol", mean=1359.95, std=251.332, min=858, max=6805, wavelength_um=0.443),
        BandSpec("s2", "blue", "02 - Blue", mean=1125.53, std=339.685, min=0, max=28000, wavelength_um=0.49),
        BandSpec("s2", "green", "03 - Green", mean=1055, std=396.733, min=0, max=28000, wavelength_um=0.56),
        BandSpec("s2", "red", "04 - Red", mean=957.344, std=592.838, min=0, max=28000, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", mean=1219.67, std=555.015, min=182, max=23381, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", mean=2051.62, std=852.505, min=157, max=24975, wavelength_um=0.74),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", mean=2433.6, std=1081.57, min=131, max=25412, wavelength_um=0.783),
        BandSpec("s2", "nir", "08 - NIR", mean=2360.1, std=1115.12, min=0, max=28002, wavelength_um=0.842),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", mean=751.164, std=404.572, min=41, max=3761, wavelength_um=0.865),
        BandSpec("s2", "water_vapour", "09 - Water vapour", mean=12.2881, std=4.7966, min=1, max=90, wavelength_um=0.945),
        BandSpec("s2", "swir_cirrus", "10 - SWIR - Cirrus", mean=1848.9, std=978.83, min=7, max=24704, wavelength_um=1.375),
        BandSpec("s2", "swir_1", "11 - SWIR", mean=1131.27, std=745.284, min=1, max=22210, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "12 - SWIR", mean=2665.44, std=1223.88, min=95, max=25752, wavelength_um=2.19),
    ]
    # fmt: on
