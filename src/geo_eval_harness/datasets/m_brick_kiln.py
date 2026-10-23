"""MBrickKiln (GeoBench V1) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MBrickKiln(_V1Dataset):
    """Sentinel-2 brick kiln detection (2 classes).

    Based on the Brick-Kiln dataset with 13 Sentinel-2 spectral bands.
    """

    name = "m-brick-kiln"
    task = "classification"
    num_classes = 2
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 15063, "val": 999, "test": 999}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "coastal_aerosol", "01 - Coastal aerosol", mean=572.205, std=190.09, min=9.6923, max=2823.22, wavelength_um=0.443),
        BandSpec("s2", "blue", "02 - Blue", mean=669, std=234.367, min=48.6667, max=3959.5, wavelength_um=0.49),
        BandSpec("s2", "green", "03 - Green", mean=879.878, std=272.815, min=98, max=5260.67, wavelength_um=0.56),
        BandSpec("s2", "red", "04 - Red", mean=807.583, std=358.861, min=58.3333, max=5365.33, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", mean=1127.9, std=361.903, min=108.286, max=5057.48, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", mean=1960.74, std=721.657, min=83.7143, max=11046.8, wavelength_um=0.74),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", mean=2075.83, std=813.801, min=73.4615, max=11896.5, wavelength_um=0.783),
        BandSpec("s2", "nir", "08 - NIR", mean=2039.48, std=789.61, min=86, max=12515, wavelength_um=0.842),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", mean=1625.46, std=788.954, min=51.5, max=5887.48, wavelength_um=0.865),
        BandSpec("s2", "water_vapour", "09 - Water vapour", mean=1135.59, std=692.586, min=33.1111, max=7890, wavelength_um=0.945),
        BandSpec("s2", "swir_cirrus", "10 - SWIR - Cirrus", mean=82.471, std=36.0834, min=7.5, max=255, wavelength_um=1.375),
        BandSpec("s2", "swir_1", "11 - SWIR", mean=89.8659, std=27.5559, min=10.3333, max=255, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "12 - SWIR", mean=68.4528, std=23.7711, min=6, max=255, wavelength_um=2.19),
    ]
    # fmt: on
