"""MSo2Sat (GeoBench V1) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v1 import _V1Dataset


class MSo2Sat(_V1Dataset):
    """Sentinel-2 + SAR local climate zone classification (17 classes).

    Based on the So2Sat dataset with 10 Sentinel-2 and 8 SAR bands.
    """

    name = "m-so2sat"
    task = "classification"
    num_classes = 17
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 19992, "val": 986, "test": 986}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("sar", "vh_real", "01 - VH.Real", mean=0, std=0.2156, min=-107.636, max=45.3088),
        BandSpec("s2", "blue", "02 - Blue", mean=0.1295, std=0.0414, min=0.0001, max=2.8, wavelength_um=0.49),
        BandSpec("sar", "vh_imag", "02 - VH.Imaginary", mean=0.0001, std=0.2142, min=-107.636, max=107.633),
        BandSpec("s2", "green", "03 - Green", mean=0.1172, std=0.052, min=0.0001, max=2.8, wavelength_um=0.56),
        BandSpec("sar", "vv_real", "03 - VV.Real", mean=0, std=0.5442, min=-107.636, max=108.199),
        BandSpec("s2", "red", "04 - Red", mean=0.1138, std=0.0733, min=0.0001, max=2.8, wavelength_um=0.665),
        BandSpec("sar", "vv_imag", "04 - VV.Imaginary", mean=-0.0001, std=0.5328, min=-101.503, max=107.633),
        BandSpec("sar", "vh_lee", "05 - VH.LEE Filtered", mean=0.0615, std=5.4568, min=0, max=10867.4),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", mean=0.1272, std=0.0693, min=0.0001, max=2.8001, wavelength_um=0.705),
        BandSpec("sar", "vv_lee", "06 - VV.LEE Filtered", mean=0.3435, std=11.8128, min=0.0005, max=9950.03),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", mean=0.1707, std=0.075, min=0.0001, max=2.8002, wavelength_um=0.74),
        BandSpec("sar", "vh_lee_real", "07 - VH.LEE Filtered.Real", mean=0.0006, std=3.7719, min=-2167.66, max=7875.37),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", mean=0.1928, std=0.0856, min=0.0001, max=2.8002, wavelength_um=0.783),
        BandSpec("s2", "nir", "08 - NIR", mean=0.1855, std=0.0865, min=0.0001, max=2.8001, wavelength_um=0.842),
        BandSpec("sar", "vv_lee_imag", "08 - VV.LEE Filtered.Imaginary", mean=0.0032, std=2.4939, min=-1453.07, max=5448.86),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", mean=0.2073, std=0.094, min=0.0001, max=2.8001, wavelength_um=0.865),
        BandSpec("s2", "swir_1", "11 - SWIR", mean=0.1768, std=0.1024, min=0.0001, max=2.8, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "12 - SWIR", mean=0.1285, std=0.0923, min=0.0001, max=2.8, wavelength_um=2.19),
    ]
    # fmt: on
