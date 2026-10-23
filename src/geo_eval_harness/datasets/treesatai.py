"""TreeSatAI (GeoBench V2) benchmark dataset."""

from typing import ClassVar

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class TreeSatAI(_V2Dataset):
    """Aerial + Sentinel-2 + SAR tree species classification (15 classes).

    Multi-sensor dataset with aerial RGB+NIR, 12 Sentinel-2 bands, and 3 SAR bands.
    Class indices follow the upstream ``GeoBenchTreeSatAI.classes`` ordering:
    Abies, Acer, Alnus, Betula, Cleared, Fagus, Fraxinus, Larix, Picea, Pinus,
    Populus, Prunus, Pseudotsuga, Quercus, Tilia.
    """

    band_order_strategy = "by_sensor"

    name = "treesatai"
    task = "classification"
    num_classes = 15
    multilabel = True
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        # German DOP ortho-aerial centre wavelengths (R/G/B/NIR).
        BandSpec("aerial", "red", "red", mean=154.416, std=48.5986, min=0, max=255, wavelength_um=0.66),
        BandSpec("aerial", "green", "green", mean=92.4992, std=33.6488, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "blue", mean=85.5702, std=28.041, min=0, max=255, wavelength_um=0.48),
        BandSpec("aerial", "nir", "nir", mean=79.8672, std=33.6009, min=0, max=255, wavelength_um=0.83),
        BandSpec("s2", "b02", "B02", mean=241.428, std=129.435, min=0, max=3059, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=384.216, std=142.58, min=0, max=3253, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=247.127, std=148.153, min=0, max=3195, wavelength_um=0.665),
        BandSpec("s2", "b08", "B08", mean=2828.33, std=762.904, min=0, max=6124, wavelength_um=0.842),
        BandSpec("s2", "b05", "B05", mean=623.686, std=202.742, min=0, max=3062, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=2116.06, std=510.903, min=0, max=4433, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=2710.51, std=688.66, min=0, max=5808, wavelength_um=0.783),
        BandSpec("s2", "b8a", "B8A", mean=2985.2, std=752.79, min=0, max=6175, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=1318.65, std=417.57, min=0, max=4093, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=594.947, std=250.849, min=0, max=3471, wavelength_um=2.19),
        BandSpec("s2", "b01", "B01", mean=255.617, std=127.578, min=0, max=2050, wavelength_um=0.443),
        BandSpec("s2", "b09", "B09", mean=2972.19, std=682.024, min=0, max=5569, wavelength_um=0.945),
        BandSpec("s1", "vv", "vv", mean=60197.8, std=17913.3, min=0, max=65535),
        BandSpec("s1", "vh", "vh", mean=65496.9, std=1326.41, min=0, max=65535),
        BandSpec("s1", "vv_vh", "vv/vh", mean=88.73, std=2409.44, min=0, max=65535),
    ]
    # fmt: on
