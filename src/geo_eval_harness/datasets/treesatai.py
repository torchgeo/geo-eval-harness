"""TreeSatAI (GeoBench V2) benchmark dataset."""

from typing import ClassVar

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class TreeSatAI(_V2Dataset):
    """Aerial + Sentinel-2 + SAR tree species classification (15 classes).

    Multi-sensor dataset with aerial RGB+NIR, 12 Sentinel-2 bands, and 3 SAR bands. The Sentinel-1
    rasters hold whole-dB values and the no-data marker -9999 as two's-complement ``uint16``;
    :meth:`canonicalize_sample` restores the signed values and replaces no-data pixels with the
    mean of the channel's valid pixels in the sample.

    ``vv_vh`` is a dummy band, the VV/VH quotient of the dB values truncated to an integer, and
    carries almost no information. Select bands by name to leave it out.
    """

    band_order_strategy = "by_sensor"

    name = "treesatai"
    s1_no_data = -9999
    task = "classification"
    num_classes = 15
    class_names: ClassVar[tuple[str, ...]] = (
        "Abies",
        "Acer",
        "Alnus",
        "Betula",
        "Cleared",
        "Fagus",
        "Fraxinus",
        "Larix",
        "Picea",
        "Pinus",
        "Populus",
        "Prunus",
        "Pseudotsuga",
        "Quercus",
        "Tilia",
    )
    multilabel = True
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        # German DOP ortho-aerial centre wavelengths. The files store NIR, G, B, R, which upstream
        # reads as red, green, blue, nir; source names are upstream's labels for those channels.
        BandSpec("aerial", "nir", "red", mean=154.416, std=48.5986, min=0, max=255, wavelength_um=0.83),
        BandSpec("aerial", "green", "green", mean=92.4992, std=33.6488, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "blue", mean=85.5702, std=28.041, min=0, max=255, wavelength_um=0.48),
        BandSpec("aerial", "red", "nir", mean=79.8672, std=33.6009, min=0, max=255, wavelength_um=0.66),
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
        # Sentinel-1 statistics describe what canonicalize_sample returns: no-data pixels count at
        # their fill value, within 0.001 dB of leaving them out. They must never include the -9999
        # marker, which alone raises the VV std from 3.6 to 105.
        # vv_vh is 0 for 97.7% of training pixels; it stays declared for models that need a third
        # SAR channel.
        BandSpec("s1", "vv", "vv", mean=-5.9765, std=3.5617, min=-30, max=14, unit="db"),
        BandSpec("s1", "vh", "vh", mean=-12.1626, std=3.4576, min=-43, max=9, unit="db"),
        BandSpec("s1", "vv_vh", "vv/vh", mean=0.0194, std=0.3412, min=-85, max=46),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Decode Sentinel-1 to signed values and fill its no-data pixels.

        Stored values above 32767 are negative values plus 65536. No-data pixels (-9999, stored as
        55537) take the mean of the channel's valid pixels in the sample, so the marker reaches
        neither models nor band statistics.
        """
        s1 = sample.get("image_s1")
        if s1 is not None:
            s1 = torch.where(s1 > 32767, s1 - 65536, s1)
            valid = s1 != self.s1_no_data
            total = (s1 * valid).sum((-2, -1), keepdim=True)
            mean = total / valid.sum((-2, -1), keepdim=True).clamp(min=1)
            sample["image_s1"] = torch.where(valid, s1, mean)
        return sample
