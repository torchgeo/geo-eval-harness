"""NWPU-RESISC45 aerial scene classification via torchgeo."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from torch.utils.data import Dataset
from torchgeo.datasets import RESISC45 as TGRESISC45

from ._transforms import select_bands
from .base import BandSpec, BenchDataset


class RESISC45(BenchDataset):
    """Aerial scene classification, 45 classes, via torchgeo.

    31,500 RGB images at 256x256, with 700 images per class.

    Northwestern Polytechnical University sourced them from Google Earth.

    Uses torchgeo's 60/20/20 split: 18,900 train, 6,300 validation, and 6,300 test.

    8-bit imagery has no per-image geolocation, scale, or radiometric calibration.

    Wavelengths are nominal visible-light centres, not measured sensor responses.

    Images span 0.2--30 m/px; the ``aerial`` tag's 1 m GSD is only an approximation.

    This lets resolution-aware models accept them as RGB aerial imagery.
    """

    name = "resisc45"
    task = "classification"
    num_classes = 45
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 18900, "val": 6300, "test": 6300}
    supports_partitions = False

    # Train-split statistics in raw 0-255 units; see scripts/compute_band_statistics.py.
    # ``source_name`` identifies the RGB channel, not a band key in the JPEG.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", "red", "R", mean=93.8939, std=51.8492, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=97.1123, std=47.2366, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=87.5678, std=47.0631, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/resisc45")``; torchgeo manages the layout below."""
        return Path("data/resisc45")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the wrapped torchgeo dataset for the split.

        Select bands before transforms so unused RGB channels are not resized.
        """
        del partition
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        transform = select_bands(indices, len(self.bands), transform)
        return TGRESISC45(
            root=str(self.data_root()),
            split=split,
            transforms=transform,
        )
