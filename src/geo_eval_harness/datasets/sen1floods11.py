# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Sen1Floods11 hand-labelled flood-water segmentation from Sentinel-1 or Sentinel-2."""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets.utils import Sample

from ._transforms import select_bands
from .base import BandSpec, BenchDataset

CHIP_DIR = "data/flood_events/HandLabeled"
SPLIT_DIR = "splits/flood_handlabeled"
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
}


def chip_path(chip: str, layer: str) -> str:
    """Return a chip layer's path relative to the v1.1 release root."""
    return f"{CHIP_DIR}/{layer}/{chip}_{layer}.tif"


def _read(root: Path, chip: str, layer: str) -> np.ndarray:
    with rasterio.open(root / chip_path(chip, layer)) as raster:
        return raster.read()


class Sen1Floods11Dataset(Dataset):
    """Load one sensor's chip with the mask shared by both sensors."""

    def __init__(
        self,
        root: Path,
        chips: list[str],
        layer: str,
        fill: np.ndarray,
        transform: Callable[[Sample], Sample] | None,
    ) -> None:
        self.root = root
        self.chips = chips
        self.layer = layer
        self.fill = fill
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, index: int) -> Sample:
        chip = self.chips[index]
        s1 = _read(self.root, chip, "S1Hand")
        s2 = _read(self.root, chip, "S2Hand")
        label = _read(self.root, chip, "LabelHand")[0]
        # Ignore label -1, S1 NaN and S2 zero fill for both sensors so their masks match.
        valid = (label != -1) & np.isfinite(s1).all(0) & s2.any(0)
        mask = np.where(valid, label, 255).astype(np.int64)
        if self.layer == "S1Hand":
            image = np.where(np.isfinite(s1), s1, self.fill[:, None, None])
        else:
            image = s2
        sample = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask),
        }
        return self.transform(sample) if self.transform is not None else sample


class Sen1Floods11Base(BenchDataset):
    """Shared splits and loading for the two Sen1Floods11 sensors.

    Source: Sen1Floods11 v1.1 (Bonafilia et al., CVPRW 2020), public GCS bucket
    ``gs://sen1floods11``. Only the hand-labelled chips of the official
    ``flood_{train,valid,test}_data.csv`` splits are used (252/89/90; the Bolivia hold-out and
    the weak labels are not). Chips are 512 x 512 in EPSG:4326 at about 10 m.

    Classes: 0 = not water, 1 = water. Native -1 (no data, mostly clouds), non-finite S1 and
    all-zero S2 pixels are 255 for both sensors, so both ids score the same pixels. Non-finite
    S1 values are filled with the band's train mean.

    The splits are random chips from the same 10 flood events, so they are spatially
    correlated. A few val/test chips are entirely ignored.
    """

    task = "segmentation"
    num_classes = 2
    split_sizes: ClassVar[dict[str, int]] = {"train": 252, "val": 89, "test": 90}
    layer: ClassVar[str]

    @classmethod
    def data_root(cls) -> Path:
        """Return the release root shared by both sensors."""
        return Path("data") / "sen1floods11"

    @classmethod
    def split_chips(cls, root: Path, split: str) -> list[str]:
        """Return the official chips of *split*, checking that all their files exist."""
        if split not in SPLIT_FILES:
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        path = root / SPLIT_DIR / SPLIT_FILES[split]
        hint = f"Run: geo-eval-harness download {cls.name}"
        if not path.is_file():
            raise FileNotFoundError(f"Sen1Floods11: missing {path}. {hint}")
        with path.open(newline="") as stream:
            chips = [row[0].removesuffix("_S1Hand.tif") for row in csv.reader(stream) if row]
        if len(chips) != cls.split_sizes[split]:
            raise ValueError(
                f"Sen1Floods11: {path} lists {len(chips)} chips, expected {cls.split_sizes[split]}"
            )
        for chip in chips:
            for layer in ("S1Hand", "S2Hand", "LabelHand"):
                if not (root / chip_path(chip, layer)).is_file():
                    raise FileNotFoundError(
                        f"Sen1Floods11: missing {root / chip_path(chip, layer)}. {hint}"
                    )
        return chips

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Load raw float32 values for the selected bands and the shared two-class mask."""
        if partition != "default":
            raise ValueError(f"{type(self).__name__} supports only the default partition")
        indices = [self.bands.index(spec) for spec in self.select_band_specs(bands)]
        fill = np.array([spec.mean for spec in self.bands], dtype=np.float32)
        root = self.data_root()
        return Sen1Floods11Dataset(
            root,
            self.split_chips(root, split),
            self.layer,
            fill,
            select_bands(indices, len(self.bands), transform),
        )


class Sen1Floods11S1(Sen1Floods11Base):
    """Sen1Floods11 water segmentation from Sentinel-1 GRD VV/VH backscatter in dB."""

    name = "sen1floods11-s1"
    layer: ClassVar[str] = "S1Hand"
    rgb_bands: ClassVar[list[str]] = ["vv", "vh"]
    # Train split, valid (mask != 255) pixels only; see the PR body for the command.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s1", "vv", "VV", mean=-10.4486, std=4.0614, min=-65.45, max=29.9264, unit="db"),
        BandSpec("s1", "vh", "VH", mean=-17.3155, std=4.7851, min=-71.6397, max=17.3764, unit="db"),
    ]
    # fmt: on


class Sen1Floods11S2(Sen1Floods11Base):
    """Sen1Floods11 water segmentation from 13-band Sentinel-2 L1C (DN, reflectance x 10⁴)."""

    name = "sen1floods11-s2"
    layer: ClassVar[str] = "S2Hand"
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    # Train split, valid (mask != 255) pixels only; see the PR body for the command.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b01", "B1", mean=1500.3681, std=327.8146, min=0, max=6885, wavelength_um=0.443, bandwidth_um=0.027, gsd_m=60),
        BandSpec("s2", "b02", "B2", mean=1253.4041, std=351.7053, min=0, max=11630, wavelength_um=0.49, bandwidth_um=0.098, gsd_m=10),
        BandSpec("s2", "b03", "B3", mean=1227.5549, std=374.3196, min=0, max=15100, wavelength_um=0.56, bandwidth_um=0.045, gsd_m=10),
        BandSpec("s2", "b04", "B4", mean=1062.0387, std=531.9318, min=0, max=17418, wavelength_um=0.665, bandwidth_um=0.038, gsd_m=10),
        BandSpec("s2", "b05", "B5", mean=1335.5951, std=451.5526, min=0, max=9760, wavelength_um=0.705, bandwidth_um=0.019, gsd_m=20),
        BandSpec("s2", "b06", "B6", mean=2298.6829, std=686.3296, min=0, max=10788, wavelength_um=0.74, bandwidth_um=0.018, gsd_m=20),
        BandSpec("s2", "b07", "B7", mean=2770.1801, std=881.719, min=0, max=11306, wavelength_um=0.783, bandwidth_um=0.028, gsd_m=20),
        BandSpec("s2", "b08", "B8", mean=2558.1267, std=850.5444, min=0, max=15516, wavelength_um=0.842, bandwidth_um=0.145, gsd_m=10),
        BandSpec("s2", "b8a", "B8A", mean=3007.48, std=1009.5349, min=0, max=13253, wavelength_um=0.865, bandwidth_um=0.033, gsd_m=20),
        BandSpec("s2", "b09", "B9", mean=459.5132, std=294.0046, min=0, max=3947, wavelength_um=0.945, bandwidth_um=0.026, gsd_m=60),
        BandSpec("s2", "b10", "B10", mean=62.3493, std=146.7964, min=0, max=1426, wavelength_um=1.375, bandwidth_um=0.075, gsd_m=60),
        BandSpec("s2", "b11", "B11", mean=1934.905, std=808.4762, min=0, max=14093, wavelength_um=1.61, bandwidth_um=0.143, gsd_m=20),
        BandSpec("s2", "b12", "B12", mean=1070.1452, std=603.4922, min=0, max=13472, wavelength_um=2.19, bandwidth_um=0.242, gsd_m=20),
    ]
    # fmt: on
