# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Official DFC2025 Track I SAR-only semantic segmentation."""

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

SPLITS = {"train": ("TrainArea", 4333), "val": ("ValArea", 210), "test": ("TestArea", 490)}
RASTER_SIZE = 1024


def split_filenames(split: str) -> list[str]:
    """Return the official membership in numeric order."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
    prefix, count = SPLITS[split]
    return [f"{prefix}_{index:03d}.tif" for index in range(1, count + 1)]


def split_paths(root: Path, split: str) -> list[tuple[Path, Path]]:
    """Validate exact image/label membership and return numerically ordered pairs."""
    names = split_filenames(split)
    for kind in ("sar_images", "labels"):
        directory = root / split / kind
        actual = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        }
        missing, extra = set(names) - actual, actual - set(names)
        if missing:
            raise FileNotFoundError(
                f"OpenEarthMap-SAR: {directory} missing {len(missing)} TIFFs "
                f"(e.g. {min(missing)}). Run: geo-eval-harness download openearthmap-sar"
            )
        if extra:
            raise ValueError(f"OpenEarthMap-SAR: unexpected TIFF in {directory}: {min(extra)}")
    return [(root / split / "sar_images" / name, root / split / "labels" / name) for name in names]


def read_raster(path: Path) -> np.ndarray:
    """Read a single-band uint8 tile at the dataset's native resolution."""
    with rasterio.open(path) as raster:
        if (
            raster.count != 1
            or raster.dtypes != ("uint8",)
            or raster.shape != (RASTER_SIZE, RASTER_SIZE)
        ):
            raise ValueError(
                f"OpenEarthMap-SAR: {path} must be single-band uint8 "
                f"{RASTER_SIZE} x {RASTER_SIZE}; got {raster.count} bands, "
                f"{raster.dtypes}, {raster.shape}"
            )
        return raster.read(1)


class SARSegmentationDataset(Dataset):
    """Load paired SAR images and eight-class masks for one split."""

    def __init__(
        self, paths: list[tuple[Path, Path]], transform: Callable[[Sample], Sample] | None
    ) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Sample:
        image_path, mask_path = self.paths[index]
        image = read_raster(image_path).astype(np.float32)
        mask = read_raster(mask_path).astype(np.int64)
        if np.any(mask > 8):
            raise ValueError(f"OpenEarthMap-SAR: native labels outside 0..8 in {mask_path}")
        mask = np.where(mask == 0, 255, mask - 1)
        sample = {"image": torch.from_numpy(image[None]), "mask": torch.from_numpy(mask)}
        return self.transform(sample) if self.transform is not None else sample


class OpenEarthMapSAR(BenchDataset):
    """Eight-class SAR benchmark with native zero ignored in both loss and metrics."""

    name = "openearthmap-sar"
    task = "segmentation"
    num_classes = 8
    multilabel = False
    supports_partitions = False
    rgb_bands: ClassVar[list[str]] = ["sar"]
    split_sizes: ClassVar[dict[str, int]] = {split: count for split, (_, count) in SPLITS.items()}
    bands: ClassVar[list[BandSpec]] = [
        BandSpec(
            "sar", "sar", "sar", mean=49.4892, std=34.9148, min=0, max=255, wavelength_um=None
        ),
    ]

    @classmethod
    def data_root(cls) -> Path:
        """Return the fixed SAR-only installation root."""
        return Path("data") / "openearthmap-sar"

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Load raw SAR bytes as float32 and remap native labels to eight classes."""
        if partition != "default":
            raise ValueError("OpenEarthMap-SAR supports only the default partition")
        specs = self.select_band_specs(bands)
        if specs != self.bands:
            raise ValueError("OpenEarthMap-SAR requires exactly the single sar band")
        transform = select_bands([0], 1, transform)
        return SARSegmentationDataset(split_paths(self.data_root(), split), transform)
