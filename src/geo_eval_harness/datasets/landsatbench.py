"""Landsat-Bench classification datasets: EuroSAT-L and BigEarthNet-L.

Landsat 8 Collection 2 Level-2 surface-reflectance variants (7 OLI bands, SR_B1..SR_B7)
from Corley et al. (2025), https://github.com/isaaccorley/landsatbench.
"""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchgeo.datasets import BigEarthNet as TGBigEarthNet

from .base import BandSpec, BenchDataset

_LANDSAT8_BAND_META: tuple[tuple[str, str, float], ...] = (
    ("coastal_aerosol", "SR_B1", 0.443),
    ("blue", "SR_B2", 0.482),
    ("green", "SR_B3", 0.562),
    ("red", "SR_B4", 0.655),
    ("nir", "SR_B5", 0.865),
    ("swir_1", "SR_B6", 1.609),
    ("swir_2", "SR_B7", 2.201),
)


def _landsat8_bands(
    means: tuple[float, ...],
    stds: tuple[float, ...],
    maxs: tuple[float, ...],
) -> list[BandSpec]:
    return [
        BandSpec(
            "landsat",
            name,
            source_name,
            mean=mean,
            std=std,
            min=0.0,
            max=hi,
            wavelength_um=wavelength_um,
        )
        for (name, source_name, wavelength_um), mean, std, hi in zip(
            _LANDSAT8_BAND_META, means, stds, maxs, strict=True
        )
    ]


class _LandsatSplit(Dataset):
    """GeoTIFF classification split backed by ``index.parquet``."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int] | list[np.ndarray],
        band_indices: list[int],
        tile_size: int,
        transform: Callable | None,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.band_indices = band_indices
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with rasterio.open(self.paths[index]) as src:
            array = src.read(self.band_indices).astype(np.float32)
        image = torch.from_numpy(array).clamp(min=0.0)
        if image.shape[-2:] != (self.tile_size, self.tile_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.tile_size, self.tile_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        sample = {"image": image, "label": torch.tensor(self.labels[index], dtype=torch.long)}
        return sample if self.transform is None else self.transform(sample)


class _LandsatBenchDataset(BenchDataset):
    """Shared loader for Landsat-Bench datasets stored as GeoTIFFs + ``index.parquet``."""

    task = "classification"
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    supports_partitions = False
    tile_size: ClassVar[int]
    split_column: ClassVar[str] = "split"
    _data_dir: ClassVar[str]

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/<_data_dir>")``."""
        return Path("data") / cls._data_dir

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the split as a PyTorch dataset."""
        del partition
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        root = self.data_root()
        index_path = root / "index.parquet"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"{self.name} not found under {root}. "
                f"Run `geo-eval-harness download {self._data_dir}` first."
            )
        df = pd.read_parquet(index_path, columns=["path", self.split_column, "label"])
        split_df = df[df[self.split_column] == split]
        paths = [root / rel for rel in split_df["path"]]
        labels = split_df["label"].tolist()
        band_indices = [self.bands.index(spec) + 1 for spec in self.select_band_specs(bands)]
        return _LandsatSplit(paths, labels, band_indices, self.tile_size, transform)


class EuroSATL(_LandsatBenchDataset):
    """Landsat-8 land-use classification (10 classes), from Landsat-Bench.

    27,000 Landsat 8 Collection 2 Level-2 chips corresponding to EuroSAT locations
    (16,200 / 5,400 / 5,400 train/val/test), 7 OLI SR bands resized to 22x22.
    Source: https://huggingface.co/datasets/isaaccorley/eurosat-l.
    """

    name = "eurosat-l"
    _data_dir = "eurosat-l"
    num_classes = 10
    class_names: ClassVar[tuple[str, ...]] = (
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    )
    multilabel = False
    split_sizes: ClassVar[dict[str, int]] = {"train": 16200, "val": 5400, "test": 5400}
    tile_size = 22

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(8515.321, 8829.601, 9795.921, 9891.655, 15499.884, 13146.115, 11004.212),
        stds=(3938.756, 4057.718, 4205.960, 4434.423, 6150.640, 4923.020, 4044.071),
        maxs=(62584.0, 63593.0, 65454.0, 65454.0, 64960.0, 65454.0, 65454.0),
    )


class EuroSATLSpatial(EuroSATL):
    """EuroSAT-L with EuroSAT's longitude-based 60/20/20 train/val/test spatial splits."""

    name = "eurosat-l-spatial"
    split_column = "spatial_split"


class BigEarthNetL(_LandsatBenchDataset):
    """Landsat-8 multi-label land-cover classification (19 classes), from Landsat-Bench.

    519,284 Landsat 8 Collection 2 Level-2 chips corresponding to BigEarthNet v1
    (269,695 / 123,723 / 125,866 train/val/test), 7 OLI SR bands resized to 40x40.
    Source: https://huggingface.co/datasets/isaaccorley/bigearthnet-l.
    """

    name = "bigearthnet-l"
    _data_dir = "bigearthnet-l"
    num_classes = 19
    class_names: ClassVar[tuple[str, ...]] = tuple(TGBigEarthNet.class_sets[19])
    multilabel = True
    split_sizes: ClassVar[dict[str, int]] = {"train": 269695, "val": 123723, "test": 125866}
    tile_size = 40

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(9405.194, 9649.677, 10425.686, 10444.589, 16067.627, 12699.184, 10596.475),
        stds=(6016.344, 6095.472, 5839.462, 6068.493, 6271.440, 4256.046, 3130.776),
        maxs=(63979.918, 65330.309, 65454.0, 65454.0, 65454.0, 65454.004, 65454.004),
    )
