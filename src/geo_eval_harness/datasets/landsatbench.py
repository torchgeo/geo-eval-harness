"""Landsat-Bench classification datasets: EuroSAT-L, BigEarthNet-L, and LC100-L.

Landsat 8 Collection 2 Level-2 surface-reflectance variants (7 OLI bands, SR_B1..SR_B7)
from Corley et al. (2025), https://github.com/isaaccorley/landsatbench.
"""

import json
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
from torchgeo.datasets import CopernicusBenchLC100ClsS3

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

_BEN_CLASS2IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(TGBigEarthNet.class_sets[43])
}

LC100_MISSING_PID = "0669864_-114.00_26.25"


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


def _read_landsat_image(path: Path, band_indices: list[int], tile_size: int) -> torch.Tensor:
    """Read 1-based GeoTIFF bands as float32, clamp negatives to 0, and resize to square."""
    with rasterio.open(path) as src:
        array = src.read(band_indices).astype(np.float32)
    image = torch.from_numpy(array).clamp(min=0.0)
    if image.shape[-2:] != (tile_size, tile_size):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image


class _LandsatClassificationSplit(Dataset):
    """GeoTIFF classification split with per-sample band selection and square resize."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int] | np.ndarray,
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
        image = _read_landsat_image(self.paths[index], self.band_indices, self.tile_size)
        sample = {"image": image, "label": torch.tensor(self.labels[index], dtype=torch.long)}
        return sample if self.transform is None else self.transform(sample)


class _BigEarthNetLSplit(Dataset):
    """BigEarthNet-L split reading 7-band GeoTIFFs and 19-class JSON label metadata."""

    def __init__(
        self,
        image_paths: list[Path],
        label_paths: list[Path],
        band_indices: list[int],
        tile_size: int,
        transform: Callable | None,
    ) -> None:
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.band_indices = band_indices
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = _read_landsat_image(self.image_paths[index], self.band_indices, self.tile_size)
        raw_labels = json.loads(self.label_paths[index].read_text(encoding="utf-8"))["labels"]
        target = torch.zeros(19, dtype=torch.long)
        for name in raw_labels:
            mapped = TGBigEarthNet.label_converter.get(_BEN_CLASS2IDX[name])
            if mapped is not None:
                target[mapped] = 1
        sample = {"image": image, "label": target}
        return sample if self.transform is None else self.transform(sample)


class _LandsatBenchDataset(BenchDataset):
    """Shared defaults and helpers for Landsat-Bench classification datasets."""

    task = "classification"
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    supports_partitions = False
    tile_size: ClassVar[int]

    @classmethod
    def _band_indices(cls, bands: tuple[str, ...] | None) -> list[int]:
        return [cls.bands.index(spec) + 1 for spec in cls.select_band_specs(bands)]

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")


class EuroSATL(_LandsatBenchDataset):
    """Landsat-8 land-use classification (10 classes), from Landsat-Bench.

    27,000 Landsat 8 Collection 2 Level-2 chips corresponding to EuroSAT locations
    (16,200 / 5,400 / 5,400 train/val/test). Exposes the 7 OLI surface reflectance
    bands (SR_B1..SR_B7, excluding thermal ST_B10), clipped at 0 and resized to 22x22.
    Source: https://huggingface.co/datasets/isaaccorley/eurosat-l.
    """

    name = "eurosat-l"
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
    split_prefix: ClassVar[str] = "eurosat"

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(8515.321, 8829.601, 9795.921, 9891.655, 15499.884, 13146.115, 11004.212),
        stds=(3938.756, 4057.718, 4205.960, 4434.423, 6150.640, 4923.020, 4044.071),
        maxs=(62584.0, 63593.0, 65454.0, 65454.0, 64960.0, 65454.0, 65454.0),
    )

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/eurosat-l")`` (shared by :class:`EuroSATLSpatial`)."""
        return Path("data/eurosat-l")

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
        self._check_split(split)
        root = self.data_root()
        split_file = root / f"{self.split_prefix}-{split}.txt"
        image_root = root / "eurosat-l"
        if not split_file.is_file() or not image_root.is_dir():
            raise FileNotFoundError(
                f"EuroSAT-L not found under {root}. "
                "Run `geo-eval-harness download eurosat-l` first."
            )
        valid_names = {
            line.strip().replace(".jpg", ".tif")
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        paths: list[Path] = []
        labels: list[int] = []
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = image_root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.name in valid_names:
                    paths.append(path)
                    labels.append(class_idx)
        return _LandsatClassificationSplit(
            paths, labels, self._band_indices(bands), self.tile_size, transform
        )


class EuroSATLSpatial(EuroSATL):
    """EuroSAT-L with EuroSAT's longitude-based 60/20/20 train/val/test spatial splits.

    Shares the same 27,000 Landsat-8 chips, classes, bands, and statistics as :class:`EuroSATL`.
    """

    name = "eurosat-l-spatial"
    split_prefix = "eurosat-spatial"


class BigEarthNetL(_LandsatBenchDataset):
    """Landsat-8 multi-label land-cover classification (19 classes), from Landsat-Bench.

    519,284 Landsat 8 Collection 2 Level-2 chips corresponding to BigEarthNet v1
    (269,695 / 123,723 / 125,866 train/val/test). Exposes the 7 OLI surface reflectance
    bands (SR_B1..SR_B7), clipped at 0 and resized to 40x40.
    Source: https://huggingface.co/datasets/isaaccorley/bigearthnet-l.
    """

    name = "bigearthnet-l"
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

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/bigearthnet-l")``."""
        return Path("data/bigearthnet-l")

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
        self._check_split(split)
        root = self.data_root()
        split_file = root / f"bigearthnet-{split}.csv"
        image_dir = root / "bigearthnet-l" / "images"
        label_dir = root / "bigearthnet-l" / "labels"
        if not split_file.is_file() or not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"BigEarthNet-L not found under {root}. "
                "Run `geo-eval-harness download bigearthnet-l` first."
            )
        patch_ids = [
            line.split(",", 1)[0].strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        image_paths = [image_dir / f"{patch_id}.tif" for patch_id in patch_ids]
        label_paths = [label_dir / f"{patch_id}_labels_metadata.json" for patch_id in patch_ids]
        return _BigEarthNetLSplit(
            image_paths, label_paths, self._band_indices(bands), self.tile_size, transform
        )


class LC100L(_LandsatBenchDataset):
    """Landsat-8 global multi-label land-cover classification (23 classes), from Landsat-Bench.

    8,634 Landsat 8 Collection 2 Level-2 scenes paired with Copernicus-Bench LC100 labels
    (5,181 / 1,727 / 1,726 train/val/test, excluding one test PID without Landsat coverage).
    Exposes the 7 OLI surface reflectance bands (SR_B1..SR_B7), clipped at 0 and resized
    to 224x224. Source: https://huggingface.co/datasets/isaaccorley/lc100-l.
    """

    name = "lc100-l"
    num_classes = 23
    class_names: ClassVar[tuple[str, ...]] = tuple(CopernicusBenchLC100ClsS3.classes)
    multilabel = True
    split_sizes: ClassVar[dict[str, int]] = {"train": 5181, "val": 1727, "test": 1726}
    tile_size = 224

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(10349.788, 10614.703, 11293.564, 11854.381, 14793.030, 12222.565, 10831.328),
        stds=(10919.979, 10937.509, 10435.630, 10778.187, 10573.886, 8040.635, 7096.610),
        maxs=(65454.0, 65454.0, 65454.0, 65454.0, 65454.0, 65175.0, 61457.0),
    )

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/lc100-l")``."""
        return Path("data/lc100-l")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the static split as a PyTorch dataset, joining images and labels by PID."""
        del partition
        self._check_split(split)
        root = self.data_root()
        labels_file = root / f"multilabel-{split}.csv"
        static_file = root / f"static_fnames-{split}.csv"
        image_root = root / "lc100-l"
        if not labels_file.is_file() or not static_file.is_file() or not image_root.is_dir():
            raise FileNotFoundError(
                f"LC100-L not found under {root}. Run `geo-eval-harness download lc100-l` first."
            )
        labels_df = pd.read_csv(labels_file)
        labels_df = labels_df[labels_df["PID"] != LC100_MISSING_PID]
        static_df = pd.read_csv(static_file, header=None, names=["PID", "filename"])
        # static_fnames-test.csv is ordered differently from multilabel-test.csv; align by PID.
        fname_by_pid = dict(zip(static_df["PID"], static_df["filename"], strict=True))
        paths = [image_root / pid / fname_by_pid[pid] for pid in labels_df["PID"]]
        labels = labels_df.iloc[:, 1:].to_numpy(dtype=np.int64)
        return _LandsatClassificationSplit(
            paths, labels, self._band_indices(bands), self.tile_size, transform
        )
