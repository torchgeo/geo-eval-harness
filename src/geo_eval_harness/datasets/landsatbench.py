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

LC100_MISSING_PID = "0669864_-114.00_26.25"


def _landsat8_bands(
    means: tuple[float, ...],
    stds: tuple[float, ...],
    mins: tuple[float, ...],
    maxs: tuple[float, ...],
) -> list[BandSpec]:
    return [
        BandSpec(
            "landsat",
            name,
            source_name,
            mean=mean,
            std=std,
            min=lo,
            max=hi,
            wavelength_um=wavelength_um,
        )
        for (name, source_name, wavelength_um), mean, std, lo, hi in zip(
            _LANDSAT8_BAND_META, means, stds, mins, maxs, strict=True
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
        label = torch.tensor(self.labels[index], dtype=torch.long)
        sample = {"image": image, "label": label}
        return sample if self.transform is None else self.transform(sample)


_BEN_CLASS2IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(TGBigEarthNet.class_sets[43])
}


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


class EuroSATL(BenchDataset):
    """Landsat-8 land-use classification (10 classes), from Landsat-Bench.

    27,000 Landsat 8 Collection 2 Level-2 chips corresponding to EuroSAT locations
    (16,200 / 5,400 / 5,400 train/val/test). Exposes the 7 OLI surface reflectance
    bands (SR_B1..SR_B7, excluding thermal ST_B10), clipped at 0 and resized to 22x22.
    Source: https://huggingface.co/datasets/isaaccorley/eurosat-l.
    """

    name = "eurosat-l"
    task = "classification"
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
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 16200, "val": 5400, "test": 5400}
    supports_partitions = False
    tile_size = 22

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(
            8515.3212890625,
            8829.6005859375,
            9795.9208984375,
            9891.6552734375,
            15499.8837890625,
            13146.115234375,
            11004.2119140625,
        ),
        stds=(
            3938.755859375,
            4057.7177734375,
            4205.96044921875,
            4434.4228515625,
            6150.6396484375,
            4923.02001953125,
            4044.071044921875,
        ),
        mins=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        maxs=(62584.0, 63593.0, 65454.0, 65454.0, 64960.0, 65454.0, 65454.0),
    )

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/eurosat-l")``."""
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
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        root = self.data_root()
        split_file = root / f"eurosat-{split}.txt"
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
        specs = self.select_band_specs(bands)
        band_indices = [self.bands.index(spec) + 1 for spec in specs]
        return _LandsatClassificationSplit(paths, labels, band_indices, self.tile_size, transform)


class BigEarthNetL(BenchDataset):
    """Landsat-8 multi-label land-cover classification (19 classes), from Landsat-Bench.

    519,284 Landsat 8 Collection 2 Level-2 chips corresponding to BigEarthNet v1
    (269,695 / 123,723 / 125,866 train/val/test). Exposes the 7 OLI surface reflectance
    bands (SR_B1..SR_B7), clipped at 0 and resized to 40x40.
    Source: https://huggingface.co/datasets/isaaccorley/bigearthnet-l.
    """

    name = "bigearthnet-l"
    task = "classification"
    num_classes = 19
    class_names: ClassVar[tuple[str, ...]] = (
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    )
    multilabel = True
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 269695, "val": 123723, "test": 125866}
    supports_partitions = False
    tile_size = 40

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(
            9405.1943359375,
            9649.6767578125,
            10425.685546875,
            10444.5888671875,
            16067.626953125,
            12699.18359375,
            10596.474609375,
        ),
        stds=(
            6016.34375,
            6095.47216796875,
            5839.4619140625,
            6068.49267578125,
            6271.4404296875,
            4256.0458984375,
            3130.7763671875,
        ),
        mins=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        maxs=(
            63979.91796875,
            65330.30859375,
            65454.0,
            65454.0,
            65454.0,
            65454.00390625,
            65454.00390625,
        ),
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
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
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
        specs = self.select_band_specs(bands)
        band_indices = [self.bands.index(spec) + 1 for spec in specs]
        return _BigEarthNetLSplit(image_paths, label_paths, band_indices, self.tile_size, transform)


class LC100L(BenchDataset):
    """Landsat-8 global multi-label land-cover classification (23 classes), from Landsat-Bench.

    8,634 Landsat 8 Collection 2 Level-2 scenes paired with Copernicus-Bench LC100 labels
    (5,181 / 1,727 / 1,726 train/val/test, excluding one test PID without Landsat coverage).
    Exposes the 7 OLI surface reflectance bands (SR_B1..SR_B7), clipped at 0 and resized
    to 224x224. Source: https://huggingface.co/datasets/isaaccorley/lc100-l.
    """

    name = "lc100-l"
    task = "classification"
    num_classes = 23
    class_names: ClassVar[tuple[str, ...]] = (
        "Unknown",
        "Shrubs",
        "Herbaceous vegetation",
        "Cultivated and managed vegetation / agriculture",
        "Urban / built up",
        "Bare / sparse vegetation",
        "Snow and ice",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Moss and lichen",
        "Closed forest, evergreen needle leaf",
        "Closed forest, evergreen broad leaf",
        "Closed forest, deciduous needle leaf",
        "Closed forest, deciduous broad leaf",
        "Closed forest, mixed",
        "Closed forest, not matching any of the other definitions",
        "Open forest, evergreen needle leaf",
        "Open forest, evergreen broad leaf",
        "Open forest, deciduous needle leaf",
        "Open forest, deciduous broad leaf",
        "Open forest, mixed",
        "Open forest, not matching any of the other definitions",
        "Oceans, seas",
    )
    multilabel = True
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 5181, "val": 1727, "test": 1726}
    supports_partitions = False
    tile_size = 224

    bands: ClassVar[list[BandSpec]] = _landsat8_bands(
        means=(
            10349.7880859375,
            10614.703125,
            11293.564453125,
            11854.380859375,
            14793.0302734375,
            12222.5654296875,
            10831.328125,
        ),
        stds=(
            10919.978515625,
            10937.5087890625,
            10435.6298828125,
            10778.1865234375,
            10573.8857421875,
            8040.63525390625,
            7096.6103515625,
        ),
        mins=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
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
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
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
        specs = self.select_band_specs(bands)
        band_indices = [self.bands.index(spec) + 1 for spec in specs]
        return _LandsatClassificationSplit(paths, labels, band_indices, self.tile_size, transform)
