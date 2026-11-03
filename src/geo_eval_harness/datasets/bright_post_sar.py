# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""BRIGHT post-event SAR building damage segmentation."""

from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets.utils import Sample

from ._transforms import select_bands
from .base import BandSpec, BenchDataset


class BRIGHTSegmentationDataset(Dataset):
    """Load post-event SAR tiles and four-class damage masks for one split."""

    def __init__(
        self, paths: list[tuple[Path, Path]], transform: Callable[[Sample], Sample] | None
    ) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Sample:
        image_path, mask_path = self.paths[index]
        image = BRIGHTPostSAR.read_raster(image_path).astype(np.float32)
        mask = BRIGHTPostSAR.read_raster(mask_path).astype(np.int64)
        if np.any(mask >= BRIGHTPostSAR.num_classes):
            raise ValueError(f"BRIGHT: damage labels outside 0..3 in {mask_path}")
        sample = {"image": torch.from_numpy(image[None]), "mask": torch.from_numpy(mask)}
        return self.transform(sample) if self.transform is not None else sample


class BRIGHTPostSAR(BenchDataset):
    """Building damage segmentation from very-high-resolution post-event SAR (BRIGHT).

    BRIGHT pairs pre-event optical and post-event SAR tiles over 14 disasters. This benchmark
    uses only the post-event SAR (Capella and Umbra X-band, single-polarisation, 0.3-0.6 m GSD,
    1024 x 1024 px) and the authors' ``standard_ML`` split (2,918 / 450 / 878 tiles; every event
    appears in every split). Images are 8-bit display-scaled amplitude, read as raw float32 in
    0..255; zero is the swath fill value but is kept, because labelled buildings overlap it.
    Masks keep the native classes: 0 background, 1 intact, 2 damaged, 3 destroyed. No pixel is
    ignored.

    Source: Chen et al., "BRIGHT", ESSD 17, 2025 (https://github.com/ChenHongruixuan/BRIGHT),
    files ``post-event.zip`` and ``target.zip`` from the ``Kullervo/BRIGHT`` Hugging Face
    dataset. SAR is CC-BY-4.0; labels follow the per-event optical licence (mostly CC-BY-NC-4.0).
    """

    name = "bright-post-sar"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    supports_partitions = False
    rgb_bands: ClassVar[list[str]] = ["sar"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 2918, "val": 450, "test": 878}
    raster_size: ClassVar[int] = 1024
    # Train split, zeros included:
    # python scripts/compute_band_statistics.py --dataset bright-post-sar
    bands: ClassVar[list[BandSpec]] = [
        BandSpec(
            "sar", "sar", "sar", mean=56.8231, std=45.8553, min=0, max=255, wavelength_um=None
        ),
    ]

    @classmethod
    def data_root(cls) -> Path:
        """Return the fixed installation root."""
        return Path("data") / "bright-post-sar"

    @classmethod
    def split_ids(cls, split: str) -> list[str]:
        """Return the packaged, sorted tile ids (``<event>_<8 digits>``) of one split."""
        if split not in cls.split_sizes:
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        splits = files("geo_eval_harness").joinpath("datasets", "splits", "bright-post-sar")
        return splits.joinpath(f"{split}.txt").read_text().split()

    @classmethod
    def split_paths(cls, root: Path, split: str) -> list[tuple[Path, Path]]:
        """Return the split's (SAR, mask) pairs, failing if any file is missing."""
        pairs = [
            (
                root / "post-event" / f"{tile}_post_disaster.tif",
                root / "target" / f"{tile}_building_damage.tif",
            )
            for tile in cls.split_ids(split)
        ]
        missing = [path for pair in pairs for path in pair if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"BRIGHT: {len(missing)} {split} files missing (e.g. {missing[0]}). "
                "Run: geo-eval-harness download bright-post-sar"
            )
        return pairs

    @classmethod
    def read_raster(cls, path: Path) -> np.ndarray:
        """Read a single-band uint8 tile at the dataset's native resolution."""
        size = cls.raster_size
        with rasterio.open(path) as raster:
            if raster.count != 1 or raster.dtypes != ("uint8",) or raster.shape != (size, size):
                raise ValueError(
                    f"BRIGHT: {path} must be single-band uint8 {size} x {size}; "
                    f"got {raster.count} bands, {raster.dtypes}, {raster.shape}"
                )
            return raster.read(1)

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Load raw post-event SAR bytes as float32 with native damage labels."""
        if partition != "default":
            raise ValueError("BRIGHT supports only the default partition")
        specs = self.select_band_specs(bands)
        if specs != self.bands:
            raise ValueError("BRIGHT requires exactly the single sar band")
        transform = select_bands([0], 1, transform)
        return BRIGHTSegmentationDataset(self.split_paths(self.data_root(), split), transform)
