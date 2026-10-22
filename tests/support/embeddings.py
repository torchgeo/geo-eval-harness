"""Toy footprints, products and datasets for offline embedding tests."""

from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from torchgeo_bench.datasets.base import BenchDataset, Footprint
from torchgeo_bench.embeddings.products import EmbeddingProduct

UTM = "EPSG:32632"
SIZE = 8


def utm_footprint(x0: float, y0: float = 5_500_000.0, *, year: int = 2020) -> Footprint:
    """A north-up 8 x 8 grid of 10 m cells with its top-left corner at ``(x0, y0)``."""
    return Footprint(UTM, (10.0, 0.0, x0, 0.0, -10.0, y0), (SIZE, SIZE), year)


class ToyProduct(EmbeddingProduct):
    """Channel 0 is the easting relative to the grid centre and channel 1 the absolute
    easting, so positions predict masks and labels. Cell ``(0, 0)`` is missing.
    """

    name = "aef"
    channels = 3
    years = range(2019, 2022)

    def __init__(self) -> None:
        self.reads: list[Footprint] = []

    def read(self, grid: Footprint) -> np.ndarray:
        self.reads.append(grid)
        height, width = grid.shape
        a, _, c = grid.transform[:3]
        x = a * (np.arange(width) + 0.5) + c
        out = np.empty((self.channels, height, width), dtype=np.float32)
        out[0] = (x - x.mean()) / 10
        out[1] = x / 100_000
        out[2] = grid.year - 2020
        out[:, 0, 0] = np.nan
        return out


class _Split(list):
    def __init__(self, split: str, samples: list[dict[str, torch.Tensor]]) -> None:
        super().__init__(samples)
        self.split = split


class ToyBench(BenchDataset):
    """Samples spread along the easting; ``label`` is the side of x = 500 km.

    With ``task="segmentation"`` the ``mask`` marks the eastern half of each sample.
    """

    name = "m-eurosat"
    task = "classification"
    num_classes = 10
    bands: ClassVar = []
    rgb_bands: ClassVar = []
    split_sizes: ClassVar = {"train": 12, "val": 6, "test": 6}

    def __init__(self, name: str = "m-eurosat", task: str = "classification") -> None:
        self.name = name
        self.task = task  # type: ignore[assignment]

    @classmethod
    def data_root(cls) -> Path:
        raise NotImplementedError

    def get_dataset(self, split: str, **_: object) -> _Split:
        samples = []
        for index in range(self.split_sizes[split]):
            sample = {"image": torch.zeros(1, SIZE, SIZE)}
            if self.task == "segmentation":
                sample["mask"] = torch.zeros(SIZE, SIZE, dtype=torch.long)
                sample["mask"][:, SIZE // 2 :] = 1
            else:
                sample["label"] = torch.tensor(index % 2)
            samples.append(sample)
        return _Split(split, samples)

    def footprint(self, dataset: object, index: int) -> Footprint:
        offset = {"train": 0, "val": 1, "test": 2}[dataset.split]  # type: ignore[attr-defined]
        x0 = 400_000.0 + (index % 2) * 200_000.0 + (3 * index + offset) * 1_000.0
        return utm_footprint(x0, year=2018 + index % 5)
