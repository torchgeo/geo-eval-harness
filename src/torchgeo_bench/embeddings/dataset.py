"""Load a stored embedding collection as a PyTorch dataset for the standard probes."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torchgeo_bench.bands import BandSpec

from . import EMBEDDINGS_ROOT, collection_dir

__all__ = ["EmbeddingDataset", "embedding_bands", "embedding_loaders"]


def embedding_bands(channels: int) -> list[BandSpec]:
    """Describe embedding channels as bands; models do not normalize them."""
    return [
        BandSpec("embedding", f"e{i:03d}", f"e{i:03d}", mean=0.0, std=1.0, min=-1.0, max=1.0)
        for i in range(channels)
    ]


class EmbeddingDataset(Dataset):
    """One split of a stored collection.

    Samples hold ``image``, the ``(C, H, W)`` float32 embeddings with NaN for
    missing cells, and the ``label`` or ``mask`` from the dataset wrapper.
    """

    def __init__(self, directory: Path, split: str) -> None:
        self.files = sorted((directory / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No {split} embeddings in {directory}")
        self.metadata = json.loads((directory / "metadata.json").read_text())

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.files[index]) as sample:
            item = {"image": torch.from_numpy(sample["embedding"].astype(np.float32))}
            for key in ("label", "mask"):
                if key in sample:
                    item[key] = torch.from_numpy(sample[key])
        return item


def embedding_loaders(
    product: str,
    dataset: str,
    *,
    batch_size: int,
    num_workers: int,
    root: Path = EMBEDDINGS_ROOT,
) -> tuple[EmbeddingDataset, DataLoader, DataLoader, DataLoader]:
    """Return ``(train_dataset, train_loader, val_loader, test_loader)`` like ``get_datasets``."""
    directory = collection_dir(product, dataset, root)
    if not (directory / "metadata.json").exists():
        raise FileNotFoundError(
            f"No {product!r} embeddings for {dataset!r} in {directory}. Run "
            f"`torchgeo-bench embeddings download --product {product} --dataset {dataset}`."
        )
    splits = [EmbeddingDataset(directory, split) for split in ("train", "val", "test")]
    loaders = [
        DataLoader(
            split,
            batch_size=batch_size,
            shuffle=index == 0,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for index, split in enumerate(splits)
    ]
    return splits[0], loaders[0], loaders[1], loaders[2]
