"""GeoBench V1 loaders for local HDF5 files and WebDataset shards.

HDF5 samples require JSON metadata and partition files, not the upstream ``geobench`` package.

Standard V1 downloads use JSON tar shards instead.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ._metadata import read_hdf5_metadata
from .base import BenchDataset

V1_ROOT = Path("data/classification_v1.0")
V1_SHARDED_ROOT = Path("data/classification_v1.0_wds")


class GeoBenchv1(Dataset):
    """PyTorch Dataset for GeoBench V1 classification benchmarks.

    Args:
        root: Path to the GeoBench V1 collection (e.g. ``data/classification_v1.0``).
        dataset_name: Dataset name (e.g. ``"m-eurosat"``).
        split: Split name (``"train"``, ``"valid"``, or ``"test"``).
        partition: Partition name (e.g. ``"default"``, ``"0.01x_train"``).
        bands: Tuple of source band names (``"04 - Red"``, etc.) to load. If
            ``None``, loads all bands present in the first sample.
        transform: Optional callable applied to each sample dict.
    """

    def __init__(  # noqa: PLR0913 - public dataset constructor.
        self,
        root: str | Path,
        dataset_name: str,
        split: Literal["train", "valid", "test"],
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: object = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.partition = partition
        self.transform = transform

        self.dataset_dir = self.root / dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}. "
                "Run `geo-eval-harness download geobench_v1`."
            )

        partition_file = self.dataset_dir / f"{partition}_partition.json"
        if not partition_file.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_file}")

        with open(partition_file) as f:
            partition_data = json.load(f)

        if split not in partition_data:
            raise ValueError(
                f"Split '{split}' not found in partition. Available: {list(partition_data.keys())}"
            )
        self.sample_ids = partition_data[split]

        if bands is None:
            sample_meta = self._load_sample_metadata(self.sample_ids[0])
            self.band_names: list[str] = list(sample_meta["bands_order"])
        else:
            self.band_names = list(bands)

    def _load_sample_metadata(self, sample_id: str) -> dict:
        """Load JSON metadata from HDF5 attributes."""
        sample_path = self.dataset_dir / f"{sample_id}.hdf5"
        with h5py.File(sample_path, "r") as f:
            return read_hdf5_metadata(f.attrs)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sample_id = self.sample_ids[index]
        sample_path = self.dataset_dir / f"{sample_id}.hdf5"

        metadata = self._load_sample_metadata(sample_id)
        label = metadata["label"]

        bands_data = []
        with h5py.File(sample_path, "r") as f:
            available_keys = list(f.keys())
            for band_name in self.band_names:
                if band_name in available_keys:
                    data = f[band_name][:]  # type: ignore[index]
                else:
                    # Temporal datasets (e.g. m-forestnet) suffix bands with dates.
                    matching = [k for k in available_keys if k.startswith(band_name)]
                    if not matching:
                        raise KeyError(
                            f"Band '{band_name}' not found in {sample_path}. "
                            f"Available: {available_keys[:5]}..."
                        )
                    data = f[matching[0]][:]  # type: ignore[index]
                bands_data.append(data)

        image = np.stack(bands_data, axis=0).astype(np.float32)

        image_t = torch.from_numpy(image)
        label_arr = np.asarray(label)
        label_t: torch.Tensor
        if label_arr.ndim > 0:
            label_t = torch.from_numpy(label_arr.astype(np.float32))
        else:
            label_t = torch.tensor(label_arr.item(), dtype=torch.long)

        sample: dict = {"image": image_t, "label": label_t, "sample_id": sample_id}
        if self.transform is not None:
            sample = self.transform(sample)  # type: ignore[misc]
        return sample


class _V1Dataset(BenchDataset):
    """Load GeoBench V1 splits for wrappers that declare dataset metadata."""

    supports_partitions = True

    @classmethod
    def data_root(cls) -> Path:
        return V1_ROOT

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a torch :class:`Dataset` for the split (raw values).

        Loaders are tried in this order:

        1. **Sharded WebDataset** at :data:`V1_SHARDED_ROOT` if shards exist locally.
        2. **Custom JSON-metadata HDF5** at :data:`V1_ROOT` if present.

        Missing data must be downloaded with ``geo-eval-harness download`` first.
        """
        v1_split: Literal["train", "valid", "test"] = "valid" if split == "val" else split  # type: ignore[assignment]
        source_bands = tuple(spec.source_name for spec in self.select_band_specs(bands))

        sharded_dir = V1_SHARDED_ROOT / self.name
        hdf5_dir = self.data_root() / self.name
        if sharded_dir.exists() and any(sharded_dir.glob("shard_*.tar")):
            from ._v1_webdataset import GeoBenchv1Sharded

            return GeoBenchv1Sharded(
                root=V1_SHARDED_ROOT,
                dataset_name=self.name,
                split=v1_split,
                partition=partition,
                bands=source_bands,
                transform=transform,
            )
        if not hdf5_dir.exists():
            raise FileNotFoundError(
                f"GeoBench V1 dataset '{self.name}' is not downloaded. "
                f"Run `geo-eval-harness download {self.name}` first."
            )
        return GeoBenchv1(
            root=self.data_root(),
            dataset_name=self.name,
            split=v1_split,
            partition=partition,
            bands=source_bands,
            transform=transform,
        )
