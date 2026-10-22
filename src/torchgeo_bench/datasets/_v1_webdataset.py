"""Read GeoBench V1 from indexed WebDataset tar shards.

Use indexed tar shards to avoid per-sample HDF5 opens over NFS.

:func:`download_sharded_root` fetches the pinned JSON mirror and verifies each archive.

Shards hold ``<sid>.bands.npz`` and ``<sid>.meta.json`` pairs for roughly 1000 samples.

Index byte offsets once; each worker opens its own file descriptors for fork-safe reads.

Output matches :class:`~torchgeo_bench.datasets.geobench_v1.GeoBenchv1`.
"""

import io
import json
import logging
import os
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from ..hub import fetch_pinned, load_checksums
from ._metadata import decode_metadata

logger = logging.getLogger(__name__)

V1_HF_REPO_ID = "calebrob6/geobenchv1-webdataset"
V1_HF_REVISION = "18c293d3a963c73e8e055a2fef6fca9e029c6e95"


def download_sharded_root(
    sharded_root: Path,
    datasets: list[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Download and verify the selected pickle-free V1 datasets.

    Args:
        sharded_root: Destination collection directory.
        datasets: Dataset names, or ``None`` for the full classification suite.
        cache_dir: Optional Hugging Face download cache.
    """
    checksums = load_checksums("geobench_v1.sha256")
    available = {name.split("/", 1)[0] for name in checksums}
    names = sorted(available) if datasets is None else list(dict.fromkeys(datasets))
    if not names:
        raise ValueError("datasets must contain at least one GeoBench V1 dataset name")
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"Unknown GeoBench V1 dataset(s): {', '.join(unknown)}.")
    sharded_root = Path(sharded_root)
    sharded_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GeoBench V1 from %s -> %s", V1_HF_REPO_ID, sharded_root)
    fetch_pinned(
        V1_HF_REPO_ID,
        V1_HF_REVISION,
        sharded_root,
        {name: digest for name, digest in checksums.items() if name.split("/", 1)[0] in names},
        allow_patterns=[f"{name}/*" for name in names],
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
    )
    logger.info("GeoBench V1 download complete.")


class GeoBenchv1Sharded(Dataset):
    """GeoBench V1 dataset reading from WebDataset tar shards."""

    def __init__(  # noqa: PLR0913 - public dataset constructor.
        self,
        root: str | Path,
        dataset_name: str,
        split: Literal["train", "valid", "test"],
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(root) / dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Sharded dataset dir not found: {self.dataset_dir}. "
                f"Run `torchgeo-bench download geobench_v1 --datasets {dataset_name}`."
            )

        partition_file = self.dataset_dir / f"{partition}_partition.json"
        with open(partition_file) as f:
            partition_data = json.load(f)
        if split not in partition_data:
            raise ValueError(
                f"Split '{split}' not found in partition. Available: {list(partition_data.keys())}"
            )
        self.sample_ids: list[str] = partition_data[split]
        self.transform = transform

        shard_paths = sorted(self.dataset_dir.glob("shard_*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.tar in {self.dataset_dir}")
        # Sample IDs can contain dots; strip the suffix instead of splitting on the first dot.
        self._index: dict[str, dict[str, tuple[Path, int, int]]] = {}
        for path in shard_paths:
            with tarfile.open(path, "r") as t:
                for m in t.getmembers():
                    for ext in ("bands.npz", "meta.json"):
                        suffix = "." + ext
                        if m.name.endswith(suffix):
                            base = m.name[: -len(suffix)]
                            self._index.setdefault(base, {})[ext] = (
                                path,
                                m.offset_data,
                                m.size,
                            )
                            break

        if bands is None:
            sample_meta = self._load_meta(self.sample_ids[0])
            self.band_names: list[str] = list(sample_meta["bands_order"])
        else:
            self.band_names = list(bands)

    def _read(self, ref: tuple[Path, int, int]) -> bytes:
        path, offset, size = ref
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    def _load_meta(self, sample_id: str) -> dict:
        parts = self._index[sample_id]
        if "meta.json" not in parts:
            raise ValueError(
                f"Sample {sample_id!r} requires '.meta.json' metadata. "
                "Legacy pickle shards are not supported. Replace this cache with "
                f"'torchgeo-bench download geobench_v1 --datasets {self.dataset_dir.name}'."
            )
        return decode_metadata(self._read(parts["meta.json"]))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sid = self.sample_ids[index]
        parts = self._index[sid]
        meta = self._load_meta(sid)
        with np.load(io.BytesIO(self._read(parts["bands.npz"])), allow_pickle=False) as archive:
            bands_dict = {name: archive[name] for name in archive.files}

        bands_data = []
        available = list(bands_dict)
        for band_name in self.band_names:
            if band_name in bands_dict:
                bands_data.append(bands_dict[band_name])
                continue
            matching = [k for k in available if k.startswith(band_name)]
            if not matching:
                raise KeyError(
                    f"Band '{band_name}' not found in shard sample {sid}. "
                    f"Available: {available[:5]}..."
                )
            bands_data.append(bands_dict[matching[0]])

        image = np.stack(bands_data, axis=0).astype(np.float32)
        image_t = torch.from_numpy(image)

        label = meta["label"]
        label_arr = np.asarray(label)
        if label_arr.ndim > 0:
            label_t: torch.Tensor = torch.from_numpy(label_arr.astype(np.float32))
        else:
            label_t = torch.tensor(label_arr.item(), dtype=torch.long)

        sample: dict = {"image": image_t, "label": label_t, "sample_id": sid}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
