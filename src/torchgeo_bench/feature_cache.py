"""Disk cache for frozen-backbone segmentation features.

Several probes often reuse one encoder's features for the same dataset split: different heads,
seeds, or fitting recipes. Caching the extracted features on disk lets those jobs skip
re-encoding the images, which is CPU-bound under small container CPU quotas.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from torchgeo_bench.segmentation_probe import CachedFeaturesDataset

logger = logging.getLogger(__name__)

# Bump when extraction changes in a way the key cannot see, so stale files are no longer found.
FEATURE_CACHE_VERSION = 1


def weights_digest(module: torch.nn.Module) -> str:
    """Return a SHA-256 over every parameter and buffer, in name order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        values = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}:{tuple(values.shape)}:{values.dtype}".encode())
        digest.update(memoryview(values.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def feature_cache_key(identity: dict[str, object]) -> str:
    """Return a stable file key for everything that determines one split's features."""
    payload = {"version": FEATURE_CACHE_VERSION, **identity}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


# Masks are stored as int16; every benchmark label (including the 255 ignore index) fits.
MASK_DTYPE = np.int16


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stream_features(
    directory: Path,
    batches: Iterable[tuple[list[torch.Tensor], torch.Tensor]],
    num_samples: int,
) -> None:
    """Write feature batches straight into memory-mapped ``.npy`` files under ``directory``.

    Each layer goes to ``layer<i>.npy`` and the masks to ``masks.npy``, preallocated once the first
    batch gives their shapes, so host memory stays at about one batch whatever the split size.
    Files are written in a temporary sibling directory that is renamed into place only after
    everything is flushed, so readers see either no directory or the complete one.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}.", suffix=".tmp")
    )
    try:
        layers: list[np.memmap] = []
        masks: np.memmap | None = None
        offset = 0
        for features, labels in batches:
            if masks is None:
                layers = [
                    np.lib.format.open_memmap(
                        staged / f"layer{i}.npy",
                        mode="w+",
                        dtype=torch.empty((), dtype=f.dtype).numpy().dtype,
                        shape=(num_samples, *f.shape[1:]),
                    )
                    for i, f in enumerate(features)
                ]
                masks = np.lib.format.open_memmap(
                    staged / "masks.npy",
                    mode="w+",
                    dtype=MASK_DTYPE,
                    shape=(num_samples, *labels.shape[1:]),
                )
            end = offset + len(labels)
            if end > num_samples:
                raise ValueError(f"Got more than the expected {num_samples} samples")
            info = np.iinfo(MASK_DTYPE)
            if labels.numel() and not (
                info.min <= int(labels.min()) and int(labels.max()) <= info.max
            ):
                raise ValueError(f"Mask values do not fit {MASK_DTYPE.__name__}")
            for layer, f in zip(layers, features, strict=True):
                layer[offset:end] = f.numpy()
            masks[offset:end] = labels.numpy()
            offset = end
        if masks is None or offset != num_samples:
            raise ValueError(f"Expected {num_samples} samples, got {offset}")
        for array in [*layers, masks]:
            array.flush()
        del layers, masks
        for file in staged.iterdir():
            _fsync_path(file)
        _fsync_path(staged)
        os.rename(staged, directory)
        _fsync_path(directory.parent)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def load_features(path: Path) -> CachedFeaturesDataset:
    """Memory-map cached features: a streamed ``.npy`` directory or a legacy ``.pt`` file.

    Masks keep their stored integer dtype; batches are converted to int64 as they are used.
    """
    if path.is_dir():
        files = sorted(path.glob("layer*.npy"), key=lambda p: int(p.stem.removeprefix("layer")))
        # Copy-on-write maps give writable arrays without ever writing to the cache files.
        layers = [torch.from_numpy(np.load(f, mmap_mode="c")) for f in files]
        masks = torch.from_numpy(np.load(path / "masks.npy", mmap_mode="c"))
        return CachedFeaturesDataset(layers, masks)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return CachedFeaturesDataset(payload["layer_tensors"], payload["masks"])


def cached_features(
    path: Path,
    batches: Callable[[], Iterable[tuple[list[torch.Tensor], torch.Tensor]]],
    num_samples: int,
) -> CachedFeaturesDataset:
    """Load one split's cached features, streaming them to disk first if they are missing.

    ``path`` names the legacy ``.pt`` file, which is still read when present; new caches go to
    the directory of the same name without the suffix. A lock beside the file makes concurrent
    jobs with the same key extract only once.
    """
    directory = path.with_suffix("")
    with FileLock(f"{path}.lock"):
        for existing in (path, directory):
            if existing.exists():
                logger.info("Loading cached features from %s", existing)
                return load_features(existing)
        stream_features(directory, batches(), num_samples)
        logger.info("Saved features to %s", directory)
        return load_features(directory)
