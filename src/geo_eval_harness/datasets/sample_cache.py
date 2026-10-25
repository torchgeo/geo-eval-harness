"""Exact in-memory or memory-mapped copies of preprocessed segmentation samples.

Segmentation training reads every sample once per epoch, and some datasets decode compressed,
full-size, all-band files each time. A sample cache stores the transformed ``image`` and ``mask``
once (after band selection and resizing) and serves batches from it:

* ``disk``: ``.npy`` memory maps under a cache directory, built once and shared by concurrent runs
  through the OS page cache (one RAM copy however many jobs read it).
* ``host``: tensors in this process's RAM.
* ``gpu``: tensors on the training device; batches need no host-to-device copy.

Images stay float32 and masks are stored in an integer type that holds every label, so batches
are bitwise identical to the uncached loader, including the shuffled order. The cache is
therefore an execution setting and does not enter the config hash.
"""

import contextlib
import hashlib
import inspect
import json
import logging
import os
import shutil
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

SampleCacheMode = Literal["none", "disk", "host", "gpu"]
_FORMAT_VERSION = 1
_LOCK_STALE_SECONDS = 6 * 3600


def _mask_dtype(masks: torch.Tensor) -> np.dtype:
    low, high = int(masks.min()), int(masks.max())
    if low >= 0 and high <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if np.iinfo(np.int16).min <= low and high <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    return np.dtype(np.int64)


def _source_digest(dataset: Dataset) -> str:
    """Hash the code that produces samples: the dataset's module and the loading transforms."""
    from geo_eval_harness.datasets import loading

    digest = hashlib.sha256()
    for obj in (type(dataset), loading):
        try:
            digest.update(inspect.getsource(inspect.getmodule(obj) or obj).encode())
        except (OSError, TypeError):
            digest.update(repr(obj).encode())
    return digest.hexdigest()[:16]


@contextlib.contextmanager
def _worker_threads() -> Iterator[None]:
    """Run Torch ops with one intra-op thread, as ``DataLoader`` workers do.

    CPU resizing can round differently with other thread counts, so samples produced in the main
    process would otherwise differ in the last bits from worker-produced ones.
    """
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def cache_key(dataset: Dataset, identity: dict[str, object]) -> str:
    """Return a key for ``dataset`` from its run identity, code, length, and two sample digests.

    The sample digests catch changed data files under an unchanged configuration.
    """
    probe = hashlib.sha256()
    n = len(dataset)  # type: ignore[arg-type]
    with _worker_threads():
        for index in sorted({0, n - 1}) if n else []:
            sample = dataset[index]
            probe.update(sample["image"].numpy().tobytes())
            probe.update(sample["mask"].numpy().tobytes())
    payload = {
        "format": _FORMAT_VERSION,
        "identity": identity,
        "length": n,
        "source": _source_digest(dataset),
        "samples": probe.hexdigest()[:16],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _iterate(dataset: Dataset, num_workers: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    if num_workers:
        for batch in loader:
            yield batch["image"], batch["mask"]
        return
    iterator = iter(loader)
    while True:
        with _worker_threads():
            batch = next(iterator, None)
        if batch is None:
            return
        yield batch["image"], batch["mask"]


def _materialize(dataset: Dataset, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    images, masks = [], []
    for image, mask in _iterate(dataset, num_workers):
        if image.dtype != torch.float32:
            raise TypeError(f"Sample cache expects float32 images, got {image.dtype}.")
        images.append(image)
        masks.append(mask)
    return torch.cat(images), torch.cat(masks)


def _build_disk(dataset: Dataset, directory: Path, num_workers: int) -> None:
    """Write ``images.npy``/``masks.npy`` into a temporary directory, then rename it into place."""
    tmp = directory.with_name(f"{directory.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = len(dataset)  # type: ignore[arg-type]
    images_mm = masks_mm = None
    offset = 0
    mask_chunks = []
    for image, mask in _iterate(dataset, num_workers):
        if image.dtype != torch.float32:
            raise TypeError(f"Sample cache expects float32 images, got {image.dtype}.")
        if images_mm is None:
            images_mm = np.lib.format.open_memmap(
                tmp / "images.npy", mode="w+", dtype=np.float32, shape=(n, *image.shape[1:])
            )
        images_mm[offset : offset + len(image)] = image.numpy()
        mask_chunks.append(mask)
        offset += len(image)
    if offset != n or images_mm is None:
        raise RuntimeError(f"Sample cache read {offset} of {n} samples.")
    images_mm.flush()
    del images_mm
    masks = torch.cat(mask_chunks)
    masks_mm = np.lib.format.open_memmap(
        tmp / "masks.npy", mode="w+", dtype=_mask_dtype(masks), shape=tuple(masks.shape)
    )
    masks_mm[:] = masks.numpy()
    masks_mm.flush()
    del masks_mm
    (tmp / "complete").write_text(json.dumps({"samples": n, "created": time.time()}))
    tmp.rename(directory)


def _dead_lock_owner(lock: Path) -> int | None:
    """Return the PID that took ``lock`` if it was on this host and has exited, else ``None``."""
    try:
        host, pid = (lock / "owner").read_text().split()
    except (OSError, ValueError):  # allow-except: owner not written yet, or a legacy lock
        return None
    if host != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return int(pid)
    except PermissionError:  # allow-except: the PID exists but belongs to another user
        return None
    return None


def _open_disk(
    dataset: Dataset, directory: Path, num_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return memory maps for ``directory``, building them under a lock if needed."""
    lock = directory.with_name(f"{directory.name}.lock")
    while not (directory / "complete").exists():
        try:
            lock.mkdir(parents=True)
        except FileExistsError:
            dead = _dead_lock_owner(lock)
            if dead is not None:
                logger.warning("Removing sample cache lock %s left by exited PID %d", lock, dead)
                shutil.rmtree(
                    directory.with_name(f"{directory.name}.tmp-{dead}"), ignore_errors=True
                )
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.time() - lock.stat().st_mtime > _LOCK_STALE_SECONDS:
                raise RuntimeError(
                    f"Sample cache lock {lock} is older than {_LOCK_STALE_SECONDS} s; remove it "
                    "if no other run is building this cache."
                ) from None
            time.sleep(10)
            continue
        (lock / "owner").write_text(f"{socket.gethostname()} {os.getpid()}\n")
        try:
            if not (directory / "complete").exists():
                logger.info("Building sample cache %s (%d samples)", directory, len(dataset))  # type: ignore[arg-type]
                _build_disk(dataset, directory, num_workers)
        finally:
            shutil.rmtree(lock, ignore_errors=True)
    return (
        np.load(directory / "images.npy", mmap_mode="r"),
        np.load(directory / "masks.npy", mmap_mode="r"),
    )


class CachedSamples(Dataset):
    """Serve ``{"image", "mask"}`` samples from cached arrays or tensors."""

    def __init__(self, images: np.ndarray | torch.Tensor, masks: np.ndarray | torch.Tensor) -> None:
        if len(images) != len(masks):
            raise ValueError(f"{len(images)} images but {len(masks)} masks.")
        self.images = images
        self.masks = masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, mask = self.images[index], self.masks[index]
        if isinstance(image, np.ndarray):
            image, mask = torch.from_numpy(np.array(image)), torch.from_numpy(np.array(mask))
        return {"image": image, "mask": mask.long()}

    @property
    def nbytes(self) -> int:
        """Return the cached bytes."""
        return sum(
            a.nbytes if isinstance(a, np.ndarray) else a.numel() * a.element_size()
            for a in (self.images, self.masks)
        )


class DeviceBatchLoader:
    """Yield batches of device-resident samples in the order a ``DataLoader`` would.

    Mirrors ``DataLoader(shuffle=...)`` RNG use: each epoch first draws the loader's base seed from
    the global generator, then ``RandomSampler`` draws its own seed, so shuffles match exactly.
    """

    def __init__(
        self, images: torch.Tensor, masks: torch.Tensor, *, batch_size: int, shuffle: bool
    ) -> None:
        self.images, self.masks = images, masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = range(len(images))

    def __len__(self) -> int:
        return -(-len(self.images) // self.batch_size)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        torch.empty((), dtype=torch.int64).random_()  # DataLoader iterator base seed
        sampler = RandomSampler(self.dataset) if self.shuffle else SequentialSampler(self.dataset)
        for indices in BatchSampler(sampler, self.batch_size, drop_last=False):
            index = torch.as_tensor(indices, device=self.images.device)
            yield {"image": self.images[index], "mask": self.masks[index].long()}


def cache_loader(  # noqa: PLR0913 - one entry point for every mode
    loader: DataLoader,
    *,
    mode: SampleCacheMode,
    identity: dict[str, object],
    cache_dir: Path,
    device: torch.device,
    num_workers: int,
) -> DataLoader | DeviceBatchLoader:
    """Return a loader that serves the same batches as ``loader`` from a sample cache.

    Args:
        loader: Uncached loader; its dataset, batch size, and shuffle setting are reused.
        mode: ``disk``, ``host``, or ``gpu`` (``none`` returns ``loader``).
        identity: Run settings that determine the samples (dataset, split, bands, size, ...).
        cache_dir: Root for ``disk`` caches.
        device: Training device for ``gpu``.
        num_workers: Workers used to read the dataset while building the cache.
    """
    if mode == "none":
        return loader
    dataset = loader.dataset
    shuffle = isinstance(loader.sampler, RandomSampler)
    batch_size = int(loader.batch_size or 1)
    started = time.perf_counter()
    if mode == "disk":
        directory = Path(cache_dir).expanduser() / cache_key(dataset, identity)
        images, masks = _open_disk(dataset, directory, num_workers)
    elif mode in ("host", "gpu"):
        images, masks = _materialize(dataset, num_workers)
    else:
        raise ValueError(f"Unknown sample cache mode {mode!r}.")
    cached = CachedSamples(images, masks)
    logger.info(
        "Sample cache %s: %d samples, %.1f MiB, ready in %.1f s",
        mode,
        len(cached),
        cached.nbytes / 2**20,
        time.perf_counter() - started,
    )
    if mode == "gpu":
        if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
            raise TypeError("GPU sample cache needs materialized tensors.")
        return DeviceBatchLoader(
            images.to(device), masks.to(device), batch_size=batch_size, shuffle=shuffle
        )
    return DataLoader(
        cached,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
