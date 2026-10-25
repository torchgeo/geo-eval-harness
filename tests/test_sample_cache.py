"""Sample caches must serve exactly the batches of the uncached loader."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from geo_eval_harness.datasets import sample_cache
from geo_eval_harness.datasets.sample_cache import DeviceBatchLoader, cache_key, cache_loader

IDENTITY = {"dataset": "toy", "split": "train", "bands": "rgb", "image_size": 8}


class ToySegmentation(Dataset):
    """Deterministic samples with a non-integer image and an ignore label, plus an extra key."""

    def __init__(self, n: int = 13, offset: float = 0.0) -> None:
        generator = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 8, 8, generator=generator) + offset
        self.masks = torch.randint(0, 4, (n, 8, 8), generator=generator)
        self.masks[:, 0, 0] = 255

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "index": torch.tensor(index),
        }


def _epochs(loader: DataLoader | DeviceBatchLoader, seed: int, epochs: int = 2) -> list:
    torch.manual_seed(seed)
    return [
        [(batch["image"].cpu(), batch["mask"].cpu()) for batch in loader] for _ in range(epochs)
    ]


@pytest.mark.parametrize("mode", ["disk", "host", "gpu"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_cached_batches_are_bitwise_identical(tmp_path, mode: str, *, shuffle: bool) -> None:
    loader = DataLoader(ToySegmentation(), batch_size=4, shuffle=shuffle, num_workers=0)
    device = torch.device("cuda:0" if mode == "gpu" and torch.cuda.is_available() else "cpu")
    cached = cache_loader(
        loader, mode=mode, identity=IDENTITY, cache_dir=tmp_path, device=device, num_workers=0
    )

    expected, actual = _epochs(loader, seed=3), _epochs(cached, seed=3)

    assert len(cached) == len(loader)
    for epoch_expected, epoch_actual in zip(expected, actual, strict=True):
        assert len(epoch_expected) == len(epoch_actual)
        for (image, mask), (cached_image, cached_mask) in zip(
            epoch_expected, epoch_actual, strict=True
        ):
            assert cached_image.dtype == torch.float32
            assert cached_mask.dtype == torch.long
            assert torch.equal(image, cached_image)
            assert torch.equal(mask.long(), cached_mask)
    if shuffle:
        assert not torch.equal(expected[0][0][0], expected[1][0][0])


def test_disk_cache_is_built_once_and_stores_small_mask_dtype(tmp_path, monkeypatch) -> None:
    loader = DataLoader(ToySegmentation(), batch_size=4)
    options = {"mode": "disk", "identity": IDENTITY, "cache_dir": tmp_path, "num_workers": 0}
    cache_loader(loader, device=torch.device("cpu"), **options)
    (directory,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert np.load(directory / "masks.npy", mmap_mode="r").dtype == np.uint8

    def fail(*_args: object) -> None:
        raise AssertionError("cache rebuilt")

    monkeypatch.setattr(sample_cache, "_build_disk", fail)
    cache_loader(loader, device=torch.device("cpu"), **options)
    assert [p.name for p in tmp_path.iterdir()] == [directory.name]


def test_cache_key_tracks_identity_and_data() -> None:
    key = cache_key(ToySegmentation(), IDENTITY)
    assert key == cache_key(ToySegmentation(), dict(IDENTITY))
    assert key != cache_key(ToySegmentation(), {**IDENTITY, "split": "val"})
    assert key != cache_key(ToySegmentation(offset=1.0), IDENTITY)
    assert key != cache_key(ToySegmentation(n=12), IDENTITY)


def test_sample_cache_does_not_change_the_config_hash() -> None:
    from geo_eval_harness.config.presets import resolve_run_config
    from geo_eval_harness.config.run import RunConfig
    from geo_eval_harness.resume import resume_config_hash

    def run_hash(**runtime: object) -> str:
        config = RunConfig.model_validate(
            {"model": {"name": "rcf"}, "datasets": ["m-eurosat"], "runtime": runtime}
        )
        resolved, preset = resolve_run_config(config, "m-eurosat")
        return resume_config_hash(resolved, preset, task="segmentation")

    assert run_hash() == run_hash(sample_cache="disk", sample_cache_dir="/elsewhere")


@pytest.mark.parametrize("mode", ["none", "host", "disk"])
def test_run_wraps_every_segmentation_split_in_the_sample_cache(tmp_path, mode: str) -> None:
    from geo_eval_harness.config.run import RunConfig
    from geo_eval_harness.main import LoaderSplits, _cache_segmentation_loaders

    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {
                "device": "cpu",
                "workers": 0,
                "sample_cache": mode,
                "sample_cache_dir": str(tmp_path),
            },
        }
    )
    loaders = LoaderSplits(
        *(DataLoader(ToySegmentation(n), batch_size=4, num_workers=0) for n in (9, 5, 6))
    )

    cached = _cache_segmentation_loaders(config, "toy", loaders)

    for split in ("train", "val", "test"):
        loader, served = getattr(loaders, split), getattr(cached, split)
        assert (served is loader) == (mode == "none")
        assert [b["image"].tolist() for b in served] == [b["image"].tolist() for b in loader]
    assert any(tmp_path.iterdir()) == (mode == "disk")


def test_lock_left_by_an_exited_build_is_removed(tmp_path) -> None:
    import socket
    import subprocess
    import sys

    dataset = ToySegmentation()
    directory = tmp_path / cache_key(dataset, IDENTITY)
    dead = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    pid = int(dead.stdout)
    lock = directory.with_name(f"{directory.name}.lock")
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"{socket.gethostname()} {pid}\n")
    partial = directory.with_name(f"{directory.name}.tmp-{pid}")
    partial.mkdir()
    loader = DataLoader(dataset, batch_size=4)
    cache_loader(
        loader,
        mode="disk",
        identity=IDENTITY,
        cache_dir=tmp_path,
        device=torch.device("cpu"),
        num_workers=0,
    )
    assert (directory / "complete").exists()
    assert not lock.exists()
    assert not partial.exists()


def test_lock_held_by_a_live_process_is_kept(tmp_path) -> None:
    import os
    import socket

    lock = tmp_path / "x.lock"
    lock.mkdir()
    (lock / "owner").write_text(f"{socket.gethostname()} {os.getpid()}\n")
    assert sample_cache._dead_lock_owner(lock) is None
    (lock / "owner").write_text(f"other-host {os.getpid()}\n")
    assert sample_cache._dead_lock_owner(lock) is None
