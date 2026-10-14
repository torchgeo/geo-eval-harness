"""Tests for the on-disk segmentation feature cache."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from torchgeo_bench import feature_cache
from torchgeo_bench.segmentation_probe import CachedFeaturesDataset

IDENTITY = {
    "model": {"name": "m", "target": "pkg.Model", "kwargs": {}},
    "weights": "abc",
    "input": {"bands": "rgb", "time_steps": None},
    "layers": ["blocks.11"],
    "dataset": "burn_scars",
    "split": "train",
}


@pytest.mark.parametrize(
    "change",
    [
        {"model": {"name": "m", "target": "pkg.Model", "kwargs": {"pool": "cls"}}},
        {"weights": "abd"},
        {"input": {"bands": "rgb", "time_steps": 6}},
        {"layers": ["blocks.8"]},
        {"dataset": "pastis"},
        {"split": "val"},
    ],
)
def test_key_changes_with_each_identity_field(change: dict) -> None:
    key = feature_cache.feature_cache_key(IDENTITY)
    assert key == feature_cache.feature_cache_key(dict(reversed(IDENTITY.items())))
    assert key != feature_cache.feature_cache_key({**IDENTITY, **change})


def test_weights_digest_tracks_parameter_values() -> None:
    model = torch.nn.Linear(2, 2)
    before = feature_cache.weights_digest(model)
    with torch.no_grad():
        model.bias.add_(1)
    assert feature_cache.weights_digest(model) != before


def _batches(n: int = 5, batch_size: int = 2) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
    layer = torch.arange(n * 4, dtype=torch.float16).reshape(n, 1, 2, 2)
    other = -torch.arange(n * 3, dtype=torch.float16).reshape(n, 3, 1, 1)
    masks = torch.arange(n * 4).reshape(n, 2, 2) % 3
    masks[0, 0, 0] = 255
    return [
        ([layer[i : i + batch_size], other[i : i + batch_size]], masks[i : i + batch_size])
        for i in range(0, n, batch_size)
    ]


def _concat(batches: list[tuple[list[torch.Tensor], torch.Tensor]]) -> CachedFeaturesDataset:
    layers = [torch.cat([b[0][i] for b in batches]) for i in range(len(batches[0][0]))]
    return CachedFeaturesDataset(layers, torch.cat([b[1] for b in batches]))


def test_cached_features_stream_once_then_load(tmp_path: Path) -> None:
    calls = []

    def batches() -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        calls.append(1)
        return _batches()

    path = tmp_path / "burn_scars" / "key-train.pt"
    first = feature_cache.cached_features(path, batches, num_samples=5)
    second = feature_cache.cached_features(path, batches, num_samples=5)

    assert len(calls) == 1
    assert not path.exists()
    assert sorted(p.name for p in path.with_suffix("").iterdir()) == [
        "layer0.npy",
        "layer1.npy",
        "masks.npy",
    ]
    expected = _concat(_batches())
    for loaded in (first, second):
        assert loaded.masks.dtype == torch.int16
        torch.testing.assert_close(loaded.masks.long(), expected.masks)
        for got, want in zip(loaded.layer_tensors, expected.layer_tensors, strict=True):
            assert got.dtype == torch.float16
            torch.testing.assert_close(got, want)
    assert list(path.parent.glob(".*.tmp")) == []


def test_cached_features_read_legacy_pt_files_without_extracting(tmp_path: Path) -> None:
    path = tmp_path / "key-val.pt"
    legacy = _concat(_batches())
    torch.save({"layer_tensors": legacy.layer_tensors, "masks": legacy.masks.to(torch.int16)}, path)

    def batches() -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        raise AssertionError("a legacy cache file must not be re-extracted")

    loaded = feature_cache.cached_features(path, batches, num_samples=5)

    torch.testing.assert_close(loaded.masks.long(), legacy.masks)
    torch.testing.assert_close(loaded.layer_tensors[1], legacy.layer_tensors[1])
    assert not path.with_suffix("").exists()


def test_failed_stream_leaves_no_partial_cache(tmp_path: Path) -> None:
    def broken() -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        yield _batches()[0]
        raise OSError("disk full")

    path = tmp_path / "key-train.pt"
    with pytest.raises(OSError, match="disk full"):
        feature_cache.cached_features(path, broken, num_samples=5)

    assert [p.name for p in tmp_path.iterdir() if p.suffix != ".lock"] == []


@pytest.mark.parametrize("num_samples", [4, 6])
def test_stream_rejects_a_sample_count_mismatch(tmp_path: Path, num_samples: int) -> None:
    with pytest.raises(ValueError, match=r"5|more than"):
        feature_cache.stream_features(tmp_path / "key-test", _batches(), num_samples)
    assert list(tmp_path.iterdir()) == []


def test_stream_rejects_masks_that_do_not_fit_int16(tmp_path: Path) -> None:
    features, masks = _batches()[0]
    with pytest.raises(ValueError, match="int16"):
        feature_cache.stream_features(tmp_path / "key", [(features, masks + 40_000)], len(masks))
    assert list(tmp_path.iterdir()) == []


def test_streamed_features_match_in_memory_extraction(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    from torchgeo_bench.segmentation_probe import SegmentationProbe

    backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.Conv2d(4, 2, 1))
    probe = SegmentationProbe(
        backbone=backbone, layer_names=["0", "1"], num_classes=3, head_type="linear"
    )
    data = TensorDataset(torch.rand(5, 3, 4, 4), torch.randint(0, 3, (5, 4, 4)))
    loader = DataLoader(data, batch_size=2)

    in_memory = probe.extract_segmentation_features(loader)
    streamed = feature_cache.cached_features(
        tmp_path / "key-train.pt",
        lambda: probe.iter_segmentation_features(loader),
        num_samples=len(data),
    )

    torch.testing.assert_close(streamed.masks.long(), in_memory.masks)
    for got, want in zip(streamed.layer_tensors, in_memory.layer_tensors, strict=True):
        torch.testing.assert_close(got, want)
