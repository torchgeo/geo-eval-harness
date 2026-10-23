"""HOTOSM building segmentation: conversion, loading, band selection, and download."""

import io
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from geo_eval_harness import download, hub
from geo_eval_harness.datasets.hotosm_buildings import HOTOSMBuildings, convert_hotosm_split

RGB = (11, 22, 33)


def _encode(image: Image.Image, fmt: str) -> dict[str, bytes | str]:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return {"bytes": buffer.getvalue(), "path": "tile.tif"}


def _write_upstream_shard(path: Path, tile_ids: list[str]) -> None:
    """Write upstream-format rows alternating lossless RGB and RGBA tiles with nodata."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 255
    images, masks = [], []
    for index in range(len(tile_ids)):
        if index % 2:
            rgba = np.zeros((8, 8, 4), dtype=np.uint8)
            rgba[..., :3] = RGB
            rgba[..., 3] = 255
            rgba[:2, :, :] = 0  # top two rows are nodata
            images.append(_encode(Image.fromarray(rgba, "RGBA"), "PNG"))
        else:
            # PNG keeps the fixture lossless; upstream RGB tiles are JPEG.
            images.append(_encode(Image.new("RGB", (8, 8), RGB), "PNG"))
        masks.append(_encode(Image.fromarray(mask), "TIFF"))
    n = len(tile_ids)
    table = pa.table(
        {
            "image": images,
            "mask": masks,
            "tile_id": tile_ids,
            "project_id": pa.array([7] * n, pa.int32()),
            "country": ["Peru"] * n,
            "bbox_west": [10.0] * n,
            "bbox_south": [-5.0] * n,
            "bbox_east": [12.0] * n,
            "bbox_north": [-3.0] * n,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=1)


@pytest.fixture
def converted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    upstream = tmp_path / "upstream"
    _write_upstream_shard(upstream / "train-00000.parquet", ["a", "b"])
    _write_upstream_shard(upstream / "train-00001.parquet", ["c"])
    root = tmp_path / "data" / "hotosm_buildings"
    root.mkdir(parents=True)
    rows = convert_hotosm_split(
        upstream.glob("train-*.parquet"), root / "train.parquet", expected_rows=3
    )
    assert rows == 3
    return root


def test_conversion_keeps_metadata_and_marks_nodata(converted: Path) -> None:
    table = pq.read_table(converted / "train.parquet")
    assert table["tile_id"].to_pylist() == ["a", "b", "c"]
    assert table["lon"].to_pylist() == [11.0] * 3
    assert table["lat"].to_pylist() == [-4.0] * 3

    dataset = HOTOSMBuildings().get_dataset("train")
    assert len(dataset) == 3
    for index, has_nodata in enumerate((False, True, False)):
        sample = dataset[index]
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (3, 8, 8)
        mask = sample["mask"]
        assert mask.dtype == torch.int64
        expected = torch.zeros(8, 8, dtype=torch.int64)
        expected[:, 4:] = 1
        if has_nodata:
            expected[:2] = 255
        torch.testing.assert_close(mask, expected)


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        (None, [11.0, 22.0, 33.0]),
        (("red",), [11.0]),
        (("blue", "red"), [33.0, 11.0]),
    ],
)
def test_band_selection_precedes_transform(
    converted: Path, bands: tuple[str, ...] | None, expected: list[float]
) -> None:
    del converted
    seen: list[torch.Tensor] = []

    def transform(sample: dict) -> dict:
        seen.append(sample["image"].clone())
        return sample

    sample = HOTOSMBuildings().get_dataset("train", bands=bands, transform=transform)[0]
    assert seen[0].shape == (len(expected), 8, 8)
    torch.testing.assert_close(sample["image"][:, 4, 4], torch.tensor(expected))


def test_missing_split_points_to_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="download hotosm_buildings"):
        HOTOSMBuildings().get_dataset("val")


def test_download_converts_pinned_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HOTOSMBuildings, "split_sizes", {"train": 2, "val": 1, "test": 1})
    monkeypatch.setattr(download, "load_checksums", lambda manifest: {})

    def fake_snapshot(*, local_dir: Path, **_: object) -> None:
        for prefix, ids in (("train", ["a", "b"]), ("validation", ["v"]), ("test", ["t"])):
            _write_upstream_shard(local_dir / "data" / f"{prefix}-00000-of-00001.parquet", ids)

    with mock.patch.object(hub, "snapshot_download", side_effect=fake_snapshot) as snapshot:
        download.download_datasets(["hotosm_buildings"], tmp_path)

    root = tmp_path / "hotosm_buildings"
    snapshot.assert_called_once_with(
        repo_id="hotosm/vhr-building-segmentation",
        repo_type="dataset",
        revision=download.HOTOSM_REVISION,
        local_dir=root / "upstream",
        allow_patterns=["data/*.parquet"],
        cache_dir=None,
    )
    assert sorted(path.name for path in root.iterdir()) == [
        "test.parquet",
        "train.parquet",
        "val.parquet",
    ]
    assert pq.read_table(root / "val.parquet")["tile_id"].to_pylist() == ["v"]


@pytest.mark.parametrize("existing_split", [False, True])
def test_download_rejects_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, existing_split: bool
) -> None:
    monkeypatch.setattr(download, "load_checksums", lambda manifest: {})
    target = tmp_path / "hotosm_buildings" / "train.parquet"
    if existing_split:
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing valid split")

    def fake_snapshot(*, local_dir: Path, **_: object) -> None:
        _write_upstream_shard(local_dir / "data" / "train-00000-of-00001.parquet", ["a"])

    with (
        mock.patch.object(hub, "snapshot_download", side_effect=fake_snapshot),
        pytest.raises(ValueError, match="expected 57890 rows, got 1"),
    ):
        download.download_hotosm_buildings(tmp_path)

    if existing_split:
        assert target.read_bytes() == b"existing valid split"
    else:
        assert not target.exists()
    assert not target.with_suffix(".parquet.tmp").exists()
    assert (target.parent / "upstream" / "data").is_dir()


def test_spawn_serialization_requires_zero_workers(converted: Path) -> None:
    del converted
    pickle.dumps(HOTOSMBuildings())
    dataset = HOTOSMBuildings().get_dataset("train")
    assert isinstance(dataset, HOTOSMBuildings)
    with pytest.raises(ValueError, match="--workers 0"):
        pickle.dumps(dataset)

    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=0)))
    assert batch["image"].shape == (3, 3, 8, 8)
    torch.testing.assert_close(batch["mask"][1, :2], torch.full((2, 8), 255))
