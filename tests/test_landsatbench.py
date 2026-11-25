"""Tests for Landsat-Bench GeoTIFF and Parquet loading, band selection, and geography."""

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torchvision.transforms.functional as TF

from geo_eval_harness import geography
from geo_eval_harness.datasets.landsatbench import (
    BigEarthNetL,
    EuroSATL,
    EuroSATLSpatial,
)
from tests.support.data import geotiff_bytes


def _write_landsat_geotiff(path: Path, height: int, width: int) -> None:
    pixels = np.empty((7, height, width), dtype=np.int16)
    for b in range(7):
        pixels[b] = 100 * (b + 1)
    pixels[:, 0, 0] = -50
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(geotiff_bytes(pixels))


def test_eurosat_l_loads_standard_and_spatial_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eurosat-l"
    _write_landsat_geotiff(root / "images/AnnualCrop/a.tif", 21, 23)
    _write_landsat_geotiff(root / "images/SeaLake/b.tif", 22, 22)
    pq.write_table(
        pa.table(
            {
                "path": ["images/AnnualCrop/a.tif", "images/SeaLake/b.tif"],
                "split": ["train", "test"],
                "spatial_split": ["test", "train"],
                "label": pa.array([0, 9], type=pa.int64()),
                "lon": [10.0, 12.0],
                "lat": [45.0, 46.0],
            }
        ),
        root / "index.parquet",
    )

    sample = EuroSATL().get_dataset("train")[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (7, 22, 22)
    assert sample["image"].min().item() >= 0.0
    assert int(sample["label"]) == 0

    rgb = EuroSATL().get_dataset("test", bands=("red", "green", "blue"))[0]
    torch.testing.assert_close(
        rgb["image"][:, -1, -1], torch.tensor([400.0, 300.0, 200.0], dtype=torch.float32)
    )

    spatial = EuroSATLSpatial().get_dataset("train")
    assert len(spatial) == 1
    assert int(spatial[0]["label"]) == 9


def test_bigearthnet_l_loads_multilabel_and_extracts_geography(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "bigearthnet-l"
    rng = np.random.default_rng(0)
    raw = rng.integers(7000, 18000, size=(7, 45, 45), dtype=np.int16)
    path = root / "images/sample.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(geotiff_bytes(raw))
    vec = [1] + [0] * 18
    pq.write_table(
        pa.table(
            {
                "path": ["images/sample.tif"],
                "split": ["train"],
                "label": pa.array([vec], type=pa.list_(pa.int8(), 19)),
                "lon": [10.5],
                "lat": [45.2],
            }
        ),
        root / "index.parquet",
    )

    sample = BigEarthNetL().get_dataset("train", bands=("nir", "red"))[0]
    expected_img = TF.resize(torch.from_numpy(raw[[4, 3]].astype(np.float32)), [40, 40])
    torch.testing.assert_close(sample["image"], expected_img)
    assert sample["label"].shape == (19,)
    assert sample["label"].dtype == torch.long
    assert int(sample["label"][0]) == 1

    monkeypatch.setattr(
        geography, "_attribute_continents", lambda lon, lat: Counter(Europe=len(lon))
    )
    record = geography.extract_geography("bigearthnet-l")
    assert (record.status, record.version, record.n) == ("extracted", "index", 1)
