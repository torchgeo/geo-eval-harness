"""Landsat-Bench datasets: split loading, band selection, clamping/resizing, and downloads."""

import io
import json
import tarfile
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from geo_eval_harness import download, geography
from geo_eval_harness.datasets.landsatbench import (
    LC100_MISSING_PID,
    LC100L,
    BigEarthNetL,
    EuroSATL,
    EuroSATLSpatial,
)
from tests.support.data import geotiff_bytes


def _make_landsat_geotiff(n_bands: int, height: int, width: int, offset: float = 0.0) -> bytes:
    """Write a GeoTIFF with band *b* set to ``100 * (b + 1) + offset`` and a negative corner."""
    pixels = np.empty((n_bands, height, width), dtype=np.int16)
    for b in range(n_bands):
        pixels[b] = int(100 * (b + 1) + offset)
    pixels[:, 0, 0] = -50
    return geotiff_bytes(pixels)


def test_eurosat_l_loads_standard_and_spatial_splits_and_excludes_thermal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eurosat-l"
    root.mkdir(parents=True)
    for cls_name, stem in (
        ("AnnualCrop", "AnnualCrop_1"),
        ("Forest", "Forest_2"),
        ("SeaLake", "SeaLake_3"),
    ):
        path = root / "eurosat-l" / cls_name / f"{stem}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Upstream files contain 8 bands (SR_B1..SR_B7 + ST_B10) at ~21x23 pixels.
        path.write_bytes(_make_landsat_geotiff(8, 21, 23))

    (root / "eurosat-train.txt").write_text("AnnualCrop_1.jpg\nForest_2.jpg\n", encoding="utf-8")
    (root / "eurosat-test.txt").write_text("SeaLake_3.jpg\n", encoding="utf-8")
    (root / "eurosat-spatial-train.txt").write_text("SeaLake_3.jpg\n", encoding="utf-8")

    bench = EuroSATL()
    train = bench.get_dataset("train")
    assert len(train) == 2
    sample = train[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (7, 22, 22)
    assert sample["image"].min().item() >= 0.0
    assert [int(train[i]["label"]) for i in range(len(train))] == [
        bench.class_names.index("AnnualCrop"),
        bench.class_names.index("Forest"),
    ]

    rgb = bench.get_dataset("test", bands=("red", "green", "blue"))[0]
    torch.testing.assert_close(
        rgb["image"][:, -1, -1], torch.tensor([400.0, 300.0, 200.0], dtype=torch.float32)
    )

    spatial_train = EuroSATLSpatial().get_dataset("train")
    assert len(spatial_train) == 1
    assert int(spatial_train[0]["label"]) == bench.class_names.index("SeaLake")


def test_bigearthnet_l_loads_19_class_multihot_labels_and_resizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "bigearthnet-l"
    image_dir = root / "bigearthnet-l" / "images"
    label_dir = root / "bigearthnet-l" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    (root / "bigearthnet-train.csv").write_text("S2A_patch_1,S1B_unused\n", encoding="utf-8")
    (image_dir / "S2A_patch_1.tif").write_bytes(_make_landsat_geotiff(7, 39, 41))
    (label_dir / "S2A_patch_1_labels_metadata.json").write_text(
        json.dumps({"labels": ["Continuous urban fabric", "Broad-leaved forest", "Airports"]}),
        encoding="utf-8",
    )

    bench = BigEarthNetL()
    sample = bench.get_dataset("train", bands=("nir", "red"))[0]
    assert sample["image"].shape == (2, 40, 40)
    assert sample["image"].min().item() >= 0.0
    torch.testing.assert_close(
        sample["image"][:, -1, -1], torch.tensor([500.0, 400.0], dtype=torch.float32)
    )
    # Continuous urban fabric -> Urban fabric; Broad-leaved forest kept; Airports dropped in 19-cls.
    expected = torch.zeros(19, dtype=torch.long)
    expected[bench.class_names.index("Urban fabric")] = 1
    expected[bench.class_names.index("Broad-leaved forest")] = 1
    torch.testing.assert_close(sample["label"], expected)


def test_lc100_l_aligns_static_files_by_pid_and_drops_missing_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "lc100-l"
    root.mkdir(parents=True)
    header = "PID," + ",".join(f"c{i}" for i in range(23))

    pid_a, pid_b = "0000001_-70.25_-55.25", "0000002_10.50_45.00"
    rows = [
        f"{pid_a},1," + ",".join(["0"] * 22),
        f"{pid_b},0,1," + ",".join(["0"] * 21),
    ]
    for split in ("train", "val", "test"):
        split_rows = (
            [*rows, f"{LC100_MISSING_PID},0,0,1," + ",".join(["0"] * 20)]
            if split == "test"
            else rows
        )
        static_rows = [f"{pid_b},b.tif", f"{pid_a},a.tif"]
        if split == "test":
            static_rows.append(f"{LC100_MISSING_PID},missing.tif")
        (root / f"multilabel-{split}.csv").write_text("\n".join([header, *split_rows]) + "\n")
        (root / f"static_fnames-{split}.csv").write_text("\n".join(static_rows) + "\n")

    for pid, fname, offset in ((pid_a, "a.tif", 10.0), (pid_b, "b.tif", 50.0)):
        path = root / "lc100-l" / pid / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_make_landsat_geotiff(7, 16, 16, offset=offset))

    test_ds = LC100L().get_dataset("test", bands=("coastal_aerosol",))
    assert len(test_ds) == 2
    first, second = test_ds[0], test_ds[1]
    assert first["image"].shape == (1, 224, 224)
    # pid_a is first in multilabel-test.csv and must pair with a.tif (offset 10), not b.tif.
    torch.testing.assert_close(first["image"][0, -1, -1], torch.tensor(110.0))
    assert int(first["label"][0]) == 1
    assert int(first["label"][1]) == 0
    torch.testing.assert_close(second["image"][0, -1, -1], torch.tensor(150.0))
    assert int(second["label"][0]) == 0
    assert int(second["label"][1]) == 1

    monkeypatch.setattr(
        geography, "_attribute_continents", lambda lon, lat: Counter(Europe=len(lon))
    )
    record = geography.extract_geography("lc100-l")
    assert (record.status, record.version, record.n) == ("extracted", "manifest", 6)


def test_download_lc100_l_streams_multipart_tarball(tmp_path: Path) -> None:
    out = tmp_path / "data"
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as archive:
        payload = b"lc100-geotiff-bytes" * 16
        info = tarfile.TarInfo("lc100-l/0000001_-70.25_-55.25/scene.tif")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    raw = tar_buf.getvalue()
    part_names = [f"lc100-l.tar.gz.part.a{c}" for c in "abcd"]
    step = max(1, len(raw) // 4)
    chunks = [raw[0:step], raw[step : 2 * step], raw[2 * step : 3 * step], raw[3 * step :]]

    def fetch(
        repo_id: str, revision: str, local_dir: Path, *args: object, **kwargs: object
    ) -> None:
        for name, chunk in zip(part_names, chunks, strict=True):
            (local_dir / name).write_bytes(chunk)

    with mock.patch.object(download, "fetch_pinned", side_effect=fetch):
        download.download_datasets(["lc100-l"], out)

    extracted = out / "lc100-l" / "lc100-l" / "0000001_-70.25_-55.25" / "scene.tif"
    assert extracted.read_bytes() == b"lc100-geotiff-bytes" * 16
