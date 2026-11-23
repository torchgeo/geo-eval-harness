"""Landsat-Bench datasets: split loading, band selection, clamping/resizing, and downloads."""

import io
import json
import tarfile
import zipfile
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
)
from tests.support.data import geotiff_bytes


def _make_landsat_geotiff(n_bands: int, height: int, width: int, offset: float = 0.0) -> bytes:
    """Write a GeoTIFF with band *b* set to ``100 * (b + 1) + offset`` and a negative corner."""
    pixels = np.empty((n_bands, height, width), dtype=np.int16)
    for b in range(n_bands):
        pixels[b] = int(100 * (b + 1) + offset)
    pixels[:, 0, 0] = -50
    return geotiff_bytes(pixels)


def test_eurosat_l_loads_splits_excludes_thermal_band_and_resizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eurosat-l"
    for split, items in (
        ("train", [("AnnualCrop", "AnnualCrop_1"), ("Forest", "Forest_2")]),
        ("val", [("River", "River_3")]),
        ("test", [("SeaLake", "SeaLake_4")]),
    ):
        (root / f"eurosat-{split}.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / f"eurosat-{split}.txt").write_text(
            "\n".join(f"{stem}.jpg" for _, stem in items) + "\n", encoding="utf-8"
        )
        for cls_name, stem in items:
            path = root / "eurosat-l" / cls_name / f"{stem}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Upstream files contain 8 bands (SR_B1..SR_B7 + ST_B10) at ~21x23 pixels.
            path.write_bytes(_make_landsat_geotiff(8, 21, 23))

    bench = EuroSATL()
    train = bench.get_dataset("train")
    assert len(train) == 2
    sample = train[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (7, 22, 22)
    assert sample["image"].min().item() >= 0.0
    assert sample["label"].dtype == torch.long
    assert [int(train[i]["label"]) for i in range(len(train))] == [
        bench.class_names.index("AnnualCrop"),
        bench.class_names.index("Forest"),
    ]

    rgb = bench.get_dataset("test", bands=("red", "green", "blue"))[0]
    assert rgb["image"].shape == (3, 22, 22)
    # SR_B4=400, SR_B3=300, SR_B2=200 away from the clamped corner pixel.
    torch.testing.assert_close(
        rgb["image"][:, -1, -1], torch.tensor([400.0, 300.0, 200.0], dtype=torch.float32)
    )


def test_bigearthnet_l_loads_19_class_multihot_labels_and_resizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "bigearthnet-l"
    image_dir = root / "bigearthnet-l" / "images"
    label_dir = root / "bigearthnet-l" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for split, patch_id, labels in (
        ("train", "S2A_patch_1", ["Continuous urban fabric", "Broad-leaved forest", "Airports"]),
        ("val", "S2A_patch_2", ["Pastures"]),
        ("test", "S2A_patch_3", ["Sea and ocean"]),
    ):
        (root / f"bigearthnet-{split}.csv").write_text(f"{patch_id},S1B_unused\n", encoding="utf-8")
        (image_dir / f"{patch_id}.tif").write_bytes(_make_landsat_geotiff(7, 39, 41))
        (label_dir / f"{patch_id}_labels_metadata.json").write_text(
            json.dumps({"labels": labels}), encoding="utf-8"
        )

    bench = BigEarthNetL()
    sample = bench.get_dataset("train", bands=("nir", "red"))[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (2, 40, 40)
    assert sample["image"].min().item() >= 0.0
    torch.testing.assert_close(
        sample["image"][:, -1, -1], torch.tensor([500.0, 400.0], dtype=torch.float32)
    )
    assert sample["label"].shape == (19,)
    assert sample["label"].dtype == torch.long
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

    pid_a = "0000001_-70.25_-55.25"
    pid_b = "0000002_10.50_45.00"
    labels_a = [1] + [0] * 22
    labels_b = [0, 1] + [0] * 21
    labels_missing = [0, 0, 1] + [0] * 20

    for split in ("train", "val", "test"):
        rows = [
            f"{pid_a}," + ",".join(map(str, labels_a)),
            f"{pid_b}," + ",".join(map(str, labels_b)),
        ]
        static_rows = [f"{pid_b},b.tif", f"{pid_a},a.tif"]
        if split == "test":
            rows.append(f"{LC100_MISSING_PID}," + ",".join(map(str, labels_missing)))
            static_rows.append(f"{LC100_MISSING_PID},missing.tif")
        (root / f"multilabel-{split}.csv").write_text("\n".join([header, *rows]) + "\n")
        (root / f"static_fnames-{split}.csv").write_text("\n".join(static_rows) + "\n")

    for pid, fname, offset in ((pid_a, "a.tif", 10.0), (pid_b, "b.tif", 50.0)):
        path = root / "lc100-l" / pid / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_make_landsat_geotiff(7, 16, 16, offset=offset))

    bench = LC100L()
    test_ds = bench.get_dataset("test", bands=("coastal_aerosol",))
    assert len(test_ds) == 2
    first = test_ds[0]
    second = test_ds[1]
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


@pytest.mark.parametrize(
    ("cls", "name"),
    [(EuroSATL, "eurosat-l"), (BigEarthNetL, "bigearthnet-l"), (LC100L, "lc100-l")],
)
def test_missing_data_and_invalid_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type, name: str
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        cls().get_dataset("bogus")
    with pytest.raises(FileNotFoundError, match=f"download {name}"):
        cls().get_dataset("train")


def test_download_eurosat_l_extracts_zip(tmp_path: Path) -> None:
    out = tmp_path / "data"

    def fetch(
        repo_id: str, revision: str, local_dir: Path, *args: object, **kwargs: object
    ) -> None:
        with zipfile.ZipFile(local_dir / "eurosat-l.zip", "w") as archive:
            archive.writestr("eurosat-l/Forest/Forest_1.tif", b"tif")

    with mock.patch.object(download, "fetch_pinned", side_effect=fetch) as pinned:
        download.download_datasets(["eurosat-l"], out)

    assert pinned.call_args.args[:3] == (
        download.EUROSAT_L_REPO,
        download.EUROSAT_L_REVISION,
        out / "eurosat-l",
    )
    assert (out / "eurosat-l" / "eurosat-l" / "Forest" / "Forest_1.tif").read_bytes() == b"tif"


def test_download_bigearthnet_l_extracts_tarball(tmp_path: Path) -> None:
    out = tmp_path / "data"

    def fetch(
        repo_id: str, revision: str, local_dir: Path, *args: object, **kwargs: object
    ) -> None:
        with tarfile.open(local_dir / "bigearthnet-l.tar.gz", "w:gz") as archive:
            payload = b"tif"
            info = tarfile.TarInfo("bigearthnet-l/images/patch.tif")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with mock.patch.object(download, "fetch_pinned", side_effect=fetch) as pinned:
        download.download_datasets(["bigearthnet-l"], out)

    assert pinned.call_args.args[:3] == (
        download.BIGEARTHNET_L_REPO,
        download.BIGEARTHNET_L_REVISION,
        out / "bigearthnet-l",
    )
    assert (out / "bigearthnet-l" / "bigearthnet-l" / "images" / "patch.tif").read_bytes() == b"tif"


def test_download_lc100_l_streams_multipart_tarball(tmp_path: Path) -> None:
    out = tmp_path / "data"
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as archive:
        payload = b"lc100-geotiff-bytes" * 16
        info = tarfile.TarInfo("lc100-l/0000001_-70.25_-55.25/scene.tif")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    raw = tar_buf.getvalue()
    part_names = [
        "lc100-l.tar.gz.part.aa",
        "lc100-l.tar.gz.part.ab",
        "lc100-l.tar.gz.part.ac",
        "lc100-l.tar.gz.part.ad",
    ]
    step = max(1, len(raw) // 4)
    chunks = [raw[0:step], raw[step : 2 * step], raw[2 * step : 3 * step], raw[3 * step :]]

    def fetch(
        repo_id: str, revision: str, local_dir: Path, *args: object, **kwargs: object
    ) -> None:
        for name, chunk in zip(part_names, chunks, strict=True):
            (local_dir / name).write_bytes(chunk)

    with mock.patch.object(download, "fetch_pinned", side_effect=fetch) as pinned:
        download.download_datasets(["lc100-l"], out)

    assert pinned.call_args.args[:3] == (
        download.LC100_L_REPO,
        download.LC100_L_REVISION,
        out / "lc100-l",
    )
    extracted = out / "lc100-l" / "lc100-l" / "0000001_-70.25_-55.25" / "scene.tif"
    assert extracted.read_bytes() == b"lc100-geotiff-bytes" * 16
