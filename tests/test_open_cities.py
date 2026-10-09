"""Tests for the Open Cities building segmentation dataset."""

import hashlib
import io
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets import open_cities as oc
from geo_eval_harness.download import download_open_cities, fetch_open_cities_file

CHIP = oc.CHIP_SIZE
ORIGIN = (500_000.0, 9_250_000.0)  # UTM 37S, near Dar es Salaam
GSD = 0.1


def _pixel_lonlat(col: float, row: float) -> list[float]:
    from pyproj import Transformer

    x, y = ORIGIN[0] + col * GSD, ORIGIN[1] - row * GSD
    lon, lat = Transformer.from_crs(32737, 4326, always_xy=True).transform(x, y)
    return [lon, lat]


def _polygon(cols: list[float], rows: list[float]) -> dict:
    ring = [_pixel_lonlat(c, r) for c, r in zip(cols, rows, strict=True)]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """Write ``dar/abc123.tif`` (2 chips wide) and its labels under ``tmp_path``.

    Chip 0 holds a 100x100 px building (twice, plus an invalid bow-tie) and three kinds
    of nodata: alpha 0, JPEG-like alpha noise 1-127, and black RGB under opaque alpha.
    Chip 1 is clean background.
    """
    root = tmp_path / "data" / "open_cities"
    (root / "dar").mkdir(parents=True)
    rgba = np.full((4, CHIP, 2 * CHIP), 120, dtype=np.uint8)
    rgba[3] = 255
    rgba[3, :50, :50] = 0
    rgba[3, :50, 50:100] = np.arange(1, 128, dtype=np.uint8).repeat(50)[: 50 * 50].reshape(50, 50)
    rgba[:3, 400:450, 400:450] = 5
    profile = {
        "driver": "GTiff",
        "width": 2 * CHIP,
        "height": CHIP,
        "count": 4,
        "dtype": "uint8",
        "crs": "EPSG:32737",
        "transform": from_origin(*ORIGIN, GSD, GSD),
        "tiled": True,
        "blockxsize": CHIP,
        "blockysize": CHIP,
    }
    with rasterio.open(root / "dar" / "abc123.tif", "w", **profile) as dst:
        dst.write(rgba)
    building = _polygon([200, 300, 300, 200], [200, 200, 300, 300])
    bowtie = _polygon([100, 150, 100, 150], [300, 350, 350, 300])
    features = [
        {"type": "Feature", "properties": {"building": "yes"}, "geometry": g}
        for g in (building, building, bowtie)
    ]
    (root / "dar" / "abc123.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    return root


@pytest.fixture
def index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chip_id": ["dar_abc123_0_0", "dar_abc123_0_512"],
            "city": "dar",
            "scene": "abc123",
            "col_off": [0, CHIP],
            "row_off": [0, 0],
            "split": ["train", "test"],
        }
    )


def test_mask_marks_buildings_and_every_kind_of_nodata(scene: Path, index: pd.DataFrame) -> None:
    path = oc.build_scene_mask(
        scene / "dar" / "abc123.tif", scene / "dar" / "abc123.geojson", index
    )
    with rasterio.open(path) as mask_ds, rasterio.open(scene / "dar" / "abc123.tif") as image_ds:
        mask = mask_ds.read(1)
        assert mask_ds.transform == image_ds.transform
        assert (mask_ds.width, mask_ds.height) == (image_ds.width, image_ds.height)
        invalid = ~oc.is_valid(image_ds.read())
    assert set(np.unique(mask)) == {0, 1, 255}
    np.testing.assert_array_equal(mask == 255, invalid)
    assert (mask[:50, :100] == 255).all()  # alpha 0 and alpha noise 1-127
    assert (mask[400:450, 400:450] == 255).all()  # black RGB under opaque alpha
    assert (mask[200:300, 200:300] == 1).all()
    # The repaired bow-tie (two triangles) adds a few hundred pixels at most.
    assert 100 * 100 <= (mask == 1).sum() < 100 * 100 + 50 * 50
    assert not mask[:, CHIP:].any()


def test_dataset_reads_native_chips_by_window(
    scene: Path, index: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    oc.build_scene_mask(scene / "dar" / "abc123.tif", scene / "dar" / "abc123.geojson", index)
    monkeypatch.chdir(scene.parents[1])
    monkeypatch.setattr(oc, "load_index", lambda: index)
    bench = get_bench_dataset_class("open_cities")()

    sample = bench.get_dataset("train")[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (3, CHIP, CHIP)
    assert sample["mask"].dtype == torch.int64
    assert sample["mask"].shape == (CHIP, CHIP)
    assert sample["mask"][250, 250] == 1
    assert sample["mask"][0, 0] == 255
    assert sample["image"][0, 0, 0] == 120

    test = bench.get_dataset("test", bands=("blue", "red"))
    assert len(test) == 1
    assert test[0]["image"].shape == (2, CHIP, CHIP)
    assert not test[0]["mask"].any()


def test_mask_preserves_polygons_in_nested_repaired_collections(
    scene: Path, index: pd.DataFrame
) -> None:
    bowtie_with_tail = _polygon(
        [100, 150, 100, 150, 100, 80, 100], [100, 150, 150, 100, 100, 100, 100]
    )
    geometry = {
        "type": "GeometryCollection",
        "geometries": [{"type": "GeometryCollection", "geometries": [bowtie_with_tail]}],
    }
    label_path = scene / "dar" / "abc123.geojson"
    label_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": geometry}],
            }
        )
    )
    path = oc.build_scene_mask(scene / "dar" / "abc123.tif", label_path, index)
    with rasterio.open(path) as ds:
        mask = ds.read(1)
    assert mask[110, 125] == 1
    assert mask[140, 125] == 1
    assert (mask[105:110, 120:130] == 1).all()
    assert (mask[140:145, 120:130] == 1).all()
    assert not mask[100, 80:100].any()


def test_spawn_worker_reads_dataset_after_parent_sample_access(
    scene: Path, index: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    oc.build_scene_mask(scene / "dar" / "abc123.tif", scene / "dar" / "abc123.geojson", index)
    monkeypatch.chdir(scene.parents[1])
    monkeypatch.setattr(oc, "load_index", lambda: index)
    dataset = oc.OpenCities().get_dataset("train")
    expected = dataset[0]
    loader = DataLoader(
        dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn", timeout=30
    )
    batches = list(loader)
    assert len(batches) == 1
    for key, value in expected.items():
        torch.testing.assert_close(batches[0][key][0], value)
        torch.testing.assert_close(dataset[0][key], value)


def test_dataset_reports_missing_scene_files(
    tmp_path: Path, index: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(oc, "load_index", lambda: index)
    with pytest.raises(FileNotFoundError, match="download open_cities"):
        get_bench_dataset_class("open_cities")().get_dataset("train")


def test_fetch_resumes_partial_downloads(tmp_path: Path) -> None:
    part = tmp_path / "kam" / "4e7c7f.tif.part"
    part.parent.mkdir()
    part.write_bytes(b"abc")
    response = io.BytesIO(b"def")
    response.status = 206  # type: ignore[attr-defined]
    with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
        path = fetch_open_cities_file("kam/4e7c7f.tif", tmp_path)
    request = urlopen.call_args.args[0]
    assert request.full_url == f"{oc.BASE_URL}/kam/4e7c7f/4e7c7f.tif"
    assert request.get_header("Range") == "bytes=3-"
    assert request.get_header("User-agent") == oc.USER_AGENT
    assert path.read_bytes() == b"abcdef"
    assert not part.exists()


def test_upstream_url_points_labels_to_their_stac_item() -> None:
    assert (
        oc.upstream_url("znz/076995.geojson") == f"{oc.BASE_URL}/znz/076995-labels/076995.geojson"
    )


def test_download_rejects_checksum_mismatch(scene: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oc, "load_checksums", lambda: {"dar/abc123.tif": hashlib.sha256(b"other").hexdigest()}
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_open_cities(scene.parent)


def test_download_builds_masks_and_checks_them_against_the_index(
    scene: Path, index: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = ("dar/abc123.tif", "dar/abc123.geojson")
    checksums = {f: hashlib.sha256((scene / f).read_bytes()).hexdigest() for f in files}
    monkeypatch.setattr(oc, "load_checksums", lambda: checksums)
    monkeypatch.setattr(oc, "load_index", lambda: index.assign(building_frac=[0.0, 0.0]))
    with pytest.raises(ValueError, match="train: mask building fraction"):
        download_open_cities(scene.parent)
    expected = (100 * 100 + 1250) / CHIP**2  # building plus the two bow-tie triangles
    monkeypatch.setattr(oc, "load_index", lambda: index.assign(building_frac=[expected, 0.0]))
    download_open_cities(scene.parent)
    assert (scene / "dar" / "abc123_mask.tif").is_file()


class TestCommittedIndex:
    """Invariants of the packaged chip index that fixes the benchmark splits."""

    @pytest.fixture(scope="class")
    def committed(self) -> pd.DataFrame:
        return oc.load_index()

    def test_chip_ids_are_unique(self, committed: pd.DataFrame) -> None:
        assert committed.chip_id.is_unique

    def test_windows_are_aligned_and_do_not_overlap(self, committed: pd.DataFrame) -> None:
        assert (committed.col_off % CHIP == 0).all()
        assert (committed.row_off % CHIP == 0).all()
        assert not committed.duplicated(["city", "scene", "col_off", "row_off"]).any()

    def test_each_block_has_one_split(self, committed: pd.DataFrame) -> None:
        assert (committed.groupby("block").split.nunique() == 1).all()

    def test_chips_are_mostly_valid(self, committed: pd.DataFrame) -> None:
        assert (committed.valid_frac >= 0.5).all()

    def test_zanzibar_blocks_have_buildings(self, committed: pd.DataFrame) -> None:
        zanzibar = committed[committed.city == "znz"]
        assert (zanzibar.groupby("block").building_frac.mean() >= 0.01).all()

    def test_every_city_is_in_every_split(self, committed: pd.DataFrame) -> None:
        assert (pd.crosstab(committed.city, committed.split) > 0).all().all()
        assert set(committed.city) == set(oc.CITY_NAME)

    def test_split_sizes_match_the_wrapper(self, committed: pd.DataFrame) -> None:
        assert committed.split.value_counts().to_dict() == oc.OpenCities.split_sizes

    def test_checksums_cover_every_indexed_scene(self, committed: pd.DataFrame) -> None:
        scenes = {f"{c}/{s}" for c, s in zip(committed.city, committed.scene, strict=True)}
        listed = {path.rsplit(".", 1)[0] for path in oc.load_checksums()}
        assert scenes <= listed
