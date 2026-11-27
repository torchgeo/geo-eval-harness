"""Offline tests for embedding footprints, grids and products."""

import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.errors import RasterioIOError

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import Footprint
from geo_eval_harness.datasets.geobench_v2 import stac_footprint
from geo_eval_harness.embeddings.grid import bounds, coarsen, ground_shape, paste, utm_zone
from geo_eval_harness.embeddings.products import (
    AEF,
    OlmoEarth,
    OlmoEarth64,
    Tessera,
    decode_signed_square,
    decode_tessera,
)
from tests.support.embeddings import UTM, utm_footprint


def _asset(name: str, geotransform: list[float], time: float, *, crs: str = UTM) -> dict:
    return {
        "tortilla:id": name,
        "stac:crs": crs,
        "stac:geotransform": geotransform,
        "stac:raster_shape": [120, 120],
        "stac:time_start": time,
    }


class _V2Split:
    def __init__(self, *assets: dict) -> None:
        self._assets = pd.DataFrame(assets)

    def assets(self, index: int) -> pd.DataFrame:
        return self._assets


def test_stac_footprint_converts_gdal_order_and_both_time_encodings() -> None:
    row = pd.Series(_asset("s1", [500.0, 10.0, 0.0, 900.0, 0.0, -10.0], 1_506_730_000.0))
    footprint = stac_footprint(row)
    assert footprint == Footprint(UTM, (10.0, 0.0, 500.0, 0.0, -10.0, 900.0), (120, 120), 2017)
    assert stac_footprint(row.copy().replace(1_506_730_000.0, 2016.0)).year == 2016


def test_v2_footprints_use_each_dataset_reference_asset() -> None:
    benv2 = _V2Split(
        _asset("s1", [500.0, 10.0, 0.0, 900.0, 0.0, -10.0], 2017.0),
        _asset("s2", [500.0, 60.0, 0.0, 900.0, 0.0, -60.0], 2017.0),
    )
    assert get_bench_dataset_class("benv2")().footprint(benv2, 0).transform[0] == 10.0

    spacenet = _V2Split(_asset("images", [0.0, 4.0, 0.0, 800.0, 0.0, 4.0], 2018.0))
    flipped = get_bench_dataset_class("spacenet7")().footprint(spacenet, 0)
    assert flipped.transform == (4.0, 0.0, 0.0, 0.0, -4.0, 800.0)

    with pytest.raises(NotImplementedError, match="no geographic footprint"):
        get_bench_dataset_class("m-eurosat")().footprint([], 0)
    with pytest.raises(NotImplementedError, match="no geographic footprint"):
        get_bench_dataset_class("so2sat")().footprint(benv2, 0)
    with pytest.raises(ValueError, match="no unique 'planet_0'"):
        get_bench_dataset_class("dynamic_earthnet")().footprint(spacenet, 0)


def test_grid_keeps_extent_and_row_order_when_coarsened() -> None:
    north_up = Footprint(UTM, (5.0, 0.0, 1000.0, 0.0, -5.0, 2000.0), (40, 20), 2020)
    south_up = Footprint(UTM, (5.0, 0.0, 1000.0, 0.0, 5.0, 1800.0), (40, 20), 2020)
    assert bounds(north_up) == bounds(south_up) == (1000.0, 1800.0, 1100.0, 2000.0)
    assert ground_shape(north_up) == (20, 10)
    coarse = coarsen(south_up, (20, 10))
    assert coarse.transform == (10.0, 0.0, 1000.0, 0.0, 10.0, 1800.0)
    assert bounds(coarse) == bounds(south_up)
    assert utm_zone(9.5, 48.0) == (32, "32N", "EPSG:32632")
    assert utm_zone(-71.9, -13.5) == (19, "19S", "EPSG:32719")


def test_paste_fills_missing_cells_from_successive_tiles() -> None:
    grid = utm_footprint(1000.0)
    out = np.full((1, 8, 8), np.nan, dtype=np.float32)
    west = np.arange(4, dtype=np.float32).reshape(1, 1, 4).repeat(8, axis=1)
    west[0, 0, 0] = np.nan
    paste(out, grid, west, UTM, Affine(10, 0, 1000, 0, -10, 5_500_000))
    assert np.array_equal(out[0, 1, :4], [0, 1, 2, 3])
    assert np.isnan(out[0, 0, 0])
    assert np.isnan(out[0, :, 4:]).all()
    paste(out, grid, np.full((1, 8, 8), 9, np.float32), UTM, Affine(10, 0, 1000, 0, -10, 5_500_000))
    assert out[0, 0, 0] == 9
    assert out[0, 1, 0] == 0
    assert (out[0, :, 4:] == 9).all()


def test_decoders_mark_nodata_as_nan() -> None:
    codes = np.array([[[-128, 127]], [[0, -64]]], dtype=np.int8)
    aef = decode_signed_square(codes)
    assert np.isnan(aef[:, 0, 0]).all()
    assert np.allclose(aef[:, 0, 1], [(127 / 127.5) ** 2, -((64 / 127.5) ** 2)])
    tessera = decode_tessera(codes, np.array([[0.0, 0.5]], dtype=np.float32))
    assert np.isnan(tessera[:, 0, 0]).all()
    assert np.allclose(tessera[:, 0, 1], [63.5, -32.0])


def test_aef_reads_a_south_up_tile_onto_a_north_up_grid(tmp_path: Path) -> None:
    rows = np.arange(16, dtype=np.int8).reshape(16, 1).repeat(16, axis=1)
    codes = np.repeat(rows[None], 64, axis=0)
    tile, far = tmp_path / "tile.tif", tmp_path / "far.tif"
    profile = {"driver": "GTiff", "count": 64, "dtype": "int8", "crs": UTM, "nodata": -128}
    south_up = Affine(10, 0, 1000, 0, 10, 4840)
    with rasterio.open(tile, "w", width=16, height=16, transform=south_up, **profile) as dst:
        dst.write(codes)
    with rasterio.open(far, "w", width=16, height=16, transform=Affine(10, 0, 9e5, 0, 10, 0),
                       **profile) as dst:  # fmt: skip
        dst.write(codes)
    product = AEF(root=tmp_path)
    product.tile_urls = lambda _: [str(far), str(tile)]  # type: ignore[method-assign]
    out = product.read(Footprint(UTM, (10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0), (16, 16), 2020))
    assert np.allclose(out[0, :, 0], decode_signed_square(rows[::-1, :1][None])[0, :, 0])


class _FailingTile:
    """Wrap an open tile so its reads fail like an interrupted S3 range request."""

    def __init__(self, dataset: rasterio.DatasetReader) -> None:
        self._dataset = dataset

    def __getattr__(self, name: str) -> object:
        return getattr(self._dataset, name)

    def read(self, **_: object) -> np.ndarray:
        raise RasterioIOError("Read failed")


@pytest.mark.parametrize("failures", [1, AEF.read_attempts])
def test_aef_reopens_a_tile_after_a_failed_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    codes = np.arange(64, dtype=np.int8).reshape(64, 1, 1).repeat(4, axis=1).repeat(4, axis=2)
    tile = tmp_path / "tile.tif"
    with rasterio.open(tile, "w", driver="GTiff", count=64, dtype="int8", crs=UTM, nodata=-128,
                       width=4, height=4, transform=Affine(10, 0, 0, 0, -10, 40)) as dst:  # fmt: skip
        dst.write(codes)
    opened = []

    def flaky_open(path: str) -> object:
        dataset = real_open(path)
        opened.append(dataset)
        return _FailingTile(dataset) if len(opened) <= failures else dataset

    real_open = rasterio.open
    monkeypatch.setattr(rasterio, "open", flaky_open)
    product = AEF(root=tmp_path)
    product.retry_delay = 0.0
    product.tile_urls = lambda _: [str(tile)]  # type: ignore[method-assign]
    grid = Footprint(UTM, (10.0, 0.0, 0.0, 0.0, -10.0, 40.0), (4, 4), 2020)
    if failures == AEF.read_attempts:
        with pytest.raises(RasterioIOError):
            product.read(grid)
    else:
        assert np.allclose(product.read(grid), decode_signed_square(codes))
    assert all(dataset.closed for dataset in opened[:-1])
    assert len(opened) == min(failures + 1, AEF.read_attempts)


def test_aef_downloads_the_index_once_and_selects_overlapping_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.parquet"
    pd.DataFrame(
        {
            "path": ["s3://bucket/hit.tif", "s3://bucket/other-year.tif", "s3://bucket/far.tif"],
            "year": [2020, 2019, 2020],
            "wgs84_west": [8.0, 8.0, 20.0],
            "wgs84_south": [49.0, 49.0, 49.0],
            "wgs84_east": [10.0, 10.0, 21.0],
            "wgs84_north": [50.0, 50.0, 50.0],
        }
    ).to_parquet(remote)
    fetched = []

    def fetch(url: str, path: Path) -> None:
        fetched.append(url)
        shutil.copy(remote, path)

    monkeypatch.setattr(urllib.request, "urlretrieve", fetch)
    grid = utm_footprint(500_000.0)
    for _ in range(2):
        assert AEF(root=tmp_path / "cache").tile_urls(grid) == [AEF.base_url + "bucket/hit.tif"]
    assert fetched == [AEF.index_url]


def test_tessera_reads_the_year_window_of_the_centre_zone() -> None:
    codes = np.zeros((2, 16, 16, 128), dtype=np.int8)
    codes[1, 4:12, 4:12, :] = 2

    class Zone(dict):
        attrs: ClassVar = {
            "crs": UTM,
            "spatial:transform": [10, 0, 0, 0, -10, 160],
            "spatial:shape": [16, 16],
        }

    zone = Zone(
        time=np.array(["2019-01-01", "2020-01-01"], dtype="datetime64[ns]").astype(np.int64),
        embeddings=codes,
        scales=np.ones((2, 16, 16), dtype=np.float32),
    )
    requested = []
    product = Tessera()
    product.zone = lambda name: requested.append(name) or zone  # type: ignore[method-assign]
    grid = Footprint(UTM, (10.0, 0.0, 40.0, 0.0, -10.0, 120.0), (8, 8), 2020)
    assert (product.read(grid) == 2).all()
    assert requested == ["31N"]
    assert np.isnan(product.read(replace(grid, year=2017))).all()
    outside = replace(grid, transform=(10.0, 0.0, 100_000.0, 0.0, -10.0, 120.0))
    assert np.isnan(product.read(outside)).all()


def test_olmoearth_reads_the_year_and_band_prefix_of_the_centre_zone() -> None:
    codes = np.full((2, 128, 16, 16), -128, dtype=np.int8)
    codes[1, :, 4:12, 4:12] = np.arange(128, dtype=np.int8)[:, None, None] - 64

    class Zone(dict):
        attrs: ClassVar = {
            "proj:code": UTM,
            "spatial:transform": [10, 0, 0, 0, -10, 160],
            "spatial:shape": [16, 16],
        }

    zone = Zone(time=np.array([2019, 2020], dtype=np.int32), embeddings=codes)
    requested = []
    full, prefix = OlmoEarth(), OlmoEarth64()
    for product in (full, prefix):
        product.zone = lambda number: requested.append(number) or zone  # type: ignore[method-assign]
    grid = Footprint(UTM, (10.0, 0.0, 40.0, 0.0, -10.0, 120.0), (8, 8), 2020)
    out = full.read(grid)
    expected = decode_signed_square(codes[1, :, 4:12, 4:12])
    assert np.array_equal(out, expected)
    assert np.array_equal(prefix.read(grid), out[:64])
    assert requested == [31, 31]
    assert np.isnan(full.read(replace(grid, year=2017))).all()
    outside = replace(grid, transform=(10.0, 0.0, 100_000.0, 0.0, -10.0, 120.0))
    assert np.isnan(full.read(outside)).all()
    nodata = full.read(replace(grid, transform=(10.0, 0.0, 0.0, 0.0, -10.0, 160.0)))
    assert np.isnan(nodata[:, :4, :4]).all()
    assert np.isfinite(nodata[:, 4:, 4:]).all()
