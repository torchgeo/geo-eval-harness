"""Offline tests for embedding footprints, grids, products, PCA and the model."""

import shutil
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.errors import RasterioIOError
from safetensors import safe_open

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import Footprint
from geo_eval_harness.datasets.geobench_v2 import GeoBenchv2, stac_footprint
from geo_eval_harness.embeddings.codecs import CODECS
from geo_eval_harness.embeddings.collection import CollectionWriter, read_metadata
from geo_eval_harness.embeddings.dataset import EmbeddingDataset, embedding_bands
from geo_eval_harness.embeddings.grid import bounds, coarsen, ground_shape, paste, utm_zone
from geo_eval_harness.embeddings.pca import fit_pca, pca_collection
from geo_eval_harness.embeddings.products import (
    AEF,
    OlmoEarth,
    OlmoEarth64,
    Tessera,
    decode_signed_square,
    decode_tessera,
)
from geo_eval_harness.models.embedding import PrecomputedEmbedding
from geo_eval_harness.segmentation_probe import SegmentationProbe
from tests.support.embeddings import UTM, utm_footprint


def _asset(name: str, geotransform: list[float], time: float, *, crs: str = UTM) -> dict:
    return {
        "tortilla:id": name,
        "stac:crs": crs,
        "stac:geotransform": geotransform,
        "stac:raster_shape": [120, 120],
        "stac:time_start": time,
    }


def _v2_split(*assets: dict) -> GeoBenchv2:
    split = GeoBenchv2.__new__(GeoBenchv2)
    split._inner = SimpleNamespace(data_df=SimpleNamespace(read=lambda _: pd.DataFrame(assets)))
    return split


def test_stac_footprint_converts_gdal_order_and_both_time_encodings() -> None:
    row = pd.Series(_asset("s1", [500.0, 10.0, 0.0, 900.0, 0.0, -10.0], 1_506_730_000.0))
    footprint = stac_footprint(row)
    assert footprint == Footprint(UTM, (10.0, 0.0, 500.0, 0.0, -10.0, 900.0), (120, 120), 2017)
    assert stac_footprint(row.copy().replace(1_506_730_000.0, 2016.0)).year == 2016


def test_v2_footprints_use_each_dataset_reference_asset() -> None:
    benv2 = _v2_split(
        _asset("s1", [500.0, 10.0, 0.0, 900.0, 0.0, -10.0], 2017.0),
        _asset("s2", [500.0, 60.0, 0.0, 900.0, 0.0, -60.0], 2017.0),
    )
    assert get_bench_dataset_class("benv2")().footprint(benv2, 0).transform[0] == 10.0

    spacenet = _v2_split(_asset("images", [0.0, 4.0, 0.0, 800.0, 0.0, 4.0], 2018.0))
    flipped = get_bench_dataset_class("spacenet7")().footprint(spacenet, 0)
    assert flipped.transform == (4.0, 0.0, 0.0, 0.0, -4.0, 800.0)

    with pytest.raises(NotImplementedError, match="no geographic footprint"):
        get_bench_dataset_class("m-eurosat")().footprint([], 0)
    with pytest.raises(NotImplementedError, match="no geographic footprint"):
        get_bench_dataset_class("so2sat")().footprint(benv2, 0)
    with pytest.raises(TypeError, match="Expected a dataset"):
        get_bench_dataset_class("benv2")().footprint([], 0)
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
    assert utm_zone(9.5, 48.0) == ("32N", "EPSG:32632")
    assert utm_zone(-71.9, -13.5) == ("19S", "EPSG:32719")


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
    product.max_open_tiles = 1
    product.tile_urls = lambda _: [str(far), str(tile)]  # type: ignore[method-assign]
    out = product.read(Footprint(UTM, (10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0), (16, 16), 2020))
    assert np.allclose(out[0, :, 0], decode_signed_square(rows[::-1, :1][None])[0, :, 0])
    assert list(product._tiles) == [str(tile)]


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


def test_tessera_opens_the_fixed_snapshot_once(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = []

    def open_repository(storage: dict) -> SimpleNamespace:
        opened.append(storage)
        return SimpleNamespace(
            readonly_session=lambda snapshot_id: SimpleNamespace(store=snapshot_id)
        )

    icechunk = SimpleNamespace(
        s3_storage=lambda **storage: storage,
        Repository=SimpleNamespace(open=open_repository),
    )
    zarr = SimpleNamespace(open_group=lambda store, mode: {"32N": (store, mode)})
    monkeypatch.setitem(sys.modules, "icechunk", icechunk)
    monkeypatch.setitem(sys.modules, "zarr", zarr)
    product = Tessera()
    assert product.zone("32N") == product.zone("32N") == (Tessera.snapshot, "r")
    assert len(opened) == 1
    assert opened[0]["bucket"] == Tessera.bucket
    assert opened[0]["anonymous"]


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


def test_olmoearth_opens_the_store_once(monkeypatch: pytest.MonkeyPatch) -> None:
    opened, settings = [], []
    storage = SimpleNamespace(FsspecStore=SimpleNamespace(from_url=lambda url, read_only: url))

    def open_group(store: str, mode: str) -> dict:
        opened.append((store, mode))
        return {"utm32": "zone 32"}

    zarr = SimpleNamespace(
        config=SimpleNamespace(set=settings.append), open_group=open_group, storage=storage
    )
    monkeypatch.setitem(sys.modules, "zarr", zarr)
    monkeypatch.setitem(sys.modules, "zarr.storage", storage)
    product = OlmoEarth()
    assert product.zone(32) == product.zone(32) == "zone 32"
    assert opened == [(OlmoEarth.url, "r")]
    assert settings == [{"array.sharding_coalesce_max_gap_bytes": 0}]


def _write_collection(root: Path, values: dict[str, np.ndarray]) -> Path:
    writer = CollectionWriter(root / "aef" / "toy")
    for split, embeddings in values.items():
        rows = []
        for index, embedding in enumerate(embeddings):
            height, width = embedding.shape[1:]
            rows.append({
                "index": index, "crs": UTM, "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 0.0],
                "height": height, "width": width, "product_year": 2020, "observation_year": 2019,
                "valid_fraction": float(np.isfinite(embedding).all(axis=0).mean()),
                "embedding": embedding.astype(np.float16), "label": np.array(index % 2),
            })  # fmt: skip
        writer.write_split(split, rows)
    return writer.finish({"codec": "float16", "target": "label", "channels": 4})


@pytest.mark.parametrize("codec", ["signed-square-int8", "scaled-int8", "float16"])
def test_codecs_store_downloaded_values_exactly(codec: str) -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(-127, 128, size=(8, 3, 5), dtype=np.int8)
    codes[0] = 127
    values = {
        "signed-square-int8": decode_signed_square(codes),
        "scaled-int8": decode_tessera(codes, rng.uniform(0.01, 0.1, (3, 5)).astype(np.float32)),
        "float16": rng.normal(size=codes.shape),
    }[codec].astype(np.float16)
    values[:, 0, 0] = np.nan
    arrays = CODECS[codec].encode(values)
    assert np.array_equal(CODECS[codec].decode(arrays), values.astype(np.float32), equal_nan=True)
    assert arrays["embedding"].nbytes == values.size * (1 if codec != "float16" else 2)
    if codec != "float16":
        with pytest.raises(ValueError, match="not exactly representable"):
            CODECS[codec].encode(np.full((2, 1, 1), [[[0.3]], [[np.nan]]], dtype=np.float16))


def test_pca_fits_distinct_train_embeddings_and_keeps_missing_cells(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(6, 4, 3, 3)).astype(np.float16).astype(np.float32)
    shifted = train + 100
    shifted[0, :, 0, 0] = np.nan
    _write_collection(tmp_path, {"train": train, "val": shifted, "test": shifted})
    target = pca_collection("aef", "toy", 2, root=tmp_path)
    assert target == tmp_path / "aef" / "toy_pca2"
    val = EmbeddingDataset(target, "val")
    assert val[0]["image"].shape == (2, 3, 3)
    assert int(val[0]["label"]) == 0
    assert torch.isnan(val[0]["image"][:, 0, 0]).all()
    metadata = read_metadata(target)
    assert metadata["pca"]["whiten"] is False
    assert metadata["splits"]["val"]["mean_valid_fraction"] == pytest.approx(53 / 54)
    with safe_open(target / "pca.safetensors", "np") as params:
        assert params.get_tensor("components").shape == (2, 4)

    pca = fit_pca(tmp_path / "aef" / "toy", 2, 1_000, seed=0)
    assert np.allclose(pca.mean_, train.transpose(0, 2, 3, 1).reshape(-1, 4).mean(0), atol=1e-2)
    repeated = np.concatenate([train, np.repeat(train[:1], 20, axis=0)])
    _write_collection(tmp_path / "repeated", {"train": repeated})
    assert np.allclose(fit_pca(tmp_path / "repeated" / "aef" / "toy", 2, 1_000, 0).mean_, pca.mean_)
    with pytest.raises(ValueError, match="valid training cells"):
        fit_pca(tmp_path / "aef" / "toy", 100, 1_000, seed=0)
    with pytest.raises(FileNotFoundError, match="embeddings fetch --product tessera"):
        pca_collection("tessera", "toy", 2, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="No train embeddings"):
        EmbeddingDataset(tmp_path / "tessera" / "toy", "train")


def test_collection_rows_must_share_array_shapes_and_follow_the_index(tmp_path: Path) -> None:
    rows = [
        {"index": i, "crs": UTM, "transform": [1.0] * 6, "height": 1, "width": 1,
         "product_year": 2020, "observation_year": 2020, "valid_fraction": 1.0,
         "embedding": np.zeros((2, 1, size), np.float16), "label": np.array(0)}
        for i, size in ((0, 1), (1, 2))
    ]  # fmt: skip
    with pytest.raises(ValueError, match="Row 1 has arrays"):
        CollectionWriter(tmp_path / "a").write_split("train", rows)
    with pytest.raises(ValueError, match="train row 0 has index 1"):
        CollectionWriter(tmp_path / "b").write_split("train", rows[1:])


def test_model_pools_valid_cells_and_exposes_a_filled_feature_map() -> None:
    model = PrecomputedEmbedding(embedding_bands(3), "aef")
    images = torch.ones(2, 3, 2, 2)
    images[0, :, 0, 0] = float("nan")
    images[1] = float("nan")
    assert torch.equal(model(images), torch.tensor([[1.0] * 3, [0.0] * 3]))
    assert model.effective_normalization == "identity"
    probe = SegmentationProbe(model, ["feature_map"], num_classes=2, head_type="linear")
    assert probe.channels_list == [3]
    assert not torch.isnan(probe(images)).any()
