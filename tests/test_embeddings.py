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

from tests.support.embeddings import UTM, utm_footprint
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.datasets.base import Footprint
from torchgeo_bench.datasets.geobench_v2 import GeoBenchv2, stac_footprint
from torchgeo_bench.embeddings.dataset import EmbeddingDataset, embedding_bands
from torchgeo_bench.embeddings.download import save_sample, write_metadata
from torchgeo_bench.embeddings.grid import bounds, coarsen, ground_shape, paste, utm_zone
from torchgeo_bench.embeddings.pca import fit_pca, pca_collection, project
from torchgeo_bench.embeddings.products import AEF, Tessera, decode_aef, decode_tessera
from torchgeo_bench.models.embedding import PrecomputedEmbedding
from torchgeo_bench.segmentation_probe import SegmentationProbe


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
    aef = decode_aef(codes)
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
    assert np.allclose(out[0, :, 0], decode_aef(rows[::-1, :1][None])[0, :, 0])
    assert list(product._tiles) == [str(tile)]


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


def _write_collection(root: Path, values: dict[str, np.ndarray]) -> Path:
    directory = root / "aef" / "toy"
    for split, embedding in values.items():
        for index, sample in enumerate(embedding):
            save_sample(
                directory / split / f"{index:06d}.npz",
                embedding=sample.astype(np.float16),
                label=np.array(index % 2),
                year=2020,
                valid_fraction=1.0,
            )
    write_metadata(directory, {"product": "aef", "dataset": "toy", "channels": 4})
    return directory


def test_pca_fits_train_cells_only_and_keeps_missing_cells(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(6, 4, 3, 3)).astype(np.float32)
    shifted = train + 100
    shifted[0, :, 0, 0] = np.nan
    _write_collection(tmp_path, {"train": train, "val": shifted, "test": shifted})
    target = pca_collection("aef", "toy", 2, root=tmp_path)
    assert target == tmp_path / "aef-pca2" / "toy"
    with np.load(target / "val" / "000000.npz") as sample:
        assert sample["embedding"].shape == (2, 3, 3)
        assert int(sample["label"]) == 0
        assert np.isnan(sample["embedding"][:, 0, 0]).all()
    files = sorted((tmp_path / "aef" / "toy" / "train").glob("*.npz"))
    pca = fit_pca(files, 2, 1_000, seed=0)
    assert np.allclose(pca.mean_, train.transpose(0, 2, 3, 1).reshape(-1, 4).mean(0), atol=1e-2)
    assert np.allclose(project(train[0], pca), project(train[0], fit_pca(files, 2, 1_000, 0)))
    with pytest.raises(ValueError, match="valid training cells"):
        fit_pca(files, 100, 1_000, seed=0)
    with pytest.raises(FileNotFoundError, match="No training embeddings"):
        pca_collection("tessera", "toy", 2, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="No train embeddings"):
        EmbeddingDataset(tmp_path / "tessera" / "toy", "train")


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
