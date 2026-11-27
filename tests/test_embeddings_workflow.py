"""Offline workflow: retrieve a toy product for dataset samples and arbitrary footprints."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from geo_eval_harness import datasets
from geo_eval_harness.cli import main as cli
from geo_eval_harness.datasets.base import Footprint
from geo_eval_harness.embeddings.download import download, locality_groups, retrieve
from geo_eval_harness.embeddings.footprints import read_footprints
from geo_eval_harness.embeddings.products import PRODUCTS
from tests.support.embeddings import UTM, ToyBench, ToyProduct, utm_footprint

pytestmark = pytest.mark.integration


def test_download_pairs_labels_clamps_years_and_resumes(tmp_path: Path) -> None:
    product, bench = ToyProduct(), ToyBench()
    directory = download(product, bench, root=tmp_path, limit=5)
    assert directory == tmp_path / "toy" / "m-eurosat"
    with np.load(directory / "train" / "000003.npz") as sample:
        assert sample["embedding"].dtype == np.float16
        assert sample["embedding"].shape == (3, 8, 8)
        assert int(sample["label"]) == 1
        assert int(sample["year"]) == 2021
    assert [grid.year for grid in product.reads[:5]] == [2019, 2019, 2020, 2021, 2021]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["channels"] == 3
    assert metadata["splits"]["val"]["samples"] == 5
    assert metadata["splits"]["test"]["mean_valid_fraction"] == pytest.approx(63 / 64)

    reads = len(product.reads)
    download(product, bench, root=tmp_path)
    assert len(product.reads) == reads + 7 + 1 + 1
    assert json.loads((directory / "metadata.json").read_text())["splits"]["train"]["samples"] == 12


def test_download_reads_neighbours_together_in_any_worker_count(tmp_path: Path) -> None:
    far, near = 600_000.0, 400_000.0
    samples = [("0", utm_footprint(far)), ("1", utm_footprint(near)), ("2", utm_footprint(far + 100)),
               ("3", utm_footprint(near + 100)), ("4", utm_footprint(far, year=2019))]  # fmt: skip
    groups = [[key for key, _ in group] for group in locality_groups(samples)]
    assert groups == [["4"], ["1", "3"], ["0", "2"]]
    degrees = Footprint("EPSG:4326", (0.001, 0.0, 9.0, 0.0, -0.001, 48.0), (8, 8), 2020)
    shifted = replace(degrees, transform=(0.001, 0.0, 9.5, 0.0, -0.001, 48.0))
    assert len(locality_groups([("a", degrees), ("b", shifted)])) == 2

    serial = download(ToyProduct(), ToyBench(), root=tmp_path / "serial", limit=6)
    parallel = download(ToyProduct(), ToyBench(), root=tmp_path / "parallel", limit=6, workers=2)
    files = sorted(path.relative_to(serial) for path in serial.rglob("*.npz"))
    assert files == sorted(path.relative_to(parallel) for path in parallel.rglob("*.npz"))
    for file in files:
        with np.load(serial / file) as a, np.load(parallel / file) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f")


def test_cli_downloads_a_registered_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PRODUCTS, "toy", ToyProduct)
    monkeypatch.setattr(datasets, "get_bench_dataset_class", lambda name: lambda: ToyBench(name))
    cli(["embeddings", "download", "--product", "toy", "--dataset", "m-eurosat", "--split",
         "train", "--limit", "2", "--root", str(tmp_path)])  # fmt: skip
    assert len(list((tmp_path / "toy" / "m-eurosat" / "train").glob("*.npz"))) == 2
    assert not (tmp_path / "toy" / "m-eurosat" / "val").exists()
    with pytest.raises(SystemExit, match="unknown product 'nope'"):
        cli(["embeddings", "download", "--product", "nope", "--dataset", "m-eurosat"])
    with pytest.raises(SystemExit, match=r"^2$"):
        cli(["embeddings", "download", "--product", "toy", "--dataset", "m-eurosat", "--workers", "0"])  # fmt: skip


def test_cli_retrieves_table_footprints_on_their_own_grids_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "footprints.csv"
    pd.DataFrame(
        {
            "id": ["field-a", "field-b"],
            "crs": [UTM, UTM],
            "west": [500_000.0, 600_000.0],
            "south": [5_499_960.0, 5_499_800.0],
            "east": [500_080.0, 600_200.0],
            "north": [5_500_000.0, 5_500_000.0],
            "year": [2016, 2020],
        }
    ).to_csv(table, index=False)
    products: list[ToyProduct] = []
    monkeypatch.setitem(PRODUCTS, "toy", lambda: products.append(ToyProduct()) or products[-1])
    out = tmp_path / "out"
    command = ["embeddings", "retrieve", "--product", "toy", "--footprints", str(table),
               "--output", str(out)]  # fmt: skip
    cli(command)
    with np.load(out / "field-a.npz") as sample:
        assert sample["embedding"].shape == (3, 4, 8)
        assert str(sample["crs"]) == UTM
        assert tuple(sample["transform"]) == (10.0, 0.0, 500_000.0, 0.0, -10.0, 5_500_000.0)
        assert int(sample["year"]) == 2019
        assert "label" not in sample.files
    with np.load(out / "field-b.npz") as sample:
        assert sample["embedding"].shape == (3, 20, 20)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["samples"] == 2
    assert metadata["cell_size"] == 10.0

    cli(command)
    assert products[-1].reads == []
    with pytest.raises(SystemExit, match="--year applies only to --rasters"):
        cli([*command, "--year", "2020"])


def test_cli_retrieves_raster_footprints_at_a_chosen_cell_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rasters = tmp_path / "rasters"
    rasters.mkdir()
    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "crs": UTM,
        "width": 40,
        "height": 20,
    }
    with rasterio.open(rasters / "scene.tif", "w", transform=Affine(2, 0, 1000, 0, -2, 5000),
                       **profile) as dst:  # fmt: skip
        dst.write(np.zeros((1, 20, 40), dtype=np.uint8))
    monkeypatch.setitem(PRODUCTS, "toy", ToyProduct)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="--rasters needs --year"):
        cli(["embeddings", "retrieve", "--product", "toy", "--rasters", str(rasters),
             "--output", str(out)])  # fmt: skip
    cli(["embeddings", "retrieve", "--product", "toy", "--rasters", str(rasters), "--year", "2020",
         "--cell-size", "20", "--output", str(out)])  # fmt: skip
    with np.load(out / "scene.npz") as sample:
        assert sample["embedding"].shape == (3, 2, 4)
        assert tuple(sample["transform"]) == (20.0, 0.0, 1000.0, 0.0, -20.0, 5000.0)


def test_retrieve_rejects_unusable_footprints(tmp_path: Path) -> None:
    footprint = utm_footprint(500_000.0)
    with pytest.raises(ValueError, match="unique"):
        retrieve(ToyProduct(), [("a", footprint), ("A", footprint)], tmp_path)
    retrieve(ToyProduct(), [("a", footprint)], tmp_path / "out")
    with pytest.raises(ValueError, match="holds other embeddings"):
        retrieve(ToyProduct(), [("a", footprint)], tmp_path / "out", cell_size=20.0)
    with pytest.raises(ValueError, match="safe file names"):
        retrieve(ToyProduct(), [("../a", footprint)], tmp_path)
    table = tmp_path / "footprints.csv"
    pd.DataFrame({"id": ["a"], "crs": [UTM], "west": [0.0]}).to_csv(table, index=False)
    with pytest.raises(ValueError, match="lacks the footprint columns"):
        read_footprints(table)
