"""Offline workflow: retrieve a toy product, then export, fetch and reduce its collection."""

import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from affine import Affine

import geo_eval_harness.embeddings.fetch as fetch_module
from geo_eval_harness import datasets, hub
from geo_eval_harness.cli import main as cli
from geo_eval_harness.datasets.base import Footprint
from geo_eval_harness.embeddings.collection import iter_rows, read_metadata
from geo_eval_harness.embeddings.dataset import EmbeddingDataset
from geo_eval_harness.embeddings.download import download, locality_groups, retrieve
from geo_eval_harness.embeddings.export import export
from geo_eval_harness.embeddings.fetch import fetch
from geo_eval_harness.embeddings.footprints import read_footprints
from geo_eval_harness.embeddings.grid import coarsen
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


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_export_loads_exactly_the_download_with_its_grid(tmp_path: Path, task: str) -> None:
    product, bench = ToyProduct(), ToyBench("fotw", task)
    source = download(product, bench, root=tmp_path / "downloads")
    directory = export(product, bench, source_root=tmp_path / "downloads", root=tmp_path / "out")
    target = "mask" if task == "segmentation" else "label"
    for split in ("train", "val", "test"):
        loaded = pickle.loads(pickle.dumps(EmbeddingDataset(directory, split)))
        assert len(loaded) == bench.split_sizes[split]
        rows = list(iter_rows(directory, split))
        split_dataset = bench.get_dataset(split)
        for index in range(len(loaded)):
            item = loaded[index]
            with np.load(source / split / f"{index:06d}.npz") as sample:
                assert np.array_equal(
                    item["image"], sample["embedding"].astype(np.float32), equal_nan=True
                )
                assert np.array_equal(item[target], sample[target])
                assert item[target].dtype == torch.from_numpy(sample[target]).dtype
            footprint = bench.footprint(split_dataset, index)
            grid = coarsen(replace(footprint, year=product.product_year(footprint.year)), (8, 8))
            assert (rows[index]["crs"], tuple(rows[index]["transform"])) == (
                grid.crs,
                grid.transform,
            )
            assert (rows[index]["product_year"], rows[index]["observation_year"]) == (
                grid.year,
                footprint.year,
            )  # fmt: skip
    (source / "val" / "000004.npz").unlink()
    with pytest.raises(ValueError, match="missing 1 of 6 samples"):
        export(product, bench, source_root=tmp_path / "downloads", root=tmp_path / "out")


def test_fetch_verifies_a_pinned_collection_against_its_published_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    download(ToyProduct(), ToyBench(), root=tmp_path / "downloads", limit=3)
    export(ToyProduct(), ToyBench(), source_root=tmp_path / "downloads", root=remote,
           limits={"train": 3, "val": 3, "test": 3})  # fmt: skip
    requests = []

    def snapshot_download(**kwargs: object) -> None:
        requests.append((kwargs["repo_id"], kwargs["revision"], kwargs["allow_patterns"]))
        shutil.copytree(remote / "toy", Path(str(kwargs["local_dir"])), dirs_exist_ok=True)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    revision = "0" * 40
    directory = fetch(
        "toy", "m-eurosat", root=tmp_path / "local", repo_id="me/toy", revision=revision
    )
    assert requests == [("me/toy", revision, ["m-eurosat/*"])]
    assert len(EmbeddingDataset(directory, "test")) == 3
    with pytest.raises(ValueError, match="repo_id and revision go together"):
        fetch("toy", "m-eurosat", repo_id="me/toy")

    sums = remote / "toy" / "m-eurosat" / "SHA256SUMS"
    monkeypatch.setitem(fetch_module.PUBLISHED, "toy", ("me/toy", revision))
    digest = hub.sha256_file(sums)
    monkeypatch.setattr(fetch_module, "load_checksums", lambda _: {"m-eurosat/SHA256SUMS": digest})
    assert len(EmbeddingDataset(fetch("toy", "m-eurosat", root=tmp_path / "pinned"), "val")) == 3
    with pytest.raises(ValueError, match="'benv2' is not published for 'toy'"):
        fetch("toy", "benv2", root=tmp_path / "pinned")
    sums.write_bytes(sums.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        fetch("toy", "m-eurosat", root=tmp_path / "pinned")


def test_cli_exports_and_reduces_a_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PRODUCTS, "toy", ToyProduct)
    monkeypatch.setattr(datasets, "get_bench_dataset_class", lambda name: lambda: ToyBench(name))
    download(ToyProduct(), ToyBench(), root=tmp_path / "downloads")
    cli(["embeddings", "export", "--product", "toy", "--dataset", "m-eurosat", "--limit", "8", "4",
         "4", "--workers", "2", "--source-root", str(tmp_path / "downloads"),
         "--root", str(tmp_path)])  # fmt: skip
    cli(["embeddings", "pca", "--product", "toy", "--dataset", "m-eurosat", "--components", "2",
         "--root", str(tmp_path)])  # fmt: skip
    metadata = read_metadata(tmp_path / "toy" / "m-eurosat_pca2")
    assert metadata["channels"] == 2
    assert metadata["pca"]["source"] == "toy/m-eurosat"
    assert metadata["limits"] == {"train": 8, "val": 4, "test": 4}
    assert metadata["splits"]["train"]["samples"] == 8
    with pytest.raises(SystemExit, match="one value or one per split"):
        cli(["embeddings", "export", "--product", "toy", "--dataset", "m-eurosat", "--limit", "1",
             "2"])  # fmt: skip
