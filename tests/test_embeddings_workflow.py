"""Offline workflow: download a toy product, export, fetch and reduce it, then evaluate it."""

import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from geo_eval_harness import datasets, hub
from geo_eval_harness.cli import main as cli
from geo_eval_harness.embeddings import collection
from geo_eval_harness.embeddings.collection import iter_rows, read_metadata
from geo_eval_harness.embeddings.dataset import EmbeddingDataset
from geo_eval_harness.embeddings.download import download, locality_groups
from geo_eval_harness.embeddings.export import export
from geo_eval_harness.embeddings.fetch import fetch
from geo_eval_harness.embeddings.grid import coarsen
from geo_eval_harness.embeddings.products import PRODUCTS
from geo_eval_harness.main import main
from tests.support.embeddings import ToyBench, ToyProduct, utm_footprint
from tests.support.runner import _compose_cfg

pytestmark = pytest.mark.integration


def test_download_pairs_labels_clamps_years_and_resumes(tmp_path: Path) -> None:
    product, bench = ToyProduct(), ToyBench()
    directory = download(product, bench, root=tmp_path, limit=5)
    assert directory == tmp_path / "aef" / "m-eurosat"
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
    samples = [(0, utm_footprint(far)), (1, utm_footprint(near)), (2, utm_footprint(far + 100)),
               (3, utm_footprint(near + 100)), (4, utm_footprint(far, year=2019))]  # fmt: skip
    groups = [[index for index, _ in group] for group in locality_groups(samples)]
    assert groups == [[4], [1, 3], [0, 2]]

    serial = download(ToyProduct(), ToyBench(), root=tmp_path / "serial", limit=6)
    parallel = download(ToyProduct(), ToyBench(), root=tmp_path / "parallel", limit=6, workers=2)
    files = sorted(path.relative_to(serial) for path in serial.rglob("*.npz"))
    assert files == sorted(path.relative_to(parallel) for path in parallel.rglob("*.npz"))
    for file in files:
        with np.load(serial / file) as a, np.load(parallel / file) as b:
            assert a.files == b.files
            assert all(np.array_equal(a[key], b[key], equal_nan=True) for key in a.files)
    with pytest.raises(ValueError, match="workers must be at least 1"):
        download(ToyProduct(), ToyBench(), root=tmp_path, workers=0)


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_export_loads_exactly_the_download_with_its_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    monkeypatch.setattr(collection, "ROW_GROUP_BYTES", 400)
    monkeypatch.setattr(collection, "SHARD_BYTES", 1_000)
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
    metadata = read_metadata(directory)
    assert metadata["splits"]["train"]["shards"] > 1
    assert metadata["arrays"]["embedding"] == {"dtype": "float16", "shape": [3, 8, 8]}
    if task == "segmentation":
        assert metadata["arrays"]["mask"]["dtype"] == "uint8"

    pilot = export(product, bench, source_root=tmp_path / "downloads", root=tmp_path / "pilot",
                   limits={"train": 3}, workers=2)  # fmt: skip
    assert read_metadata(pilot)["limits"] == {"train": 3}
    assert [read_metadata(pilot)["splits"][s]["samples"] for s in ("train", "val", "test")] == [
        3,
        6,
        6,
    ]
    (source / "val" / "000004.npz").unlink()
    with pytest.raises(ValueError, match="missing 1 of 6 samples"):
        export(product, bench, source_root=tmp_path / "downloads", root=tmp_path / "out")


def test_fetch_verifies_every_file_of_a_pinned_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    download(ToyProduct(), ToyBench(), root=tmp_path / "downloads", limit=3)
    export(ToyProduct(), ToyBench(), source_root=tmp_path / "downloads", root=remote,
           limits={"train": 3, "val": 3, "test": 3})  # fmt: skip
    requests = []

    def snapshot_download(**kwargs: object) -> None:
        requests.append((kwargs["repo_id"], kwargs["revision"], kwargs["allow_patterns"]))
        shutil.copytree(remote / "aef", Path(str(kwargs["local_dir"])), dirs_exist_ok=True)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    revision = "0" * 40
    directory = fetch(
        "aef", "m-eurosat", root=tmp_path / "local", repo_id="me/aef", revision=revision
    )
    assert requests == [("me/aef", revision, ["m-eurosat/*"])]
    assert len(EmbeddingDataset(directory, "test")) == 3

    shard = next((remote / "aef" / "m-eurosat").glob("train-*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        fetch("aef", "m-eurosat", root=tmp_path / "local", repo_id="me/aef", revision=revision)
    with pytest.raises(ValueError, match="full commit hash"):
        fetch("aef", "m-eurosat", repo_id="me/aef", revision="main")
    with pytest.raises(ValueError, match="No published collections for 'aef'"):
        fetch("aef", "m-eurosat")


def test_cli_exports_and_reduces_a_collection_and_rejects_unknown_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PRODUCTS, "toy", ToyProduct)
    monkeypatch.setattr(datasets, "get_bench_dataset_class", lambda name: lambda: ToyBench(name))
    download(ToyProduct(), ToyBench(), root=tmp_path / "downloads")
    cli(["embeddings", "export", "--product", "toy", "--dataset", "m-eurosat", "--limit", "8", "4",
         "4", "--source-root", str(tmp_path / "downloads"), "--root", str(tmp_path)])  # fmt: skip
    cli(["embeddings", "pca", "--product", "aef", "--dataset", "m-eurosat", "--components", "2",
         "--root", str(tmp_path)])  # fmt: skip
    metadata = read_metadata(tmp_path / "aef" / "m-eurosat_pca2")
    assert metadata["channels"] == 2
    assert metadata["pca"]["source"] == "aef/m-eurosat"
    assert metadata["limits"] == {"train": 8, "val": 4, "test": 4}
    assert metadata["splits"]["train"]["samples"] == 8
    with pytest.raises(SystemExit, match="unknown product 'nope'"):
        cli(["embeddings", "download", "--product", "nope", "--dataset", "m-eurosat"])
    with pytest.raises(SystemExit, match="one value or one per split"):
        cli(
            [
                "embeddings",
                "export",
                "--product",
                "toy",
                "--dataset",
                "m-eurosat",
                "--limit",
                "1",
                "2",
            ]
        )


def test_cli_downloads_a_registered_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(PRODUCTS, "toy", ToyProduct)
    monkeypatch.setattr(datasets, "get_bench_dataset_class", lambda name: lambda: ToyBench(name))
    cli(["embeddings", "download", "--product", "toy", "--dataset", "m-eurosat", "--split",
         "train", "--limit", "2", "--root", str(tmp_path)])  # fmt: skip
    assert len(list((tmp_path / "aef" / "m-eurosat" / "train").glob("*.npz"))) == 2
    assert not (tmp_path / "aef" / "m-eurosat" / "val").exists()


def test_run_evaluates_stored_embeddings_like_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for bench in (ToyBench(), ToyBench("fotw", "segmentation")):
        download(ToyProduct(), bench)
        export(ToyProduct(), bench)
    output = tmp_path / "results.csv"
    main(_compose_cfg(output, {"model": {"name": "aef"}, "datasets": ["m-eurosat"]}))
    segmentation = {"head": "linear", "epochs": 30, "learning_rate": 0.1, "batch_size": 4}
    main(
        _compose_cfg(
            output,
            {"model": {"name": "aef"}, "datasets": ["fotw"], "segmentation": segmentation},
        )
    )

    rows = pd.read_csv(output).set_index("method")
    assert set(rows["name"]) == {"aef"}
    assert rows.loc["knn5", "metric_value"] == pytest.approx(1.0)
    assert rows.loc["linear", "metric_value"] == pytest.approx(1.0)
    assert rows.loc["linear", "feature_dim"] == 3
    assert rows.loc["seg-linear", "dataset"] == "fotw"
    assert rows.loc["seg-linear", "normalization"] == "identity"
    assert rows.loc["seg-linear", "metric_value"] > 0.4

    resumed = {"model": {"name": "aef"}, "datasets": ["m-eurosat"], "output": {"resume": True}}
    main(_compose_cfg(output, resumed))
    assert len(pd.read_csv(output)) == len(rows)
    export(ToyProduct(), ToyBench(), limits={"train": 10})
    main(_compose_cfg(output, resumed))
    assert len(pd.read_csv(output)) == len(rows) + 2


def test_run_reports_a_missing_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="embeddings fetch --product tessera"):
        main(_compose_cfg(tmp_path / "out.csv", {"model": {"name": "tessera"}}))
    with pytest.raises(FileNotFoundError, match="'m-eurosat_pca16'"):
        main(_compose_cfg(tmp_path / "out.csv", {"model": {"name": "aef-pca16"}}))
