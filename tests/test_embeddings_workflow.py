"""Offline workflow: download a toy product, reduce it with PCA and evaluate it with ``run``."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_eval_harness import datasets
from geo_eval_harness.cli import main as cli
from geo_eval_harness.embeddings.download import download, locality_groups
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


def test_cli_reduces_a_collection_and_rejects_unknown_products(tmp_path: Path) -> None:
    download(ToyProduct(), ToyBench(), root=tmp_path)
    cli(["embeddings", "pca", "--product", "aef", "--dataset", "m-eurosat", "--components", "2",
         "--root", str(tmp_path)])  # fmt: skip
    metadata = json.loads((tmp_path / "aef-pca2" / "m-eurosat" / "metadata.json").read_text())
    assert metadata["channels"] == 2
    assert metadata["pca"]["source"] == "aef"
    assert metadata["splits"]["train"]["samples"] == 12
    with pytest.raises(SystemExit, match="unknown product 'nope'"):
        cli(["embeddings", "download", "--product", "nope", "--dataset", "m-eurosat"])


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
    download(ToyProduct(), ToyBench())
    download(ToyProduct(), ToyBench("fotw", "segmentation"))
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


def test_run_reports_a_missing_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="embeddings download --product tessera"):
        main(_compose_cfg(tmp_path / "out.csv", {"model": {"name": "tessera"}}))
