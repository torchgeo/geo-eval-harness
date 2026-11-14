"""Regression tests for the shared, lightweight dataset catalog."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Literal

import pytest
import yaml

from geo_eval_harness.cli import main
from geo_eval_harness.datasets import (
    BandSpec,
    BenchDataset,
    get_bench_dataset_class,
    get_dataset_task,
    list_datasets,
    loading,
)
from geo_eval_harness.datasets.geobench_v2 import _V2_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import Dataset


@pytest.fixture(params=["classification", "segmentation"])
def registered_dataset(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> type[BenchDataset]:
    class TinyDataset(BenchDataset):
        name = "catalog-test"
        task: Literal["classification", "segmentation"] = request.param
        num_classes = 2
        bands: ClassVar[list[BandSpec]] = [BandSpec("aerial", "gray", "gray", 0, 1, 0, 1)]
        rgb_bands: ClassVar[list[str]] = ["gray"]
        split_sizes: ClassVar[dict[str, int]] = {"train": 1, "val": 1, "test": 1}

        @classmethod
        def data_root(cls) -> Path:
            return Path("data/catalog-test")

        def get_dataset(
            self,
            split: str,
            *,
            partition: str = "default",
            bands: tuple[str, ...] | None = None,
            transform: "Callable | None" = None,
        ) -> "Dataset":
            raise AssertionError("Catalogs and dry runs must not load samples")

    module = ModuleType("geo_eval_harness.datasets._catalog_test")
    module.TinyDataset = TinyDataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        loading._REGISTRY_SPEC,
        TinyDataset.name,
        ("_catalog_test", "TinyDataset", TinyDataset.task),
    )
    return TinyDataset


def test_registered_dataset_appears_in_catalog(
    registered_dataset: type[BenchDataset], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["datasets"])
    names = capsys.readouterr().out.splitlines()
    assert registered_dataset.name in names
    assert names == list_datasets()

    main(["datasets", registered_dataset.name])
    assert yaml.safe_load(capsys.readouterr().out) == {
        "name": registered_dataset.name,
        "task": registered_dataset.task,
    }
    assert get_bench_dataset_class(registered_dataset.name) is registered_dataset


@pytest.mark.parametrize("source", ["flags", "yaml"])
def test_dry_run_accepts_registered_dataset(
    registered_dataset: type[BenchDataset],
    source: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if source == "flags":
        arguments = ["--model", "rcf", "--dataset", registered_dataset.name, "--device", "cpu"]
    else:
        config = tmp_path / "run.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model": {"name": "rcf"},
                    "datasets": [registered_dataset.name],
                    "runtime": {"device": "cpu"},
                }
            ),
            encoding="utf-8",
        )
        arguments = ["--config", str(config)]

    main(["run", *arguments, "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out)["datasets"] == [registered_dataset.name]


@pytest.mark.parametrize("name", list_datasets())
def test_catalog_task_matches_registered_wrapper(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["datasets", name])
    assert yaml.safe_load(capsys.readouterr().out) == {
        "name": name,
        "task": get_bench_dataset_class(name).task,
    }


_CLASSIFICATION = [name for name in list_datasets() if get_dataset_task(name) == "classification"]


@pytest.mark.parametrize("name", _CLASSIFICATION)
def test_classification_class_names_match_num_classes(name: str) -> None:
    cls = get_bench_dataset_class(name)
    assert len(cls.class_names) == cls.num_classes
    assert len(set(cls.class_names)) == cls.num_classes


@pytest.mark.parametrize("name", sorted(set(_CLASSIFICATION) & set(_V2_REGISTRY)))
def test_v2_class_names_follow_upstream_label_order(name: str) -> None:
    import geobench_v2.datasets

    upstream = getattr(geobench_v2.datasets, _V2_REGISTRY[name])
    assert get_bench_dataset_class(name).class_names == tuple(upstream.classes)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["aid", "eurosat", "resisc45", "ucmerced"])
def test_folder_class_names_follow_loaded_label_order(name: str) -> None:
    bench = get_bench_dataset_class(name)
    if not bench.data_root().exists():
        pytest.skip(f"{name} data not found at {bench.data_root()}")
    assert bench.class_names == tuple(bench().get_dataset("test").classes)


def test_m_eurosat_band_labels_match_raw_eurosat_statistics() -> None:
    """GeoBench's shifted HDF5 keys must map to the bands their pixel statistics show."""
    subset = get_bench_dataset_class("m-eurosat").bands
    raw = get_bench_dataset_class("eurosat").bands
    assert [(b.name, b.wavelength_um) for b in subset] == [(b.name, b.wavelength_um) for b in raw]
    for subset_band, raw_band in zip(subset, raw, strict=True):
        assert subset_band.mean == pytest.approx(raw_band.mean, rel=0.1), subset_band.name


def test_unknown_dataset_task_raises() -> None:
    with pytest.raises(KeyError, match="unregistered-dataset"):
        get_dataset_task("unregistered-dataset")


def test_open_cities_public_export_matches_registry() -> None:
    from geo_eval_harness.datasets import OpenCities

    assert OpenCities is get_bench_dataset_class("open_cities")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["run", "--help"],
        ["datasets"],
        ["datasets", "m-eurosat"],
        ["datasets", "burn_scars"],
        ["datasets", "openearthmap-sar"],
        ["datasets", "open_cities"],
        ["datasets", "catalog-test"],
        ["models"],
        ["models", "rcf"],
        ["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu", "--dry-run"],
        ["run", "--model", "rcf", "--dataset", "catalog-test", "--device", "cpu", "--dry-run"],
        ["run", "--model", "rcf", "--dataset", "openearthmap-sar", "--dry-run"],
        ["run", "--model", "rcf", "--dataset", "open_cities", "--dry-run"],
    ],
)
def test_catalog_queries_do_not_import_wrappers_or_ml(arguments: list[str]) -> None:
    code = """
import sys
from geo_eval_harness.cli import main
from geo_eval_harness.datasets.loading import _REGISTRY_SPEC
_REGISTRY_SPEC['catalog-test'] = ('_unimportable', 'TinyDataset', 'segmentation')
try:
    main(sys.argv[1:])
except SystemExit as error:  # allow-except: argparse help exits successfully
    assert error.code == 0
assert not {'torch', 'torchgeo', 'timm', 'numpy', 'pandas', 'geobench_v2'} & sys.modules.keys()
assert not any(name.startswith('geo_eval_harness.models') for name in sys.modules)
assert not any(
    f'geo_eval_harness.datasets.{spec[0]}' in sys.modules
    for spec in _REGISTRY_SPEC.values()
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *arguments],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
