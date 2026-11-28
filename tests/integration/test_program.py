"""Run the canonical CLI against tiny on-disk inputs without mocking its internals."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.run import RunConfig
from tests.support.data import write_classification_files

pytestmark = pytest.mark.integration


@pytest.fixture
def classification_files(tmp_path: Path) -> Path:
    return write_classification_files(tmp_path, "m-eurosat", (2, 7))


def classification_arguments(
    output: Path,
    *,
    dataset_names: tuple[str, ...] = ("m-eurosat",),
    extra_settings: dict[str, Any] | None = None,
) -> list[str]:
    """Use a small real feature extractor and the real KNN/linear implementations."""
    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"features": 8}},
            "datasets": list(dataset_names),
            "input": {"image_size": 16},
            "runtime": {"batch_size": 8, "workers": 0, "device": "cpu"},
            "classification": {
                "bootstrap_samples": 5,
                "linear": {"c_log10_start": 1.0, "c_log10_stop": 2.0, "c_count": 2},
            },
            **(extra_settings or {}),
        }
    )
    path = output.with_suffix(".yaml")
    path.write_text(yaml.safe_dump(config.model_dump_yaml()))
    return [
        "run",
        "--config",
        str(path),
        "--output",
        str(output),
    ]


def test_saved_embeddings_align_across_models(
    tmp_path: Path, classification_files: Path, cli_main: Callable[..., None]
) -> None:
    tables, rows = [], []
    for features in (8, 16):
        output = tmp_path / f"rcf-{features}.csv"
        settings = {
            "model": {"name": "rcf", "kwargs": {"features": features}},
            "output": {"embeddings": "embeddings"},
        }
        cli_main(classification_arguments(output, extra_settings=settings))
        row = pd.read_csv(output, dtype={"config_hash": str}).iloc[0]
        rows.append(row)
        path = tmp_path / "embeddings" / "rcf" / f"m-eurosat-{row['config_hash']}.parquet"
        tables.append(pq.read_table(path))

    expected_ids = [f"val/{index}" for index in range(rows[0]["n_val"])]
    expected_ids += [f"test/{index}" for index in range(rows[0]["n_test"])]
    assert [table["sample_id"].to_pylist() for table in tables] == [expected_ids, expected_ids]
    assert tables[0]["label"].to_pylist() == tables[1]["label"].to_pylist()
    assert [table["embedding"].type.list_size for table in tables] == [
        row["feature_dim"] for row in rows
    ]


@pytest.mark.parametrize("temperature_scaling", [False, True])
def test_classification_program_handles_noncontiguous_labels(
    tmp_path: Path,
    classification_files: Path,
    cli_main: Callable[..., None],
    *,
    temperature_scaling: bool,
) -> None:
    output = tmp_path / "classification=1.csv"
    arguments = classification_arguments(output)
    if temperature_scaling:
        arguments.extend(["--no-refit-train-val", "--temp-scale"])
    cli_main(arguments)

    rows = pd.read_csv(output).set_index("method")
    assert set(rows.index) == {"knn5", "linear"}
    assert (rows["dataset"] == "m-eurosat").all()
    assert (rows["metric_name"] == "accuracy").all()
    assert (rows["metric_value"] == 1.0).all()
    assert rows.loc["knn5", "ece"] == 0
    assert rows.loc["linear", "ece"] < 0.2
    if temperature_scaling:
        assert np.isfinite(rows.loc["linear", "temperature"])
        assert rows.loc["linear", "temperature"] > 0
        assert rows.loc["linear", "ece_ts"] < 0.2
    else:
        assert rows["temperature"].isna().all()
    assert not (tmp_path / "results" / "profiles").exists()
    assert not (tmp_path / "results" / "intrinsic_dim").exists()


def test_program_profiles_features_and_resumes_without_input_files(
    tmp_path: Path, classification_files: Path, cli_main: Callable[..., None]
) -> None:
    output = tmp_path / "measurements.csv"
    arguments = classification_arguments(
        output,
        extra_settings={
            "intrinsic_dim": {"enabled": True, "estimators": []},
            "profile": {"enabled": True, "n_warmup": 0, "n_measure": 1},
        },
    )
    cli_main(arguments)
    rows = pd.read_csv(output)
    assert set(rows["method"]) == {"knn5", "linear", "profile", "intrinsic_dim"}
    assert np.isfinite(rows["metric_value"]).all()
    assert rows.loc[rows["metric_name"] == "gflops", "metric_value"].item() > 0
    assert (rows["n_train"] == 24).all()
    assert (rows["n_val"] == 8).all()
    assert (rows["n_test"] == 8).all()

    before = output.read_bytes()
    shutil.rmtree(classification_files)
    cli_main([*arguments, "--resume"])
    assert output.read_bytes() == before


def test_program_reinitializes_for_multispectral_and_multilabel_datasets(
    tmp_path: Path, cli_main: Callable[..., None]
) -> None:
    write_classification_files(tmp_path, "m-eurosat", (2, 7), all_bands=True)
    labels = tuple([int(index == label) for index in range(43)] for label in (0, 1))
    write_classification_files(tmp_path, "m-bigearthnet", labels, all_bands=True)
    output = tmp_path / "multiple.csv"
    cli_main(
        [
            *classification_arguments(output),
            "--dataset",
            "m-eurosat",
            "--dataset",
            "m-bigearthnet",
            "--bands",
            "all",
        ]
    )
    rows = pd.read_csv(output)
    assert len(rows) == 4
    for dataset, metric in (("m-eurosat", "accuracy"), ("m-bigearthnet", "micro_mAP")):
        selected = rows[rows["dataset"] == dataset]
        assert set(selected["method"]) == {"knn5", "linear"}
        assert (selected["metric_name"] == metric).all()
        assert (selected["metric_value"] > 0.99).all()
    assert (rows["bands"] == "all").all()
    assert np.isfinite(rows["metric_value"]).all()


@pytest.mark.parametrize(
    "dataset_names",
    [("m-forestnet", "m-eurosat"), ("m-eurosat", "m-forestnet")],
)
def test_program_fails_when_any_requested_data_is_unavailable(
    tmp_path: Path,
    classification_files: Path,
    cli_main: Callable[..., None],
    dataset_names: tuple[str, ...],
) -> None:
    output = tmp_path / "missing.csv"
    message = (
        r"Required files for 'm-forestnet' are missing\. "
        "Run `geo-eval-harness download geobench_v1 --datasets m-forestnet`"
    )
    with pytest.raises(SystemExit, match=message):
        cli_main(classification_arguments(output, dataset_names=dataset_names))
    if dataset_names[0] == "m-eurosat":
        rows = pd.read_csv(output)
        assert set(rows["method"]) == {"knn5", "linear"}
        assert (rows["dataset"] == "m-eurosat").all()
    else:
        assert not output.exists()


def test_flops_program_writes_both_band_configurations_and_resumes(
    tmp_path: Path, cli_main: Callable[..., None]
) -> None:
    output = tmp_path / "flops=1.csv"
    config = FlopsConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"features": 8}},
            "runtime": {"device": "cpu"},
            "input": {"image_size": 16},
            "timing": {"batch_size": 2, "n_warmup": 0, "n_measure": 1},
            "segmentation": {"band_configs": []},
            "output": {"resume": False},
        }
    )
    path = tmp_path / "flops.yaml"
    path.write_text(yaml.safe_dump(config.model_dump_yaml()))
    arguments = [
        "flops",
        "--config",
        str(path),
        "--output",
        str(output),
    ]
    cli_main(arguments)
    rows = pd.read_csv(output).set_index("band_config")
    assert set(rows.index) == {"rgb", "s2"}
    assert (rows["task"] == "classification").all()
    assert rows.loc["rgb", "n_channels"] == 3
    assert rows.loc["s2", "n_channels"] == 12
    assert (rows["gflops_total"] > 0).all()
    assert rows.loc["s2", "gflops_backbone"] > rows.loc["rgb", "gflops_backbone"]
    assert (rows["throughput_samples_per_sec"] > 0).all()
    before = output.read_bytes()
    cli_main([*arguments, "--resume"])
    assert output.read_bytes() == before
