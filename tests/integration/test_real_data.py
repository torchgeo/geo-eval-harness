"""CPU-capable program smoke tests using real data under the invocation's data/ directory.

Run with ``pytest -m slow tests/integration/test_real_data.py`` after downloading data.
Missing datasets skip individually; present but incompatible or incomplete data must fail.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from geo_eval_harness.datasets import get_bench_dataset_class
from tests.support.cli import cli_output, run_cli
from tests.support.data import require_dataset_data

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("dataset", "bands", "partition"),
    [
        ("m-eurosat", "rgb", "0.01x_train"),
        ("so2sat", "rgb", "default"),
        ("eurosat", "all", "default"),
        ("ucmerced", "rgb", "default"),
    ],
    ids=["v1-m-eurosat", "v2-so2sat", "eurosat", "ucmerced"],
)
def test_real_classification_program(
    tmp_path: Path, dataset: str, bands: str, partition: str
) -> None:
    require_dataset_data(dataset)
    output = tmp_path / "classification.csv"
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "rcf", "kwargs": {"features": 32}},
                "datasets": [dataset],
                "input": {"bands": bands, "partition": partition, "image_size": 32},
                "runtime": {"batch_size": 64, "workers": 0, "device": "cpu"},
                "classification": {
                    "bootstrap_samples": 5,
                    "linear": {"c_log10_start": -2.0, "c_log10_stop": 2.0, "c_count": 3},
                },
                "output": {"file": str(output)},
            }
        )
    )
    arguments = [
        "run",
        "--config",
        str(config),
    ]
    result = run_cli(*arguments, cwd=Path.cwd(), timeout=600)
    assert result.returncode == 0, cli_output(result)
    assert output.is_file(), cli_output(result)
    rows = pd.read_csv(output)
    assert set(rows["method"]) == {"knn5", "linear"}
    assert (rows["dataset"] == dataset).all()
    assert (rows["bands"] == bands).all()
    assert (rows["num_classes"] == get_bench_dataset_class(dataset).num_classes).all()
    assert (rows["feature_dim"] == 32).all()
    assert (rows[["n_train", "n_val", "n_test"]] > 0).all().all()
    assert np.isfinite(rows[["metric_value", "ci_lower", "ci_upper"]]).all().all()
    assert rows["metric_value"].between(0, 1).all()
    assert (rows["ci_lower"] <= rows["ci_upper"]).all()

    before = output.read_bytes()
    resumed = run_cli(*arguments, "--resume", cwd=Path.cwd(), timeout=120)
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "uncached"])
def test_real_segmentation_program(tmp_path: Path, *, cached: bool) -> None:
    require_dataset_data("caffe")
    output = tmp_path / "segmentation.csv"
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "timm/resnet18", "kwargs": {"pretrained": False}},
                "datasets": ["caffe"],
                "input": {"image_size": 32},
                "runtime": {"batch_size": 32, "workers": 0, "device": "cpu", "seed": 0},
                "classification": {"bootstrap_samples": 5},
                "segmentation": {
                    "head": "linear",
                    "epochs": 1,
                    "batch_size": 16,
                    "cache_features": cached,
                },
                "output": {"file": str(output)},
            }
        )
    )
    arguments = [
        "run",
        "--config",
        str(config),
    ]
    result = run_cli(*arguments, cwd=Path.cwd(), timeout=600)
    assert result.returncode == 0, cli_output(result)
    assert output.is_file(), cli_output(result)
    rows = pd.read_csv(output)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["dataset"] == "caffe"
    assert row["method"] == "seg-linear"
    assert row["metric_name"] == "mIoU"
    assert 0 <= row["metric_value"] <= 1
    assert np.isfinite(row[["metric_value", "ci_lower", "ci_upper"]].astype(float)).all()
    assert row["ci_lower"] <= row["ci_upper"]
    assert row["best_batch_size"] == (16 if cached else 32)

    before = output.read_bytes()
    resumed = run_cli(*arguments, "--resume", cwd=Path.cwd(), timeout=120)
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before
