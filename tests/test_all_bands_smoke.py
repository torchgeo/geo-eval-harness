"""Check all-band CSV results with real m-eurosat data.

Run ResNet-18 with all bands on a small ``m-eurosat`` partition.

Marked ``slow`` because feature extraction uses real imagery.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.support.cli import run_cli
from tests.support.data import require_dataset_data


@pytest.mark.slow
def test_all_bands_e2e(tmp_path: Path) -> None:
    require_dataset_data("m-eurosat")

    output = tmp_path / "results.csv"
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "timm/resnet18", "kwargs": {"pretrained": False}},
                "datasets": ["m-eurosat"],
                "input": {"bands": "all", "partition": "0.01x_train", "image_size": 32},
                "runtime": {"batch_size": 16, "workers": 0, "device": "cpu", "seed": 0},
                "classification": {
                    "bootstrap_samples": 10,
                    "linear": {"c_log10_start": -2.0, "c_log10_stop": 2.0, "c_count": 3},
                },
                "output": {"file": str(output)},
            }
        )
    )
    cmd = [
        "run",
        "--config",
        str(config),
    ]
    completed = run_cli(*cmd, cwd=Path.cwd(), timeout=600)
    assert completed.returncode == 0, (
        f"geo-eval-harness exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists(), f"Expected results CSV at {output}"

    df = pd.read_csv(output)
    rows = df[df["dataset"] == "m-eurosat"]
    assert not rows.empty, f"No m-eurosat rows in {output}\n{df}"

    assert "bands" in df.columns, f"`bands` column missing from CSV: {df.columns.tolist()}"
    assert (rows["bands"] == "all").all(), (
        f"Expected bands=all for every m-eurosat row, got {rows['bands'].unique().tolist()}"
    )

    methods = set(rows["method"].unique())
    assert {"knn5", "linear"}.issubset(methods), (
        f"Expected knn5 and linear methods, got {sorted(methods)}"
    )

    feature_dims = rows["feature_dim"].unique().tolist()
    assert feature_dims == [512], f"Expected resnet18 feature_dim=512, got {feature_dims}"

    metric_values = rows["metric_value"].astype(float)
    assert metric_values.between(0.0, 1.0).all(), (
        f"metric_value out of range: {metric_values.tolist()}"
    )
