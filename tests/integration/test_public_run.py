"""Public console runs use actual file readers, numerical solvers, and result storage."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.support.cli import cli_output, run_public_cli
from tests.support.data import write_classification_files

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("temperature_scaling", [False, True], ids=["multisensor", "calibrated"])
def test_public_classification_and_resume(tmp_path: Path, *, temperature_scaling: bool) -> None:
    write_classification_files(tmp_path, "m-eurosat", (2, 7), all_bands=True)
    datasets = ["m-eurosat"]
    bands = ["red", "green", "blue"] if temperature_scaling else "all"
    if not temperature_scaling:
        labels = tuple([int(index == label) for index in range(43)] for label in (0, 1))
        write_classification_files(tmp_path, "m-bigearthnet", labels, all_bands=True)
        datasets.append("m-bigearthnet")
    output = tmp_path / "classification.csv"
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "rcf"},
                "datasets": datasets,
                "input": {"bands": bands, "image_size": 16, "interpolation": "nearest"},
                "classification": {
                    "bootstrap_samples": 5,
                    "linear": {
                        "c_log10_start": 1.0,
                        "c_log10_stop": 2.0,
                        "c_count": 2,
                        "refit_train_val": not temperature_scaling,
                    },
                    "calibration": {"temp_scale": temperature_scaling},
                },
                "runtime": {"device": "cpu", "batch_size": 8, "workers": 0, "seed": 7},
                "output": {"file": str(output)},
            }
        )
    )
    completed = run_public_cli("run", "--config", str(config), cwd=tmp_path)
    assert completed.returncode == 0, cli_output(completed)
    assert output.is_file(), cli_output(completed)
    rows = pd.read_csv(output)
    assert len(rows) == 2 * len(datasets)
    assert set(rows["dataset"]) == set(datasets)
    assert rows["seed"].eq(7).all()
    assert rows["normalization"].eq("bandspec_zscore").all()
    assert rows["bands"].eq(",".join(bands) if isinstance(bands, list) else bands).all()
    assert rows["feature_dim"].eq(512).all()
    assert rows["n_train"].eq(24).all()
    assert rows["n_val"].eq(8).all()
    assert rows["n_test"].eq(8).all()
    assert rows["config_hash"].str.fullmatch(r"[0-9a-f]{16,64}").all()
    assert np.isfinite(rows[["metric_value", "ci_lower", "ci_upper"]]).all().all()
    for dataset in datasets:
        selected = rows[rows["dataset"] == dataset].set_index("method")
        assert set(selected.index) == {"knn5", "linear"}
        metric = "micro_mAP" if dataset == "m-bigearthnet" else "accuracy"
        assert selected["metric_name"].eq(metric).all()
        np.testing.assert_allclose(selected["metric_value"], 1.0, atol=1e-6)
        assert selected["ci_lower"].between(0, 1).all()
        assert selected["ci_upper"].between(0, 1).all()
        assert (selected["ci_lower"] <= selected["ci_upper"]).all()
    if temperature_scaling:
        linear = rows[rows["method"] == "linear"].iloc[0]
        assert np.isfinite(linear[["temperature", "ece", "ece_ts"]].astype(float)).all()
        assert linear["temperature"] > 0
        assert 0 <= linear["ece_ts"] < 0.2

    before = output.read_bytes()
    shutil.rmtree(tmp_path / "data")
    resumed = run_public_cli(
        "run", "--config", str(config), "--resume", "--workers", "8", cwd=tmp_path
    )
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before


def test_classification_fixture_has_no_split_leakage(tmp_path: Path) -> None:
    """Keep toy perfection meaningful: held-out examples are not training duplicates."""
    from geo_eval_harness.datasets import get_datasets

    directory = write_classification_files(tmp_path, "m-eurosat", (2, 7))
    partition = json.loads((directory / "default_partition.json").read_text())
    groups = [set(ids) for ids in partition.values()]
    assert sum(map(len, groups)) == len(set.union(*groups))
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        _, *loaders = get_datasets(
            "m-eurosat", batch_size=8, num_workers=0, bands="rgb", image_size=16, return_val=True
        )
        split_images = [
            {sample["image"].numpy().tobytes() for sample in loader.dataset} for loader in loaders
        ]
    assert [len(group) for group in split_images] == [24, 8, 8]
    assert sum(map(len, split_images)) == len(set.union(*split_images))
