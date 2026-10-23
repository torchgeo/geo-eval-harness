"""Tests for the handcrafted classification sweep's coverage and output checks."""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import yaml

from geo_eval_harness.config.presets import merge_settings, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import list_datasets
from geo_eval_harness.datasets.loading import get_dataset_task
from geo_eval_harness.resume import resume_config_hash
from tests.support.cli import cli_output, run_cli
from tests.support.data import write_classification_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sweep(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return importlib.import_module("experiments.run_handcrafted")


def test_classification_inventory_excludes_other_tasks(sweep):
    names = sweep.classification_datasets()
    assert names == [name for name in list_datasets() if get_dataset_task(name) == "classification"]
    assert {"aid", "benv2", "treesatai", "m-bigearthnet", "eurosat-spatial", "resisc45"}.issubset(
        names
    )
    assert not {"burn_scars", "pastis", "spacenet2"}.intersection(names)


def test_four_levels_produce_independent_classification_jobs(sweep):
    names = sweep.classification_datasets()
    jobs = [job for level in range(4) for job in sweep.build_jobs(level, names)]
    assert len(jobs) == 4 * len(names)
    assert len({job.label for job in jobs}) == len(jobs)
    for job in jobs:
        assert isinstance(job.config, RunConfig)
        config, preset = resolve_run_config(job.config, job.config.datasets[0])
        assert config.input.bands == "all"
        assert config.input.normalization == "none"
        assert config.input.image_size == 224
        assert config.classification.linear.refit_train_val is True
        assert config.classification.bootstrap_samples == 200
        assert config.classification.methods == ["knn", "linear"]
        assert preset.name == job.label.split("/")[0]
    assert sweep.model_name(0) == "imagestats_handcrafted_control"
    config = sweep.build_jobs(3, ["eurosat"])[0].config
    assert config.model.name == "handcrafted_level3"
    assert resolve_run_config(config, "eurosat")[1].kwargs == {"level": 3}


def _row(method):
    return {
        "dataset": "toy",
        "method": method,
        "metric_name": "micro_mAP",
        "bands": "all",
        "normalization": "identity",
        "merge_val": "True",
        "feature_dim": "50",
        "num_classes": "19",
        "image_size": "224",
        "seed": "0",
        "config_hash": "current",
        "interpolation": "bilinear",
        "partition": "default",
        "bootstrap": "200",
        "c_range_start": "-6",
        "c_range_stop": "4",
        "c_range_num": "40",
        "n_train": "20",
        "n_val": "4",
        "n_test": "4",
        "metric_value": "0.6",
        "ci_lower": "0.5",
        "ci_upper": "0.7",
    }


def _spec():
    return {
        "multilabel": True,
        "feature_dim": 50,
        "num_classes": 19,
        "config_hash": "current",
        "split_sizes": {"train": 20, "val": 4, "test": 4},
    }


def test_success_requires_both_methods_not_just_a_successful_process(sweep):
    assert not sweep.validate_case([_row("knn5"), _row("linear")], "toy", _spec())
    assert "expected one row" in sweep.validate_case([_row("knn5")], "toy", _spec())[0]
    assert sweep.validate_case([_row("knn5"), _row("linear"), _row("linear")], "toy", _spec())


def test_other_resume_configurations_do_not_count_as_current_rows(sweep):
    older = {**_row("linear"), "config_hash": "different-config"}
    assert not sweep.validate_case([_row("knn5"), _row("linear"), older], "toy", _spec())
    assert sweep.validate_case([_row("knn5"), older], "toy", _spec())


def test_wrong_settings_scores_and_feature_width_are_visible(sweep):
    for field, value in (
        ("normalization", "bandspec_zscore"),
        ("feature_dim", "49"),
        ("metric_name", "accuracy"),
        ("merge_val", "False"),
        ("n_train", "19"),
        ("metric_value", "nan"),
        ("bootstrap", "0"),
        ("c_range_num", "1"),
        ("ci_lower", "nan"),
    ):
        rows = [_row("knn5"), {**_row("linear"), field: value}]
        assert sweep.validate_case(rows, "toy", _spec())


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_feature_manifest_uses_resolved_current_hashes(sweep, level):
    datasets = ["m-pv4ger", "treesatai", "aid"]
    manifest = sweep.feature_manifest(level, datasets)
    json.dumps(manifest, allow_nan=False)
    for dataset, spec in manifest.items():
        assert len(spec["feature_names"]) == spec["feature_dim"]
        job = sweep.build_jobs(level, [dataset])[0]
        effective, preset = resolve_run_config(job.config, dataset)
        assert spec["config_hash"] == resume_config_hash(
            effective, preset, task=get_dataset_task(dataset)
        )
        if level:
            assert sorted(
                column for record in spec["feature_metadata"] for column in record["columns"]
            ) == list(range(spec["feature_dim"]))


def test_summary_does_not_mix_historical_configs_with_current_scores(
    sweep: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sweep, "feature_manifest", lambda *args: {"toy": _spec()})
    args = argparse.Namespace(
        output_dir=tmp_path, run_dir=tmp_path, datasets=["toy"], devices=[0], levels=[1]
    )
    for level in (0, 1):
        current = {**_row("linear"), "metric_value": 0.6 + 0.1 * level}
        historical = {**current, "config_hash": "historical", "metric_value": 0.9}
        pd.DataFrame([current, historical]).to_csv(
            tmp_path / f"{sweep.model_name(level)}.csv", index=False
        )
    sweep.summarize(args)
    summary = pd.read_csv(tmp_path / "summary_level1.csv")
    assert len(summary) == 1
    assert summary.iloc[0]["config_hash"] == "current"
    assert summary.iloc[0]["change_from_imagestats"] == pytest.approx(0.1)

    pd.DataFrame([historical]).to_csv(tmp_path / f"{sweep.model_name(1)}.csv", index=False)
    with pytest.raises(ValueError, match="No result rows match the current configuration"):
        sweep.summarize(args)


def test_study_file_entrypoint_dry_run_uses_current_cli(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "run_handcrafted.py"),
            "--dry-run",
            "--datasets",
            "aid",
            "--devices",
            "0",
            "2",
            "--output-dir",
            str(tmp_path / "results"),
            "--run-dir",
            str(tmp_path / "manifests"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, cli_output(result)
    assert "-m geo_eval_harness run --config" in result.stderr
    assert "--resume" in result.stderr
    assert "handcrafted_level3" in result.stderr
    assert "image_cli" not in result.stderr
    assert not list((tmp_path / "results").glob("*.csv"))
    assert len(list((tmp_path / "manifests").glob("*.features.json"))) == 4


@pytest.mark.integration
@pytest.mark.parametrize(("level", "width"), [(0, 12), (1, 27), (2, 63), (3, 111)])
def test_study_config_runs_real_cli_and_resumes(
    sweep: ModuleType, tmp_path: Path, level: int, width: int
) -> None:
    write_classification_files(tmp_path, "m-pv4ger", (0, 1), all_bands=True)
    job = sweep.build_jobs(level, ["m-pv4ger"])[0]
    config = RunConfig.model_validate(
        merge_settings(
            job.config.model_dump_yaml(),
            {
                "runtime": {"device": "cpu", "workers": 0, "batch_size": 8},
                "input": {"image_size": 16},
                "classification": {
                    "bootstrap_samples": 8,
                    "linear": {"c_log10_start": -2.0, "c_log10_stop": 0.0, "c_count": 2},
                },
            },
        )
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump_yaml()))
    output = tmp_path / "results.csv"
    arguments = ["run", "--config", str(config_path), "--output", str(output), "--resume"]
    result = run_cli(*arguments, cwd=tmp_path)
    assert result.returncode == 0, cli_output(result)
    rows = pd.read_csv(output)
    assert len(rows) == 2
    assert set(rows["method"]) == {"knn5", "linear"}
    assert set(rows["name"]) == {sweep.model_name(level)}
    assert set(rows["feature_dim"]) == {width}
    assert set(rows["normalization"]) == {"identity"}
    assert set(rows["bands"]) == {"all"}
    assert set(rows["n_train"]) == {24}
    assert set(rows["n_val"]) == {8}
    assert set(rows["n_test"]) == {8}
    assert set(rows["metric_name"]) == {"accuracy"}
    assert rows["metric_value"].between(0.99, 1).all()
    effective, preset = resolve_run_config(config, "m-pv4ger")
    assert set(rows["config_hash"]) == {
        resume_config_hash(effective, preset, task=get_dataset_task("m-pv4ger"))
    }
    before = output.read_bytes()
    resumed = run_cli(*arguments, cwd=tmp_path)
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before
