"""Strict coordinate configuration and lightweight command coverage."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from geo_eval_harness.commands._coord import load_config, run
from geo_eval_harness.commands.coord_arguments import add_coord_arguments
from geo_eval_harness.coordbench.config import CoordConfig, load_coord_config, resolve_coord_preset


def _parse(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_coord_arguments(parser)
    return parser.parse_args(args)


def test_coord_defaults() -> None:
    config = CoordConfig.model_validate({"model": {"name": "sincos"}})
    assert config.datasets == ["all"]
    assert config.evaluation.methods == ["knn", "linear"]
    assert config.evaluation.knn_device == "cpu"
    assert config.runtime.device == "cpu"
    assert config.output.directory == "results"
    assert config.output.file is None
    assert not config.output.resume
    preset = resolve_coord_preset(config)
    assert preset.name == "sincos"
    assert preset.track == "coord"
    assert preset.kwargs == {}


@pytest.mark.parametrize(
    "values",
    [
        {"unexpected": True},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"model": {"name": "sincos", "extra": 3}},
        {"datasets": "country"},
        {"datasets": []},
        {"datasets": [" "]},
        {"datasets": ["country", "country"]},
        {"datasets": ["all", "country"]},
        {"datasets": ["does-not-exist"]},
        {"evaluation": {"methods": []}},
        {"evaluation": {"methods": ["linear", "linear"]}},
        {"evaluation": {"methods": ["ridge"]}},
        {"evaluation": {"split": "official"}},
        {"evaluation": {"folds": 1}},
        {"evaluation": {"folds": "5"}},
        {"evaluation": {"folds": True}},
        {"evaluation": {"cell_deg": 0}},
        {"evaluation": {"cell_deg": "10.0"}},
        {"evaluation": {"cell_deg": float("nan")}},
        {"evaluation": {"ridge_alphas": []}},
        {"evaluation": {"ridge_alphas": [0]}},
        {"evaluation": {"ridge_alphas": [-1]}},
        {"evaluation": {"ridge_alphas": [True]}},
        {"evaluation": {"ridge_alphas": ["1.0"]}},
        {"evaluation": {"ridge_alphas": [float("nan")]}},
        {"evaluation": {"ridge_alphas": [float("inf")]}},
        {"evaluation": {"knn_k": 0}},
        {"evaluation": {"knn_device": "auto"}},
        {"evaluation": {"knn_device": None}},
        {"evaluation": {"knn_device": "cuda:bad"}},
        {"runtime": {"device": "cuda:-1"}},
        {"runtime": {"seed": -1}},
        {"runtime": {"seed": 2**64}},
        {"runtime": {"seed": 0.5}},
        {"runtime": {"batch_size": 2}},
        {"output": {"directory": None}},
        {"output": {"file": " "}},
        {"output": {"resume": "false"}},
        {"output": {"directory": " "}},
    ],
)
def test_coord_rejects_invalid_values(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CoordConfig.model_validate({"model": {"name": "sincos"}, **values})


@pytest.mark.parametrize("name", ["country", "worldclim", "worldclim-bio1", "dm-descals"])
def test_coord_accepts_families_and_names(name: str) -> None:
    config = load_config(_parse("--model", "sincos", "--dataset", name))
    assert config.datasets == [name]


@pytest.mark.parametrize("flag", ["--dataset", "--datasets", "--name", "--names"])
def test_dataset_flag_aliases(flag: str) -> None:
    config = load_config(_parse("--model", "sincos", flag, "country", "worldclim-bio1"))
    assert config.datasets == ["country", "worldclim-bio1"]


def test_all_typed_flags() -> None:
    config = load_config(
        _parse(
            "--model",
            "sincos",
            "--dataset",
            "country",
            "--methods",
            "linear",
            "knn",
            "--split",
            "both",
            "--folds",
            "3",
            "--cell-deg",
            "2.5",
            "--ridge-alphas",
            "0.1",
            "1",
            "10",
            "--knn-k",
            "7",
            "--knn-device",
            "cpu",
            "--device",
            "cpu",
            "--seed",
            "42",
            "--output",
            "runs/coord.csv",
            "--resume",
        )
    )
    assert config.evaluation.model_dump() == {
        "methods": ["linear", "knn"],
        "split": "both",
        "folds": 3,
        "cell_deg": 2.5,
        "ridge_alphas": [0.1, 1.0, 10.0],
        "knn_k": 7,
        "knn_device": "cpu",
    }
    assert config.runtime.model_dump() == {"device": "cpu", "seed": 42}
    assert config.output.model_dump() == {
        "directory": "results",
        "file": "runs/coord.csv",
        "resume": True,
    }


def test_yaml_flag_precedence(tmp_path: Path) -> None:
    path = tmp_path / "coord.yaml"
    path.write_text(
        "model:\n  name: sincos\n  kwargs:\n    batch_size: 16\n"
        "datasets: [country]\nevaluation:\n  methods: [linear]\n  cell_deg: 1e1\n"
        "runtime:\n  seed: 7\noutput:\n  file: custom.csv\n  resume: true\n"
    )
    config = load_config(_parse("--config", str(path), "--no-resume", "--seed", "0"))
    assert config.model.kwargs == {"batch_size": 16}
    assert config.evaluation.methods == ["linear"]
    assert config.evaluation.cell_deg == 10.0
    assert config.runtime.seed == 0
    assert config.output.file == "custom.csv"
    assert not config.output.resume
    assert load_coord_config(path).output.resume


def test_model_flag_replaces_yaml_constructor(tmp_path: Path) -> None:
    path = tmp_path / "coord.yaml"
    path.write_text(
        "model:\n  name: custom\n  target: custom.Encoder\n  kwargs:\n    feature: pooled\n"
    )
    config = load_config(_parse("--config", str(path), "--model", "sincos"))
    assert config.model.name == "sincos"
    assert config.model.target is None
    assert config.model.kwargs == {}


@pytest.mark.parametrize(
    "content",
    ["[]", "model: [", "model: {name: sincos, name: mind}", "model: !!python/object:a {}"],
)
def test_bad_yaml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_config(_parse("--config", str(path)))


def test_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(_parse("--config", str(tmp_path / "absent.yaml")))


@pytest.mark.parametrize("name", ["timm/resnet50", "not-a-model"])
def test_invalid_model_selection(name: str) -> None:
    with pytest.raises(ValueError, match=r"image model|Unknown model config"):
        load_config(_parse("--model", name))


def test_run_reports_config_errors_with_argparse_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        run(_parse("--model", "timm/resnet50", "--dry-run"))
    assert error.value.code == 2
    assert "image model" in capsys.readouterr().err


def test_dry_run_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(_parse("--model", "sincos", "--methods", "linear", "--dry-run"))
    path = tmp_path / "coord.yaml"
    path.write_text(capsys.readouterr().out)
    config = load_coord_config(path)
    assert config.evaluation.methods == ["linear"]
    assert load_config(_parse("--config", str(path))) == config


@pytest.mark.parametrize(
    "model",
    [
        {"name": "sincos"},
        {"name": "mind"},
        {"name": "custom", "target": "uncached_optional_encoder.Custom", "kwargs": {"dim": 8}},
    ],
)
def test_dry_run_does_not_import_runtime(tmp_path: Path, model: dict[str, Any]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": model, "datasets": ["country"]}))
    script = """
import argparse
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"torch", "numpy", "pandas", "omegaconf", "sklearn",
                    "huggingface_hub", "uncached_optional_encoder"}:
            raise AssertionError(f"dry-run imported {fullname}")
        if fullname in {"geo_eval_harness.coordbench.run", "geo_eval_harness.coordbench.datasets"}:
            raise AssertionError(f"dry-run imported {fullname}")

sys.meta_path.insert(0, BlockRuntime())
from geo_eval_harness.commands.coord_arguments import add_coord_arguments
from geo_eval_harness.commands._coord import run
parser = argparse.ArgumentParser()
add_coord_arguments(parser)
run(parser.parse_args(["--config", sys.argv[1], "--dry-run"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout)["model"] == {
        **model,
        "target": model.get("target"),
        "kwargs": model.get("kwargs", {}),
    }


def test_command_dispatches_typed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = []
    monkeypatch.setattr("geo_eval_harness.coordbench.run.run_coordbench", configs.append)
    run(_parse("--model", "sincos", "--dataset", "country"))
    assert len(configs) == 1
    assert isinstance(configs[0], CoordConfig)
    assert configs[0].datasets == ["country"]


def test_ridge_grid_yaml_and_fixed_alpha_override(tmp_path: Path) -> None:
    path = tmp_path / "coord.yaml"
    path.write_text("model: {name: sincos}\nevaluation: {ridge_alphas: [0.1, 1, 10]}\n")
    assert load_coord_config(path).evaluation.ridge_alphas == [0.1, 1.0, 10.0]
    config = load_config(_parse("--config", str(path), "--ridge-alphas", "2"))
    assert config.evaluation.ridge_alphas == [2.0]
