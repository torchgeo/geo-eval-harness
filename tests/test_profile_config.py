"""Strict YAML, explicit-flag precedence, and lightweight profile dry runs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from geo_eval_harness.cli import main as cli_main
from geo_eval_harness.commands._profile import profile
from geo_eval_harness.commands.profile_arguments import add_profile_arguments, load_profile_config
from geo_eval_harness.config import presets
from geo_eval_harness.config.presets import ModelPreset, resolve_run_config
from geo_eval_harness.config.profile import ProfileConfig, resolve_profile_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    add_profile_arguments(result)
    return result


def test_profile_defaults_are_cpu_fixed_batch(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["--model", "rcf", "--dataset", "m-eurosat"])
    assert not hasattr(args, "count_flops")
    assert not hasattr(args, "image_size")
    config = load_profile_config(args)
    assert config.runtime.device == "cpu"
    assert config.runtime.batch_size == 32
    assert config.runtime.workers == 0
    assert config.runtime.seed == 0
    assert config.warmup == 3
    assert config.measurements == 20
    assert config.precision == "float32"
    assert not config.count_flops
    assert config.model_dump_yaml() == {
        "model": {"name": "rcf"},
        "dataset": "m-eurosat",
    }


@pytest.mark.parametrize("argument", ["model=rcf", "+model.features=8", "++device=cpu"])
def test_profile_old_overrides_report_migration(
    argument: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["profile", argument])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "overrides have been retired" in message
    assert "--config" in message


def test_profile_explicit_flags_override_yaml(
    parser: argparse.ArgumentParser, tmp_path: Path
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "rcf"},
                "dataset": "caffe",
                "input": {"image_size": 128, "normalization": "model", "interpolation": "area"},
                "runtime": {"batch_size": 16, "seed": 77},
                "count_flops": True,
                "warmup": 9,
                "measurements": 8,
                "precision": "float16",
            }
        )
    )
    config = load_profile_config(
        parser.parse_args(
            [
                "--config",
                str(path),
                "--dataset",
                "m-eurosat",
                "--image-size",
                "none",
                "--no-count-flops",
                "--batch-size",
                "4",
                "--warmup",
                "0",
                "--measurements",
                "1",
                "--precision",
                "bfloat16",
                "--bands",
                "nir, red",
                "--normalization",
                "minmax_zscore",
                "--partition",
                "0.01x_train",
                "--device",
                "auto",
            ]
        )
    )
    assert config.dataset == "m-eurosat"
    assert config.input.image_size is None
    assert config.input.bands == ["nir", "red"]
    assert config.input.normalization == "minmax_zscore"
    assert config.input.interpolation == "area"
    assert config.input.partition == "0.01x_train"
    assert config.runtime.batch_size == 4
    assert config.runtime.seed == 77
    assert config.runtime.device == "auto"
    assert config.warmup == 0
    assert config.measurements == 1
    assert config.precision == "bfloat16"
    assert not config.count_flops
    assert config.model_dump_yaml()["input"]["image_size"] is None


def test_omitted_flags_preserve_yaml_false_and_null(
    parser: argparse.ArgumentParser, tmp_path: Path
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "model: {name: rcf}\ndataset: m-eurosat\ninput: {image_size: null}\ncount_flops: true\n"
    )
    config = load_profile_config(parser.parse_args(["--config", str(path)]))
    assert config.input.image_size is None
    assert config.count_flops


def test_cli_model_selection_replaces_custom_target(
    parser: argparse.ArgumentParser, tmp_path: Path
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "model: {name: custom, target: custom.Model, kwargs: {custom: 1}}\ndataset: m-eurosat\n"
    )
    config = load_profile_config(parser.parse_args(["--config", str(path), "--model", "rcf"]))
    assert config.model == ModelConfig(name="rcf")


@pytest.mark.parametrize(
    ("name", "dataset", "expected_size", "interpolation"),
    [
        ("rcf", "m-eurosat", 224, "bilinear"),
        ("torchgeo/scalemae_large_fmow", "m-eurosat", 64, "area"),
        ("torchgeo/scalemae_large_fmow", "treesatai", 144, "area"),
        ("olmoearth_nano", "m-eurosat", None, "bilinear"),
    ],
)
def test_profile_and_image_share_effective_preprocessing(
    name: str, dataset: str, expected_size: int | None, interpolation: str
) -> None:
    selection = ModelConfig(name=name)
    config, preset = resolve_profile_config(ProfileConfig(model=selection, dataset=dataset))
    image, image_preset = resolve_run_config(
        RunConfig(model=selection, datasets=[dataset]), dataset
    )
    assert config.input == image.input
    assert config.input.image_size == expected_size
    assert config.input.interpolation == interpolation
    assert preset == image_preset


def test_explicit_input_settings_override_preset_and_dataset() -> None:
    config = ProfileConfig.model_validate(
        {
            "model": {"name": "torchgeo/scalemae_large_fmow"},
            "dataset": "m-eurosat",
            "input": {"image_size": None, "interpolation": "nearest", "normalization": "none"},
        }
    )
    effective, preset = resolve_profile_config(config)
    assert effective.input.image_size is None
    assert effective.input.interpolation == "nearest"
    assert effective.input.normalization == "none"
    assert preset.kwargs["res"] == 3.5
    assert "image_size" not in preset.kwargs


def test_typed_preset_input_normalization_is_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    preset = ModelPreset.model_validate(
        {
            "name": "toy",
            "target": "custom.Model",
            "input": {"normalization": "model"},
            "kwargs": {"input_normalization": "imagenet"},
        }
    )
    monkeypatch.setattr(presets, "load_model_preset", lambda *_, **__: preset)
    effective, resolved = resolve_profile_config(
        ProfileConfig(model=ModelConfig(name="toy"), dataset="m-eurosat")
    )
    assert effective.input.normalization == "model"
    assert resolved.kwargs["input_normalization"] == "imagenet"


@pytest.mark.parametrize(
    "values",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": "1"},
        {"datasets": ["m-eurosat"]},
        {"dataset": " "},
        {"model": {"name": "rcf", "unexpected": 1}},
        {"runtime": {"device": "cuda:nope"}},
        {"runtime": {"device": "mps"}},
        {"runtime": {"batch_size": 0}},
        {"runtime": {"batch_size": True}},
        {"runtime": {"seed": -1}},
        {"runtime": {"seed": 2**32}},
        {"runtime": {"workers": -1}},
        {"warmup": -1},
        {"measurements": 0},
        {"warmup": "3"},
        {"count_flops": "false"},
        {"precision": "float64"},
        {"input": {"bands": []}},
        {"input": {"bands": ["red", "red"]}},
        {"input": {"bands": "red,green"}},
        {"input": {"normalization": "identity"}},
        {"input": {"image_size": False}},
        {"input": {"image_size": 0}},
        {"input": {"time_steps": 0}},
    ],
)
def test_profile_rejects_invalid_and_unknown_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProfileConfig.model_validate({"model": {"name": "rcf"}, "dataset": "m-eurosat", **values})


@pytest.mark.parametrize(
    ("model", "dataset", "message"),
    [
        ("unknown", "m-eurosat", "Unknown model"),
        ("rcf", "unknown", "unknown dataset"),
        ("sincos", "m-eurosat", "coordinate encoder"),
    ],
)
def test_profile_rejects_unknown_or_incompatible_selection(
    model: str, dataset: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_profile_config(ProfileConfig(model=ModelConfig(name=model), dataset=dataset))


def test_custom_model_resolution_does_not_import_target() -> None:
    selection = ModelConfig(name="custom", target="not_installed.Model", kwargs={"features": 2})
    config, preset = resolve_profile_config(ProfileConfig(model=selection, dataset="m-eurosat"))
    assert config.model == selection
    assert preset.kwargs == {"features": 2}
    assert preset.target == "not_installed.Model"


@pytest.mark.parametrize("size", ["0", "-1", "2.5", "false"])
def test_parser_rejects_invalid_resize_values(parser: argparse.ArgumentParser, size: str) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args(["--image-size", size])


@pytest.mark.parametrize(
    "content",
    ["[]", "model: [", "model: {name: rcf, name: rcf}", "model: !!python/object:bad {}"],
)
def test_profile_reports_expected_yaml_errors(
    parser: argparse.ArgumentParser,
    tmp_path: Path,
    content: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(content)
    with pytest.raises(SystemExit) as error:
        profile(parser.parse_args(["--config", str(path), "--dry-run"]))
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_profile_reports_missing_files(
    parser: argparse.ArgumentParser, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        profile(parser.parse_args(["--config", str(tmp_path / "missing.yaml"), "--dry-run"]))
    assert error.value.code == 2
    assert "No such file" in capsys.readouterr().err


def test_profile_dry_run_round_trip_is_lightweight(
    parser: argparse.ArgumentParser, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = [
        "--model",
        "torchgeo/scalemae_large_fmow",
        "--dataset",
        "m-eurosat",
        "--image-size",
        "none",
        "--no-count-flops",
        "--dry-run",
    ]
    profile(parser.parse_args(flags))
    output = capsys.readouterr().out
    path = tmp_path / "profile.yaml"
    path.write_text(output)
    original, _ = resolve_profile_config(load_profile_config(parser.parse_args(flags)))
    repeated, _ = resolve_profile_config(
        load_profile_config(parser.parse_args(["--config", str(path)]))
    )
    assert original == repeated
    assert "image_size: null" in output
    assert "count_flops: false" in output
    code = f"""
import argparse, sys
from geo_eval_harness.commands._profile import profile
from geo_eval_harness.commands.profile_arguments import add_profile_arguments
parser = argparse.ArgumentParser()
add_profile_arguments(parser)
profile(parser.parse_args({json.dumps(flags)}))
assert not set(("torch", "torchgeo", "numpy", "omegaconf", "geo_eval_harness.main",
                "geo_eval_harness.commands._profile_runtime")) & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert yaml.safe_load(result.stdout) == yaml.safe_load(output)
