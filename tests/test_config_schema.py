# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the strict core image configuration schema."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geo_eval_harness.config.flops import FlopsRuntimeConfig
from geo_eval_harness.config.profile import ProfileRuntimeConfig
from geo_eval_harness.config.run import RunConfig, load_run_config, validate_run_config
from geo_eval_harness.coordbench.config import CoordRuntimeConfig

_RUNTIME_SECTIONS = (CoordRuntimeConfig, FlopsRuntimeConfig, ProfileRuntimeConfig)


def valid_config() -> dict:
    """Return the smallest valid image benchmark configuration."""
    return {"model": {"name": "timm/resnet50"}, "datasets": ["m-eurosat"]}


def test_defaults_and_modern_shape() -> None:
    config = validate_run_config(valid_config())
    assert config.runtime.device == "cuda:0"
    assert config.input.normalization == "dataset"
    assert config.classification.linear.c_count == 40
    assert config.output.directory == "results"


def test_all_datasets_must_be_an_exclusive_selection() -> None:
    config = valid_config()
    config["datasets"] = ["all"]
    assert validate_run_config(config).datasets == ["all"]
    config["datasets"] = ["all", "m-eurosat"]
    with pytest.raises(ValidationError, match="cannot be combined"):
        validate_run_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("runtime", "batch_size", "64"),
        ("runtime", "workers", True),
        ("output", "resume", "false"),
        ("model", "name", 50),
        ("classification", "bootstrap_samples", 1.0),
    ],
)
def test_wrong_types_are_rejected(section: str, field: str, value: object) -> None:
    config = valid_config()
    config[section] = {field: value}
    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_zero_bootstrap_samples_are_rejected() -> None:
    config = valid_config()
    config["classification"] = {"bootstrap_samples": 0}
    with pytest.raises(ValidationError, match="greater than 0"):
        validate_run_config(config)


def test_false_and_null_values_are_preserved() -> None:
    config = valid_config()
    config["output"] = {"directory": "results", "file": None, "resume": False}
    config["input"] = {"image_size": None, "normalization": "none"}
    parsed = validate_run_config(config)
    assert parsed.output.resume is False
    assert parsed.output.file is None
    assert parsed.input.image_size is None


def test_unknown_nested_field_is_rejected() -> None:
    config = valid_config()
    config["runtime"] = {"batch_size": 8, "batc_size": 4}
    with pytest.raises(ValidationError, match="batc_size"):
        validate_run_config(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"datasets": []},
        {"datasets": [""]},
        {"datasets": ["x", "x"]},
        {"input": {"bands": " "}},
        {"input": {"bands": []}},
        {"input": {"bands": ["red", ""]}},
        {"input": {"bands": ["red", "red"]}},
        {"classification": {"methods": ["knn", "knn"]}},
    ],
)
def test_empty_or_duplicate_selections_are_rejected(overrides: dict[str, object]) -> None:
    config = valid_config()
    config.update(overrides)
    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_invalid_ranges_and_calibration_are_rejected() -> None:
    config = valid_config()
    config["classification"] = {"linear": {"c_log10_start": 4.0, "c_log10_stop": -6.0}}
    with pytest.raises(ValidationError):
        validate_run_config(config)
    config["classification"] = {
        "calibration": {"temp_scale": True},
        "linear": {"refit_train_val": True},
    }
    with pytest.raises(ValidationError, match="linear selected"):
        validate_run_config(config)
    config["classification"] = {"linear": {"c_log10_start": float("inf")}}
    with pytest.raises(ValidationError, match="finite"):
        validate_run_config(config)


def test_invalid_device_is_rejected() -> None:
    config = valid_config()
    config["runtime"] = {"device": "gpu:0"}
    with pytest.raises(ValidationError, match="device"):
        validate_run_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"model": {"name": " "}, "datasets": ["x"]},
        {"model": {"name": "x"}, "datasets": ["x"], "input": {"partition": " "}},
        {"model": {"name": "x"}, "datasets": ["x"], "output": {"directory": " "}},
        {
            "model": {"name": "x"},
            "datasets": ["x"],
            "classification": {"knn_device": "gpu"},
        },
    ],
)
def test_blank_names_paths_and_invalid_knn_device_are_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_yaml_loads_bare_exponent_as_float(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  name: rcf\ndatasets: [m-eurosat]\nsegmentation:\n  learning_rate: 1e-3\n",
        encoding="utf-8",
    )
    assert load_run_config(path).segmentation.learning_rate == 1e-3


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("model: {name: rcf}\nmodel: {name: other}\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_run_config(path)


def test_unhashable_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unhashable.yaml"
    path.write_text("[a, b]: value\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError, match="unhashable key"):
        load_run_config(path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scalar.yaml"
    path.write_text("null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level"):
        load_run_config(path)


def test_round_trip_dump_is_valid() -> None:
    config = RunConfig.model_validate(valid_config(), strict=True)
    restored = RunConfig.model_validate(config.model_dump(mode="json"), strict=True)
    assert restored == config
    assert config.model_dump_yaml()["model"]["name"] == "timm/resnet50"


# Every command declares its own device default, and redeclaring the field would
# drop the shared annotation's validation.
@pytest.mark.parametrize(
    "section", [RunConfig.model_fields["runtime"].annotation, *_RUNTIME_SECTIONS]
)
@pytest.mark.parametrize("device", ["gpu", "cuda:nope", "mps", " "])
def test_every_runtime_section_validates_its_device(section: type, device: str) -> None:
    with pytest.raises(ValidationError, match="device"):
        section(device=device)


@pytest.mark.parametrize(
    "section", [RunConfig.model_fields["runtime"].annotation, *_RUNTIME_SECTIONS]
)
def test_every_runtime_section_accepts_supported_devices(section: type) -> None:
    assert [section(device=name).device for name in ("cpu", "cuda", "cuda:1")] == [
        "cpu",
        "cuda",
        "cuda:1",
    ]


@pytest.mark.parametrize(
    ("segmentation", "message"),
    [
        ({"lora_rank": 8}, "lora_rank requires backbone_learning_rate"),
        ({"backbone_learning_rate": 1e-4}, "require lora_rank"),
        ({"lora_alpha": 32.0}, "require lora_rank"),
    ],
)
def test_lora_settings_require_each_other(segmentation: dict, message: str) -> None:
    config = valid_config()
    config["segmentation"] = segmentation
    with pytest.raises(ValidationError, match=message):
        validate_run_config(config)


def test_lora_targets_alone_keep_the_frozen_probe_and_name_lora_runs() -> None:
    config = valid_config()
    config["segmentation"] = {"head": "upernet", "lora_target_modules": "blocks"}
    frozen = validate_run_config(config).segmentation
    config["segmentation"] |= {"lora_rank": 8, "backbone_learning_rate": 1e-4}
    lora = validate_run_config(config).segmentation

    assert frozen.method == "seg-upernet"
    assert lora.method == "seg-upernet-lora"


def test_lora_requires_targets_from_the_preset_or_config() -> None:
    from geo_eval_harness.config.presets import resolve_run_config

    config = valid_config()
    config["datasets"] = ["burn_scars"]
    config["segmentation"] = {"lora_rank": 8, "backbone_learning_rate": 1e-4}
    with pytest.raises(ValueError, match="does not define LoRA target modules"):
        resolve_run_config(validate_run_config(config), "burn_scars")
