# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Metadata checks honor effective settings without changing reusable YAML."""

from pathlib import Path

import pytest
import torch
import yaml

from geo_eval_harness.cli import main
from geo_eval_harness.config.presets import (
    NORMALIZATIONS,
    ModelPreset,
    build_model,
    load_model_preset,
    resolve_run_config,
)
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.errors import UnsupportedNormalizationError
from geo_eval_harness.models import RCFBench


@pytest.mark.parametrize(
    ("model", "dataset", "bands", "normalization"),
    [
        ("rcf", "m-eurosat", "rgb", "dataset"),
        ("rcf", "all", "all", "none"),
        ("rcf", "m-eurosat", "nir,red,blue", "minmax"),
        ("rcf", "caffe", "rgb", "minmax_zscore"),
        ("timm/resnet50", "m-eurosat", "red,green,blue", "model"),
        ("torchgeo/scalemae_large_fmow", "m-eurosat", "rgb", "model"),
        ("torchgeo/deo_rgb", "m-eurosat", "rgb", "model"),
        ("torchgeo/deo_s2", "m-eurosat", "all", "model"),
        ("olmoearth_nano", "m-eurosat", "all", "model"),
    ],
)
def test_valid_inputs_remain_reusable(
    model: str,
    dataset: str,
    bands: str,
    normalization: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "run",
            "--model",
            model,
            "--dataset",
            dataset,
            "--bands",
            bands,
            "--normalization",
            normalization,
            "--dry-run",
        ]
    )
    config = RunConfig.model_validate(yaml.safe_load(capsys.readouterr().out))
    assert config.model.name == model
    assert config.datasets == [dataset]
    assert config.input.bands == (bands.split(",") if "," in bands else bands)
    assert config.input.normalization == normalization


@pytest.mark.parametrize("datasets", [["m-eurosat", "m-pv4ger"], ["all"]])
def test_explicit_bands_must_exist_in_every_selected_dataset(datasets: list[str]) -> None:
    flags = [flag for name in datasets for flag in ("--dataset", name)]
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", *flags, "--bands", "nir", "--dry-run"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "invalid_input", [{"bands": ["B99"]}, {"bands": "B99"}, {"normalization": "model"}]
)
def test_dataset_defaults_are_validated(
    invalid_input: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    preset = ModelPreset.model_validate(
        {
            **load_model_preset(ModelConfig(name="rcf")).model_dump_yaml(),
            "dataset_overrides": {"burn_scars": {"input": invalid_input}},
        }
    )
    monkeypatch.setattr(
        "geo_eval_harness.config.presets.load_model_preset", lambda *args, **kwargs: preset
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "--model",
                "rcf",
                "--dataset",
                "m-eurosat",
                "--dataset",
                "burn_scars",
                "--dry-run",
            ]
        )
    assert error.value.code == 2


def test_validation_respects_precedence_without_materializing_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preset = ModelPreset.model_validate(
        {
            **load_model_preset(ModelConfig(name="rcf")).model_dump_yaml(),
            "input": {"bands": ["B99"], "normalization": "model"},
            "dataset_overrides": {
                "m-eurosat": {"input": {"bands": ["nir"], "normalization": "none"}}
            },
        }
    )
    monkeypatch.setattr(
        "geo_eval_harness.config.presets.load_model_preset", lambda *args, **kwargs: preset
    )
    path = tmp_path / "run.yaml"
    values = {
        "model": {"name": "rcf"},
        "datasets": ["m-eurosat"],
        "input": {"image_size": None},
        "output": {"resume": False},
        "segmentation": {"layers": []},
    }
    path.write_text(yaml.safe_dump(values))
    main(["run", "--config", str(path), "--dry-run"])
    output = yaml.safe_load(capsys.readouterr().out)
    assert output == values
    effective, _ = resolve_run_config(RunConfig.model_validate(output), "m-eurosat")
    assert effective.input.bands == ["nir"]
    assert effective.input.normalization == "none"

    values["input"].update({"bands": ["B99"], "normalization": "model"})
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    assert str(path) in capsys.readouterr().err

    main(
        [
            "run",
            "--config",
            str(path),
            "--bands",
            "red",
            "--normalization",
            "dataset",
            "--dry-run",
        ]
    )
    output = yaml.safe_load(capsys.readouterr().out)
    assert output["input"] == {
        "bands": ["red"],
        "normalization": "dataset",
        "image_size": None,
    }
    assert output["output"] == {"resume": False}
    assert output["segmentation"] == {"layers": []}
    path.write_text(yaml.safe_dump(output))
    main(["run", "--config", str(path), "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out) == output


def test_custom_target_does_not_inherit_named_preset_capabilities(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = {
        "model": {
            "name": "rcf",
            "target": "uninstalled_custom_model.Model",
            "kwargs": {"custom_option": {"value": 3}},
        },
        "datasets": ["m-eurosat"],
        "input": {"bands": ["red"], "normalization": "model"},
    }
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(values))
    main(["run", "--config", str(path), "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out) == values
    values["input"]["bands"] = ["B99"]
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2


def test_rcf_preset_restriction_matches_runtime_normalization() -> None:
    preset = load_model_preset(ModelConfig(name="rcf", kwargs={"features": 8}))
    bands = get_bench_dataset_class("m-eurosat").resolve_band_specs("rgb")
    assert preset.supports_model_normalization is False
    with pytest.raises(UnsupportedNormalizationError, match="does not support"):
        build_model(preset, bands=bands, normalization="model_native")
    with pytest.raises(UnsupportedNormalizationError, match="expected_input_unit"):
        RCFBench(bands=bands, features=8, normalization="model_native")
    for normalization in ("dataset", "none", "minmax", "minmax_zscore"):
        model = build_model(preset, bands=bands, normalization=NORMALIZATIONS[normalization])
        features = model(torch.ones(2, len(bands), 8, 8))
        assert features.shape == (2, 8)
        assert torch.isfinite(features).all()


def test_runtime_only_normalization_errors_use_config_error_presentation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(config: RunConfig, *, strict: bool) -> None:
        raise UnsupportedNormalizationError("custom model has no pretrained statistics")

    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.main", fail)
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", "--dataset", "m-eurosat"])
    assert error.value.code == 2
    assert capsys.readouterr().err == "error: custom model has no pretrained statistics\n"
