"""Strict synthetic profiling configuration and precedence."""

from typing import Any

import pytest
from pydantic import ValidationError

from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import ModelPreset, load_model_preset
from geo_eval_harness.config.schema import ModelConfig


@pytest.mark.parametrize(
    "values",
    [
        {"schema_version": True},
        {"unknown": 1},
        {"runtime": {"device": "tpu"}},
        {"runtime": {"seed": True}},
        {"runtime": {"verbose": "false"}},
        {"input": {"band_configs": []}},
        {"input": {"band_configs": ["rgb", "rgb"]}},
        {"input": {"band_configs": ["sar"]}},
        {"input": {"band_source": " "}},
        {"input": {"image_size": None}},
        {"input": {"image_size": 0}},
        {"input": {"image_size": "32"}},
        {"input": {"normalization": "identity"}},
        {"classification": {"head": "other"}},
        {"classification": {"num_classes": 0}},
        {"segmentation": {"heads": ["fpn", "fpn"]}},
        {"segmentation": {"heads": ["other"]}},
        {"segmentation": {"heads": None}},
        {"segmentation": {"band_configs": ["s2", "s2"]}},
        {"segmentation": {"probe": {"layers": None}}},
        {"segmentation": {"probe": {"learning_rate": float("inf")}}},
        {"timing": {"batch_size": 0}},
        {"timing": {"n_warmup": -1}},
        {"timing": {"n_measure": 0}},
        {"output": {"file": ""}},
        {"output": {"resume": "true"}},
    ],
)
def test_invalid_settings_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FlopsConfig.model_validate({"model": {"name": "rcf"}, **values})


def test_presets_and_dataset_defaults_respect_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = ModelPreset.model_validate(
        {
            "name": "example",
            "target": "example.Model",
            "kwargs": {"features": 12},
            "input": {"image_size": 256},
            "segmentation": {"layers": ["deep"], "learning_rate": 0.02},
            "dataset_overrides": {
                "cloudsen12": {
                    "kwargs": {"features": 20},
                    "input": {"image_size": 384, "normalization": "model"},
                    "segmentation": {"layers": ["dataset"], "temporal_pool": "max"},
                }
            },
        }
    )
    monkeypatch.setattr("geo_eval_harness.config.flops.load_model_preset", lambda *a, **kw: preset)
    config = FlopsConfig(model=ModelConfig(name="example"))
    resolved, selected = config.resolve()
    assert resolved.input.image_size == 384
    assert resolved.input.normalization == "model"
    assert resolved.segmentation.probe.layers == ["dataset"]
    assert resolved.segmentation.probe.learning_rate == 0.02
    assert resolved.segmentation.probe.temporal_pool == "max"
    assert selected.kwargs == {"features": 20}
    explicit = FlopsConfig.model_validate(
        {
            "model": {"name": "example", "kwargs": {"features": 7}},
            "input": {"image_size": 32, "normalization": "none"},
            "segmentation": {"probe": {"layers": [], "temporal_pool": "mean"}},
            "output": {"resume": False},
        }
    )
    resolved, selected = explicit.resolve()
    assert resolved.input.image_size == 32
    assert resolved.input.normalization == "none"
    assert resolved.segmentation.probe.layers == []
    assert resolved.segmentation.probe.temporal_pool == "mean"
    assert not resolved.output.resume
    assert selected.kwargs["features"] == 7
    assert config.segmentation.probe.layers == []
    assert preset.segmentation.layers == ["deep"]


def test_explicit_null_target_and_empty_kwargs_preserve_selection() -> None:
    config = FlopsConfig.model_validate(
        {"model": {"name": "rcf", "target": None, "kwargs": {}}, "runtime": {"seed": 41}}
    )
    resolved, preset = config.resolve()
    assert preset.kwargs["seed"] == resolved.runtime.seed == 41
    assert config.model_dump_yaml()["model"]["target"] is None


def test_every_packaged_image_preset_resolves_without_construction() -> None:
    for name in list_model_configs():
        config = FlopsConfig(model=ModelConfig(name=name))
        if load_model_preset(config.model).track != "image":
            with pytest.raises(ValueError, match="coordinate encoder"):
                config.resolve()
        else:
            resolved, preset = config.resolve()
            assert resolved.input.image_size > 0, name
            assert preset.target, name


def test_custom_constructor_is_not_imported() -> None:
    config = FlopsConfig.model_validate(
        {
            "model": {
                "name": "custom",
                "target": "not_installed.Model",
                "kwargs": {"nested": {"_target_": "ordinary.metadata"}, "enabled": False},
            }
        }
    )
    _, preset = config.resolve()
    assert preset.kwargs == config.model.kwargs
