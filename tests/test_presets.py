"""Offline coverage of the typed preset boundary."""

import subprocess
import sys

import pytest

from geo_eval_harness.config import list_model_configs, model_config_path
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import (
    ModelPreset,
    build_model,
    load_model_preset,
    resolve_run_config,
)
from geo_eval_harness.config.profile import ProfileConfig, resolve_profile_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig, load_yaml
from geo_eval_harness.models.torchgeo_models import TorchGeoScaleMAEBench


@pytest.mark.parametrize("name", ["timm/resnet5", "RCF", "not-a-model", "../model/rcf"])
def test_unknown_model_config_is_rejected_without_suggestions(name: str) -> None:
    with pytest.raises(ValueError, match="Unknown model config") as error:
        model_config_path(name)
    assert str(error.value) == f"Unknown model config {name!r}."


@pytest.mark.parametrize("name", list_model_configs())
def test_packaged_preset_is_typed_without_loading_weights(name: str) -> None:
    preset = load_model_preset(ModelConfig(name=name), seed=17)
    assert isinstance(preset, ModelPreset)
    assert preset.target
    assert "name" not in preset.kwargs
    assert "eval" not in preset.kwargs
    assert "dataset_overrides" not in preset.kwargs
    assert "image_size" not in preset.kwargs
    assert "interpolation" not in preset.kwargs
    assert "${" not in str(preset.model_dump())


@pytest.mark.parametrize("name", list_model_configs())
def test_packaged_preset_omits_schema_defaults(name: str) -> None:
    raw = load_yaml(model_config_path(name))
    preset = ModelPreset.model_validate(raw)
    assert raw == preset.model_dump(mode="json", exclude_defaults=True)


def test_rcf_seed_is_explicit_and_overridable() -> None:
    assert load_model_preset(ModelConfig(name="rcf"), seed=17).kwargs["seed"] == 17
    assert (
        load_model_preset(ModelConfig(name="rcf", kwargs={"seed": 9}), seed=17).kwargs["seed"] == 9
    )


@pytest.mark.parametrize("name", ["torchgeo/deo_rgb", "torchgeo/deo_s2"])
@pytest.mark.parametrize("normalization", [None, "dataset"])
def test_deo_normalization_defaults_and_explicit_override(
    name: str, normalization: str | None
) -> None:
    settings = {"model": {"name": name}}
    if normalization is not None:
        settings["input"] = {"normalization": normalization}
    run = RunConfig.model_validate({**settings, "datasets": ["m-eurosat"]})
    profile = ProfileConfig.model_validate({**settings, "dataset": "m-eurosat"})
    flops = FlopsConfig.model_validate(settings)

    for effective, preset in (
        resolve_run_config(run, "m-eurosat"),
        resolve_profile_config(profile),
        flops.resolve(),
    ):
        assert effective.input.normalization == (normalization or "model")
        assert "normalization" not in preset.kwargs


def test_preset_layers_do_not_override_an_explicit_empty_selection() -> None:
    omitted = RunConfig(model=ModelConfig(name="timm/resnet50"), datasets=["caffe"])
    explicit = RunConfig.model_validate(
        {
            **omitted.model_dump_yaml(),
            "segmentation": {"layers": [], "cache_features": False},
            "input": {"image_size": None},
        }
    )
    effective, _ = resolve_run_config(omitted, "caffe")
    assert effective.segmentation.layers == ["layer4", "layer3", "layer2", "layer1"]
    effective, _ = resolve_run_config(explicit, "caffe")
    assert effective.segmentation.layers == []
    assert not effective.segmentation.cache_features
    assert effective.input.image_size is None
    assert explicit.model_dump_yaml()["input"] == {"image_size": None}


def test_custom_model_keeps_constructor_options_separate() -> None:
    selection = ModelConfig(
        name="custom",
        target="custom_model.Model",
        kwargs={"options": {"_target_": "ordinary.mapping"}},
    )
    preset = load_model_preset(selection)
    assert preset.kwargs == {"options": {"_target_": "ordinary.mapping"}}
    assert preset.target == "custom_model.Model"


def test_model_dataset_overrides_are_isolated_and_fall_back() -> None:
    model_cfg = ModelPreset.model_validate(
        {
            "target": "example.Model",
            "name": "example",
            "input": {"image_size": 224},
            "kwargs": {"res": 1.0, "pool": "cls"},
            "dataset_overrides": {
                "m-eurosat": {"input": {"image_size": 64}, "kwargs": {"res": 3.5}},
                "forestnet": {"input": {"image_size": 128}, "kwargs": {"pool": "mean"}},
            },
        }
    )

    eurosat = model_cfg.for_dataset("m-eurosat")
    fallback = model_cfg.for_dataset("unlisted")
    forestnet = model_cfg.for_dataset("forestnet")

    assert (eurosat.input.image_size, eurosat.kwargs["res"], eurosat.kwargs["pool"]) == (
        64,
        3.5,
        "cls",
    )
    assert (fallback.input.image_size, fallback.kwargs["res"], fallback.kwargs["pool"]) == (
        224,
        1.0,
        "cls",
    )
    assert (forestnet.input.image_size, forestnet.kwargs["res"], forestnet.kwargs["pool"]) == (
        128,
        1.0,
        "mean",
    )
    assert not eurosat.dataset_overrides
    assert model_cfg.input.image_size == 224


def test_configuration_import_and_resolution_do_not_import_omegaconf() -> None:
    code = """
import sys
from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.config.presets import load_model_preset
for name in list_model_configs():
    load_model_preset(ModelConfig(name=name))
assert not set(('omegaconf', 'torch', 'torchgeo', 'numpy', 'pandas')) & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("image_size", [None, 96])
def test_scalemae_constructor_uses_resolved_dataset_grid(
    monkeypatch: pytest.MonkeyPatch, image_size: int | None
) -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "torchgeo/scalemae_large_fmow"},
            "datasets": ["m-eurosat"],
            **({"input": {"image_size": image_size}} if image_size is not None else {}),
        }
    )
    effective, preset = resolve_run_config(config, "m-eurosat")
    received = {}

    class constructor:
        wants_resolved_image_size = TorchGeoScaleMAEBench.wants_resolved_image_size

        def __new__(cls, **kwargs: object) -> object:
            received.update(kwargs)
            return object()

    monkeypatch.setattr("geo_eval_harness.models.TorchGeoScaleMAEBench", constructor)
    build_model(preset, bands=[], normalization="identity")
    assert received["image_size"] == effective.input.image_size == (image_size or 64)
    assert received["res"] == 3.5


def test_flops_scalemae_constructor_grid_follows_explicit_synthetic_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geo_eval_harness.config.flops import FlopsConfig

    config = FlopsConfig.model_validate(
        {
            "model": {"name": "torchgeo/scalemae_large_fmow"},
            "input": {"band_source": "m-eurosat", "image_size": 96},
        }
    )
    effective, preset = config.resolve()
    received = {}

    class constructor:
        wants_resolved_image_size = TorchGeoScaleMAEBench.wants_resolved_image_size

        def __new__(cls, **kwargs: object) -> object:
            received.update(kwargs)
            return object()

    monkeypatch.setattr("geo_eval_harness.models.TorchGeoScaleMAEBench", constructor)
    build_model(preset, bands=[], normalization="identity")
    assert received["image_size"] == effective.input.image_size == 96


def test_explicit_constructor_options_override_dataset_defaults() -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "torchgeo/scalemae_large_fmow", "kwargs": {"res": 1.25}},
            "datasets": ["m-eurosat", "m-forestnet"],
        }
    )
    for dataset in config.datasets:
        _, preset = resolve_run_config(config, dataset)
        assert preset.kwargs["res"] == 1.25


def test_explicit_default_values_override_preset_lr_layers_and_range() -> None:
    omitted = RunConfig.model_validate(
        {"model": {"name": "torchgeo/croma_large"}, "datasets": ["m-eurosat"]}
    )
    effective, _ = resolve_run_config(omitted, "m-eurosat")
    assert effective.classification.linear.c_log10_stop != 4.0
    explicit = RunConfig.model_validate(
        {
            **omitted.model_dump_yaml(),
            "classification": {"linear": {"c_log10_stop": 4.0}},
            "segmentation": {"learning_rate": 0.001, "layers": [], "cache_features": False},
            "input": {"image_size": None},
        }
    )
    effective, _ = resolve_run_config(explicit, "m-eurosat")
    assert effective.classification.linear.c_log10_stop == 4.0
    assert effective.segmentation.learning_rate == 0.001
    assert effective.segmentation.layers == []
    assert effective.segmentation.cache_features is False
    assert effective.input.image_size is None


def test_typed_preset_learning_rate_and_head_defaults_have_lower_precedence(monkeypatch) -> None:
    preset = ModelPreset.model_validate(
        {
            "name": "custom",
            "target": "custom_model.Model",
            "segmentation": {"learning_rate": 0.02, "head": "linear", "layers": ["backbone"]},
        }
    )
    monkeypatch.setattr(
        "geo_eval_harness.config.presets.load_model_preset", lambda *args, **kwargs: preset
    )
    config = RunConfig.model_validate({"model": {"name": "custom"}, "datasets": ["caffe"]})
    effective, _ = resolve_run_config(config, "caffe")
    assert effective.segmentation.learning_rate == 0.02
    assert effective.segmentation.head == "linear"
    config = RunConfig.model_validate(
        {
            **config.model_dump_yaml(),
            "segmentation": {"learning_rate": 0.001, "head": "fpn", "layers": []},
        }
    )
    effective, _ = resolve_run_config(config, "caffe")
    assert effective.segmentation.learning_rate == 0.001
    assert effective.segmentation.head == "fpn"
    assert effective.segmentation.layers == []
