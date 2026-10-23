"""Typed model presets and effective image settings.

Only explicitly supplied values override preset defaults. Runtime objects such as
BandSpecs and empirical-RCF datasets are passed directly to construction, never YAML.
"""

import importlib
from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, StrictBool

from geo_eval_harness.errors import UnsupportedNormalizationError

from .run import RunConfig
from .schema import (
    ClassificationConfig,
    InputConfig,
    ModelConfig,
    SegmentationConfig,
    StrictModel,
    load_yaml,
)

NORMALIZATIONS = {
    "dataset": "bandspec_zscore",
    "model": "model_native",
    "minmax": "minmax",
    "minmax_zscore": "minmax_zscore",
    "none": "identity",
}


def merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit YAML settings; nulls and empty lists replace prior values."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class PresetDefaults(StrictModel):
    """Constructor and benchmark defaults for one dataset or model."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    input: InputConfig = Field(default_factory=InputConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)


class ModelPreset(PresetDefaults):
    """A named constructor, separate from preprocessing and evaluation metadata."""

    name: str
    target: str
    track: Literal["image", "coord"] = "image"
    seed_from_run: bool = False
    supports_model_normalization: StrictBool | None = None
    dataset_overrides: dict[str, PresetDefaults] = Field(default_factory=dict)

    def validate_normalization(self, normalization: str) -> None:
        """Reject explicitly unsupported strategies; undeclared capabilities stay runtime checks."""
        if normalization == NORMALIZATIONS["model"] and self.supports_model_normalization is False:
            raise UnsupportedNormalizationError(
                f"{self.name!r} does not support --normalization model; use --normalization dataset"
            )

    def for_dataset(self, dataset: str) -> "ModelPreset":
        """Apply a dataset override without mutating the shared preset."""
        values = self.model_dump(exclude_unset=True)
        values.pop("dataset_overrides", None)
        override = self.dataset_overrides.get(dataset)
        if override is not None:
            values = merge_settings(values, override.model_dump(exclude_unset=True))
        return ModelPreset.model_validate(values)


def load_model_preset(selection: ModelConfig, *, seed: int = 0) -> ModelPreset:
    """Load a packaged preset or a custom constructor without importing weights."""
    if selection.target is not None:
        return ModelPreset(name=selection.name, target=selection.target, kwargs=selection.kwargs)
    from .catalog import model_config_path

    preset = ModelPreset.model_validate(load_yaml(model_config_path(selection.name)))
    kwargs = dict(preset.kwargs)
    if preset.seed_from_run:
        kwargs["seed"] = seed
    kwargs.update(selection.kwargs)
    return preset.model_copy(update={"kwargs": kwargs})


def resolve_run_config(config: RunConfig, dataset: str) -> tuple[RunConfig, ModelPreset]:
    """Resolve preset and dataset defaults beneath explicitly supplied settings."""
    preset = load_model_preset(config.model, seed=config.runtime.seed).for_dataset(dataset)
    preset = preset.model_copy(update={"kwargs": {**preset.kwargs, **config.model.kwargs}})
    if preset.track != "image":
        raise ValueError(f"{config.model.name!r} is a coordinate encoder; use 'coord'")
    defaults = {
        key: getattr(preset, key).model_dump(exclude_unset=True)
        for key in ("input", "classification", "segmentation")
    }
    effective = RunConfig.model_validate(merge_settings(defaults, config.model_dump_yaml()))
    segmentation = effective.segmentation
    if segmentation.lora_rank is not None and segmentation.lora_target_modules is None:
        raise ValueError(
            f"segmentation.lora_rank requires segmentation.lora_target_modules; "
            f"{preset.name!r} does not define LoRA target modules"
        )
    return effective, preset.model_copy(update={"input": effective.input})


def build_model(preset: ModelPreset, **runtime_options: Any) -> Any:
    """Construct one model; nested target-like kwargs remain ordinary mappings."""
    options = {**preset.kwargs, **runtime_options}
    preset.validate_normalization(options.get("normalization", NORMALIZATIONS["dataset"]))
    module, _, symbol = preset.target.rpartition(".")
    constructor = getattr(importlib.import_module(module), symbol)
    if getattr(constructor, "wants_resolved_image_size", False):
        options.setdefault("image_size", preset.input.image_size)
    validated = getattr(constructor, "validated_settings", None)
    if validated is not None:
        bands = options.pop("bands")
        normalization = options.pop("normalization", "bandspec_zscore")
        return validated(**options).build(bands, normalization=normalization)
    return constructor(**options)
