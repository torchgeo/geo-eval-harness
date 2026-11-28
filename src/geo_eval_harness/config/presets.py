"""Typed model presets and effective image settings.

Only explicitly supplied values override preset defaults. Runtime objects such as
BandSpecs and empirical-RCF datasets are passed directly to construction, never YAML.
"""

import importlib
from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictStr

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.errors import UnsupportedNormalizationError

from .run import RunConfig
from .schema import (
    Adaptation,
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
    flops_band_config: Literal["rgb", "all"] | None = None
    seed_from_run: bool = False
    supports_model_normalization: StrictBool | None = None
    supported_datasets: list[StrictStr] | None = Field(default=None, min_length=1)
    dataset_overrides: dict[str, PresetDefaults] = Field(default_factory=dict)

    def validate_normalization(self, normalization: str) -> None:
        """Reject explicitly unsupported strategies; undeclared capabilities stay runtime checks."""
        if normalization == NORMALIZATIONS["model"] and self.supports_model_normalization is False:
            raise UnsupportedNormalizationError(
                f"{self.name!r} does not support --normalization model; use --normalization dataset"
            )

    def for_dataset(self, dataset: str) -> "ModelPreset":
        """Apply a dataset override without mutating the shared preset."""
        if self.supported_datasets is not None and dataset not in self.supported_datasets:
            raise BandCompatibilityError(
                f"{self.name!r} does not support dataset {dataset!r}; "
                f"verified datasets: {self.supported_datasets}."
            )
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
    defaults = {
        key: getattr(preset, key).model_dump(exclude_unset=True)
        for key in ("input", "classification", "segmentation")
    }
    effective = RunConfig.model_validate(merge_settings(defaults, config.model_dump_yaml()))
    segmentation = effective.segmentation
    if segmentation.adaptation is Adaptation.LORA and segmentation.lora_target_modules is None:
        raise ValueError(
            f"segmentation.lora_rank requires segmentation.lora_target_modules; "
            f"{preset.name!r} does not define LoRA target modules"
        )
    return effective, preset.model_copy(update={"input": effective.input})


def model_constructor(preset: ModelPreset) -> Any:
    """Import the class or factory named by a preset's ``target``."""
    module, _, symbol = preset.target.rpartition(".")
    return getattr(importlib.import_module(module), symbol)


def recorded_normalization(preset: ModelPreset, requested: str) -> str:
    """Return the normalization a model's result rows record, without building the model."""
    fixed = getattr(model_constructor(preset), "fixed_normalization", None)
    return requested if fixed is None else str(fixed)


def _gsd_scale(preset: ModelPreset, resized_from: tuple[int, int] | None) -> float:
    """Return the ground size of an input pixel relative to a pixel of the unresized tile."""
    image_size = preset.input.image_size
    if image_size is None:
        return 1.0
    if resized_from is None:
        raise ValueError(
            f"{preset.name!r} needs the tiles' size before the resize to {image_size} px to know "
            "the ground size of its input pixels; pass resized_from to build_model."
        )
    height, width = resized_from
    if height != width:
        raise ValueError(
            f"{preset.name!r} cannot give {height}x{width} tiles resized to {image_size} px one "
            "ground resolution; use --image-size none."
        )
    return height / image_size


def build_model(
    preset: ModelPreset,
    *,
    native_image_size: int | None = None,
    resized_from: tuple[int, int] | None = None,
    **runtime_options: Any,
) -> Any:
    """Construct one model; nested target-like kwargs remain ordinary mappings.

    ``resized_from`` is the ``(height, width)`` the tiles had before the run's resize, as the
    resize transform records it on each sample. Constructors declaring ``wants_gsd_scale`` receive
    ``gsd_scale``, that size over ``input.image_size`` (1.0 for unresized tiles).
    """
    options = {**preset.kwargs, **runtime_options}
    preset.validate_normalization(options.get("normalization", NORMALIZATIONS["dataset"]))
    constructor = model_constructor(preset)
    if getattr(constructor, "wants_resolved_image_size", False):
        image_size = preset.input.image_size
        if image_size is None and getattr(constructor, "wants_native_image_size", False):
            image_size = native_image_size
        options["image_size"] = image_size
    if getattr(constructor, "wants_gsd_scale", False):
        options["gsd_scale"] = _gsd_scale(preset, resized_from)
    validated = getattr(constructor, "validated_settings", None)
    if validated is not None:
        bands = options.pop("bands")
        normalization = options.pop("normalization", "bandspec_zscore")
        return validated(**options).build(bands, normalization=normalization)
    return constructor(**options)
