"""Strict settings for profiling one fixed, real dataset batch."""

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator

from ..datasets import list_datasets
from .presets import ModelPreset, resolve_run_config
from .run import RunConfig
from .schema import (
    Device,
    InputConfig,
    ModelConfig,
    OutputPath,
    RuntimeConfig,
    SchemaVersion,
    StrictModel,
)


class ProfileRuntimeConfig(RuntimeConfig):
    """Execution defaults for a bounded standalone measurement."""

    device: Device = "cpu"
    batch_size: StrictInt = Field(default=32, gt=0)
    workers: StrictInt = Field(default=0, ge=0)
    seed: StrictInt = Field(default=0, ge=0, le=2**32 - 1)


class ProfileOutputConfig(StrictModel):
    """Write JSON to stdout unless a directory or file is supplied."""

    directory: OutputPath | None = None
    file: OutputPath | None = None


class ProfileConfig(StrictModel):
    """Model, input, and timing settings for the standalone profile command."""

    schema_version: SchemaVersion = 1
    model: ModelConfig
    dataset: StrictStr = Field(min_length=1)
    input: InputConfig = Field(default_factory=InputConfig)
    runtime: ProfileRuntimeConfig = Field(default_factory=ProfileRuntimeConfig)
    output: ProfileOutputConfig = Field(default_factory=ProfileOutputConfig)
    warmup: StrictInt = Field(default=3, ge=0)
    measurements: StrictInt = Field(default=20, gt=0)
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    count_flops: StrictBool = False

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: str) -> str:
        """Require a nonblank dataset name."""
        if not value.strip():
            raise ValueError("dataset must not be blank")
        return value

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: InputConfig) -> InputConfig:
        """Require named band lists rather than an ambiguous string."""
        if isinstance(value.bands, str) and value.bands not in {"rgb", "all"}:
            raise ValueError("input.bands must be rgb, all, or a YAML list of band names")
        return value


def resolve_profile_config(config: ProfileConfig) -> tuple[ProfileConfig, ModelPreset]:
    """Apply shared model/dataset defaults without loading models or data."""
    if config.dataset not in list_datasets():
        raise ValueError(f"unknown dataset {config.dataset!r}")
    run = RunConfig.model_validate(
        {
            "model": config.model.model_dump(exclude_unset=True),
            "datasets": [config.dataset],
            "input": config.input.model_dump(exclude_unset=True),
            "runtime": config.runtime.model_dump(exclude_unset=True),
        }
    )
    effective, preset = resolve_run_config(run, config.dataset)
    return config.model_copy(update={"input": effective.input}), preset
