"""Strict YAML configuration for the image benchmark command."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from .schema import (
    ClassificationConfig,
    InputConfig,
    ModelConfig,
    OutputPaths,
    RuntimeConfig,
    SchemaVersion,
    SegmentationConfig,
    StrictModel,
    load_yaml,
)


class OutputConfig(OutputPaths):
    """Result storage settings."""

    resume: StrictBool = False


class CPUThroughputConfig(StrictModel):
    """Optional bounded CPU measurement alongside an image run."""

    enabled: StrictBool = False
    batch_size: StrictInt = Field(default=8, gt=0)
    n_warmup: StrictInt = Field(default=1, ge=0)
    n_measure: StrictInt = Field(default=5, gt=0)
    time_budget_s: StrictFloat = Field(default=300.0, gt=0)


class FeatureProfileConfig(StrictModel):
    """Additive encoder measurements stored separately from probe scores."""

    enabled: StrictBool = False
    n_warmup: StrictInt = Field(default=3, ge=0)
    n_measure: StrictInt = Field(default=20, gt=0)
    cpu_throughput: CPUThroughputConfig = Field(default_factory=CPUThroughputConfig)


def _default_splits() -> list[Literal["train", "val", "test"]]:
    """Return the default intrinsic-dimension split selection."""
    return ["train"]


class IntrinsicDimensionConfig(StrictModel):
    """Additive feature-dimension and spectrum measurements."""

    enabled: StrictBool = False
    estimators: list[StrictStr] = Field(default_factory=lambda: ["TwoNN", "MLE", "lPCA"])
    splits: list[Literal["train", "val", "test"]] = Field(
        default_factory=_default_splits, min_length=1
    )
    max_samples: StrictInt | None = Field(default=10000, gt=0)
    device: StrictStr | None = None

    @field_validator("estimators", "splits")
    @classmethod
    def validate_selections(cls, values: list[str]) -> list[str]:
        """Require distinct non-empty selections."""
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("selections must contain distinct non-empty names")
        return values


class RunConfig(StrictModel):
    """Complete core image benchmark configuration."""

    schema_version: SchemaVersion = 1
    model: ModelConfig
    datasets: list[StrictStr] = Field(min_length=1)
    input: InputConfig = Field(default_factory=InputConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    profile: FeatureProfileConfig = Field(default_factory=FeatureProfileConfig)
    intrinsic_dim: IntrinsicDimensionConfig = Field(default_factory=IntrinsicDimensionConfig)

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[StrictStr]) -> list[StrictStr]:
        """Reject empty or repeated dataset names."""
        if any(not name.strip() for name in value):
            raise ValueError("datasets must contain non-empty names")
        if len(set(value)) != len(value):
            raise ValueError("datasets must not contain duplicates")
        if "all" in value and len(value) != 1:
            raise ValueError("'all' cannot be combined with other datasets")
        return value


def load_run_config(path: str | Path) -> RunConfig:
    """Load and strictly validate a core image benchmark configuration."""
    return RunConfig.model_validate(load_yaml(path), strict=True)


def validate_run_config(value: dict[str, Any]) -> RunConfig:
    """Validate a mapping after explicit CLI overrides are applied."""
    return RunConfig.model_validate(value, strict=True)
