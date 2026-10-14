# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Shared strict configuration sections and safe YAML loading."""

import pathlib
import re
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""

    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,  # noqa: FBT001, FBT002 - PyYAML passes this positionally
    ) -> dict:
        """Construct a mapping and reject duplicate or unhashable keys."""
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:  # allow-except: report invalid YAML mapping keys
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


class StrictModel(BaseModel):
    """Base for configuration sections with strict fields and no extras."""

    model_config = ConfigDict(
        extra="forbid", strict=True, validate_default=True, allow_inf_nan=False
    )

    def model_dump_yaml(self) -> dict[str, Any]:
        """Return supplied settings without turning omitted defaults into overrides."""
        return self.model_dump(mode="json", exclude_unset=True)


def _check_device(value: str) -> str:
    """Reject malformed device strings without importing Torch."""
    if value == "auto" or re.fullmatch(r"(cpu|cuda(?::[0-9]+)?)", value):
        return value
    raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")


def _check_knn_device(value: str) -> str:
    """Reject malformed KNN device strings; FAISS has no 'auto' selection."""
    if value in {"cpu", "cuda"} or re.fullmatch(r"cuda:[0-9]+", value):
        return value
    raise ValueError("knn_device must be 'cpu', 'cuda', or 'cuda:<index>'")


def _check_methods(value: list["Method"]) -> list["Method"]:
    """Reject repeated probe selections."""
    if len(set(value)) != len(value):
        raise ValueError("methods must not contain duplicates")
    return value


def _check_schema_version(value: object) -> object:
    """Reject non-integer versions before literal validation."""
    if type(value) is not int:
        raise ValueError("schema_version must be the integer 1")  # noqa: TRY004 - Pydantic field validation
    return value


def _check_output_path(value: str) -> str:
    """Reject blank output paths without rewriting valid strings."""
    if not value.strip():
        raise ValueError("output path must not be blank")
    return value


type Method = Literal["knn", "linear"]
type Device = Annotated[StrictStr, AfterValidator(_check_device)]
type KnnDevice = Annotated[StrictStr, AfterValidator(_check_knn_device)]
type Methods = Annotated[list[Method], AfterValidator(_check_methods)]
type SchemaVersion = Annotated[Literal[1], BeforeValidator(_check_schema_version)]
type OutputPath = Annotated[StrictStr, AfterValidator(_check_output_path)]


class OutputPaths(StrictModel):
    """Output root and optional explicit file for CSV benchmark commands."""

    directory: OutputPath = "results"
    file: OutputPath | None = None


def resolve_output_path(
    directory: str | None, file: str | None, relative_path: str | pathlib.Path
) -> str:
    """Return the explicit file or a command's default path under the output root."""
    if file is not None:
        return file
    if directory is None:
        raise ValueError("An output directory or file is required")
    return str(pathlib.Path(directory) / relative_path)


def default_methods() -> list[Method]:
    """Return the default probe selection, shared by image and coordinate runs."""
    return ["knn", "linear"]


class ModelConfig(StrictModel):
    """Selected preset or importable custom model with constructor-only options."""

    name: StrictStr = Field(min_length=1)
    target: StrictStr | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str | None) -> str | None:
        """Require an importable dotted symbol, without importing optional models."""
        if value is not None and (
            "." not in value or any(not part.isidentifier() for part in value.split("."))
        ):
            raise ValueError("target must be a dotted Python symbol")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject whitespace-only model names."""
        if not value.strip():
            raise ValueError("model name must not be blank")
        return value


class InputConfig(StrictModel):
    """Input bands, resizing, dataset partition, and normalization."""

    bands: StrictStr | list[StrictStr] = "rgb"
    partition: StrictStr = "default"
    time_steps: StrictInt | None = Field(default=None, gt=0)
    image_size: StrictInt | None = Field(default=224, gt=0)
    interpolation: Literal["area", "bilinear", "bicubic", "nearest"] = "bilinear"
    normalization: Literal["dataset", "model", "minmax", "minmax_zscore", "none"] = "dataset"

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str) -> str:
        """Reject whitespace-only partition names."""
        if not value.strip():
            raise ValueError("partition must not be blank")
        return value

    @field_validator("bands")
    @classmethod
    def validate_bands(cls, value: StrictStr | list[StrictStr]) -> StrictStr | list[StrictStr]:
        """Reject empty band specifications."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("bands must not be empty")
        if isinstance(value, list) and (
            not value or any(not band.strip() for band in value) or len(set(value)) != len(value)
        ):
            raise ValueError("bands must contain non-empty names")
        return value


class LinearConfig(StrictModel):
    """Linear probe hyperparameters."""

    c_log10_start: StrictFloat = -6.0
    c_log10_stop: StrictFloat = 4.0
    c_count: StrictInt = Field(default=40, gt=0)
    refit_train_val: StrictBool = True

    @model_validator(mode="after")
    def validate_range(self) -> "LinearConfig":
        """Require an ascending regularization range."""
        if self.c_log10_stop < self.c_log10_start:
            raise ValueError("c_log10_stop must be greater than or equal to c_log10_start")
        return self


class CalibrationConfig(StrictModel):
    """Classification calibration metrics."""

    n_bins_knn: StrictInt | None = Field(default=None, gt=0)
    n_bins_linear: StrictInt = Field(default=15, gt=0)
    temp_scale: StrictBool = False


class ClassificationConfig(StrictModel):
    """KNN, linear probe, and bootstrap settings."""

    methods: Methods = Field(default_factory=default_methods, min_length=1)
    knn_k: StrictInt = Field(default=5, gt=0)
    knn_device: KnnDevice | None = None
    linear: LinearConfig = Field(default_factory=LinearConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    bootstrap_samples: StrictInt = Field(default=200, gt=0)

    @model_validator(mode="after")
    def validate_calibration(self) -> "ClassificationConfig":
        """Require held-out validation when temperature scaling is enabled."""
        if self.calibration.temp_scale and (
            "linear" not in self.methods or self.linear.refit_train_val
        ):
            raise ValueError("temp_scale requires linear selected and refit_train_val=false")
        return self


class EarlyStoppingConfig(StrictModel):
    """Stop cached probe training once validation mIoU stops improving."""

    enabled: StrictBool = False
    check_every: StrictInt = Field(default=5, gt=0)
    patience: StrictInt = Field(default=16, gt=0)
    min_delta: StrictFloat = Field(default=0.001, ge=0)
    min_epochs: StrictInt = Field(default=25, ge=0)
    max_epochs: StrictInt = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def validate_horizon(self) -> "EarlyStoppingConfig":
        """Require room to reach the minimum training length."""
        if self.max_epochs < self.min_epochs:
            raise ValueError("max_epochs must be greater than or equal to min_epochs")
        return self


def _check_learning_rates(value: list[float]) -> list[float]:
    """Reject non-positive or repeated grid values."""
    if any(rate <= 0 for rate in value) or len(set(value)) != len(value):
        raise ValueError("learning_rates must be positive and unique")
    return value


class SegmentationConfig(StrictModel):
    """Segmentation probe settings."""

    head: Literal["linear", "conv_block", "fpn", "dpt", "patch_linear"] = "fpn"
    layers: list[StrictStr] = Field(default_factory=list)
    learning_rate: StrictFloat = Field(default=1e-3, gt=0)
    learning_rates: Annotated[list[StrictFloat], AfterValidator(_check_learning_rates)] = Field(
        default_factory=list
    )
    epochs: StrictInt = Field(default=10, gt=0)
    batch_size: StrictInt = Field(default=64, gt=0)
    temporal_pool: Literal["mean", "max"] = "mean"
    scheduler: Literal["cosine", "none"] = "cosine"
    ignore_index: StrictInt = 255
    cache_features: StrictBool = True
    cache_dtype: Literal["float16", "float32"] = "float16"
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    @model_validator(mode="after")
    def validate_fitting(self) -> "SegmentationConfig":
        """Reject fitting options that the chosen training path cannot honor."""
        if self.early_stopping.enabled and self.scheduler != "none":
            raise ValueError(
                "early_stopping requires scheduler 'none'; cosine needs a fixed number of epochs"
            )
        if (self.early_stopping.enabled or self.learning_rates) and not self.cache_features:
            raise ValueError("early_stopping and learning_rates require cache_features=true")
        return self


class RuntimeConfig(StrictModel):
    """Execution settings."""

    device: Device = "cuda:0"
    batch_size: StrictInt = Field(default=64, gt=0)
    workers: StrictInt = Field(default=4, ge=0)
    seed: StrictInt = 0
    verbose: StrictBool = False


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """Load one YAML mapping with safe tags and unique keys."""
    with pathlib.Path(path).open(encoding="utf-8") as file:
        value = yaml.load(file, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be a YAML mapping")  # noqa: TRY004 - preserve the CLI configuration-error contract
    return value
