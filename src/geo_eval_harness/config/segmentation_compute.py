"""Strict settings for segmentation compute measurements (``geo-eval-harness seg-flops``)."""

from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from .schema import Device, OutputPaths, SchemaVersion, StrictModel

type Kind = Literal["inference", "train_unit", "canonical"]
type BandConfig = Literal["rgb", "s2"]

# Canonical decoder arms: decoder, neck, upsampler.
ARMS: dict[str, dict[str, str]] = {
    "upernet": {"head": "upernet", "neck": "none", "upsampler": "none"},
    "unet": {"head": "unet", "neck": "none", "upsampler": "none"},
    "upernet+vit": {"head": "upernet", "neck": "vit", "upsampler": "none"},
    "unet+vit": {"head": "unet", "neck": "vit", "upsampler": "none"},
    "upernet+anyup": {"head": "upernet", "neck": "none", "upsampler": "anyup"},
}
type Arm = Literal["upernet", "unet", "upernet+vit", "unet+vit", "upernet+anyup"]

# ``s2`` keeps the classification table's meaning: every band of the band source.
BAND_SELECTIONS: dict[str, str] = {"rgb": "rgb", "s2": "all"}


def _kinds() -> list[Kind]:
    return ["inference", "train_unit", "canonical"]


def _arms() -> list[Arm]:
    return ["upernet", "unet", "upernet+vit", "unet+vit", "upernet+anyup"]


def _band_configs() -> list[BandConfig]:
    return ["rgb", "s2"]


class SegFlopsCell(StrictModel):
    """One dataset plus a partial ``run`` configuration (model, input, segmentation)."""

    dataset: StrictStr = Field(min_length=1)
    run: dict[str, Any] = Field(default_factory=dict)
    label: StrictStr = ""

    @field_validator("run")
    @classmethod
    def validate_run(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require a model and leave the dataset to ``dataset``."""
        if "model" not in value:
            raise ValueError("run.model is required")
        if "datasets" in value:
            raise ValueError("set the cell's dataset, not run.datasets")
        return value


class SegFlopsCanonicalConfig(StrictModel):
    """Leaderboard summary cells: each model x arm x band config on one band source."""

    models: list[StrictStr] = Field(default_factory=list)
    arms: list[Arm] = Field(default_factory=_arms)
    band_source: StrictStr = "cloudsen12"
    band_configs: list[BandConfig] = Field(default_factory=_band_configs)
    image_size: StrictInt = Field(default=224, gt=0)

    @field_validator("models", "arms", "band_configs")
    @classmethod
    def validate_unique(cls, value: list[str]) -> list[str]:
        """Reject repeated selections."""
        if len(set(value)) != len(value):
            raise ValueError("selections must not contain duplicates")
        return value


class SegFlopsMeasureConfig(StrictModel):
    """Which row kinds to measure."""

    kinds: list[Kind] = Field(default_factory=_kinds, min_length=1)
    train_batch_size: StrictInt = Field(default=2, ge=2)


class SegFlopsRuntimeConfig(StrictModel):
    """Device and model initialization."""

    device: Device = "cuda"
    seed: StrictInt = 0
    verbose: StrictBool = True
    fail_fast: StrictBool = False


class SegFlopsOutputConfig(OutputPaths):
    """Append-only CSV, resumed by compute key."""

    resume: StrictBool = True


class SegFlopsConfig(StrictModel):
    """Cells from YAML, a manifest file, and/or the canonical grid."""

    schema_version: SchemaVersion = 1
    cells: list[SegFlopsCell] = Field(default_factory=list)
    manifest: StrictStr | None = None
    canonical: SegFlopsCanonicalConfig = Field(default_factory=SegFlopsCanonicalConfig)
    measure: SegFlopsMeasureConfig = Field(default_factory=SegFlopsMeasureConfig)
    runtime: SegFlopsRuntimeConfig = Field(default_factory=SegFlopsRuntimeConfig)
    output: SegFlopsOutputConfig = Field(default_factory=SegFlopsOutputConfig)

    @model_validator(mode="after")
    def validate_work(self) -> "SegFlopsConfig":
        """Require at least one source of cells."""
        if not self.cells and self.manifest is None and not self.canonical.models:
            raise ValueError("nothing to measure: give cells, a manifest, or canonical.models")
        return self
