"""Identify synthetic compute measurements and their hardware."""

import hashlib
import json
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import ModelPreset

KEY_COLUMNS = ("name", "band_config", "image_size", "task", "head_type", "config_hash")


@cache
def source_fingerprint() -> str:
    """Identify the implementation and model definitions used for a measurement."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.yaml")]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    packages: dict[str, str | None] = {
        name: version(name) for name in ("torch", "timm", "torchgeo", "transformers")
    }
    for name in ("olmoearth-pretrain-minimal", "terratorch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:  # allow-except: these model backends are optional
            packages[name] = None
    return packages


def measurement_metadata(
    config: FlopsConfig, preset: ModelPreset, device: torch.device
) -> dict[str, Any]:
    """Bind resume to the resolved settings, model options, and GPU model."""
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    values = config.model_dump(mode="json", exclude={"output"})
    values["runtime"].pop("verbose", None)
    values["segmentation"].pop("heads")
    values["segmentation"]["probe"].pop("head")
    identity = {
        "config": values,
        "preset": preset.model_dump(mode="json"),
        "gpu_name": gpu_name,
        "package_versions": _package_versions(),
        "source_hash": source_fingerprint(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "config_hash": digest,
        "band_source": config.input.band_source,
        "model_preset": config.model.name,
        "normalization": config.input.normalization,
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "source_hash": identity["source_hash"],
        "package_versions": json.dumps(identity["package_versions"], sort_keys=True),
        "precision": "float32",
        "flops_convention": "one multiply-add is two operations",
        "flops_coverage": "registered operators only; total coverage unverified",
        "warmup": config.timing.n_warmup,
        "measurements": config.timing.n_measure,
    }


def cell_key(metadata: dict[str, Any], task: str, head: str = "") -> tuple:
    """Return the stored identity for one task and head at the measured image size."""
    row = {**metadata, "task": task, "head_type": head}
    return tuple(row[column] for column in KEY_COLUMNS)
