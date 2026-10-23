# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Runtime implementation for the standalone profile command."""

import hashlib
import json
import platform
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..config.presets import NORMALIZATIONS, ModelPreset, build_model
from ..config.profile import ProfileConfig, resolve_profile_config
from ..config.schema import resolve_output_path
from ..datasets import BandSpec, get_bench_dataset_class, get_datasets
from ..devices import resolve_device
from ..model_profile import ProfileResult, profile_inference


def _load_batch(config: ProfileConfig) -> tuple[Dataset, torch.Tensor, list[BandSpec]]:
    """Load one full batch and its ordered band metadata."""
    inputs = config.input
    dataset = get_bench_dataset_class(config.dataset)()
    band_names = inputs.bands
    selected = dataset.resolve_band_specs(band_names)
    train_dataset, train_loader, _, _ = get_datasets(
        dataset_name=config.dataset,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.workers,
        return_val=True,
        image_size=inputs.image_size,
        interpolation=inputs.interpolation,
        bands=band_names,
        partition_name=inputs.partition,
        time_steps=inputs.time_steps,
    )
    batch = next(iter(train_loader))["image"]
    if batch.shape[0] != config.runtime.batch_size:
        raise RuntimeError(
            f"dataset returned batch size {batch.shape[0]}, requested {config.runtime.batch_size}"
        )
    return train_dataset, batch, selected


def _construction_preset(
    preset: ModelPreset, config: ProfileConfig, sample: torch.Tensor
) -> ModelPreset:
    """Supply the input-dependent shape required by Scale-MAE's positional embedding."""
    if preset.target in {
        "geo_eval_harness.models.TorchGeoScaleMAEBench",
        "geo_eval_harness.models.torchgeo_models.TorchGeoScaleMAEBench",
    }:
        size = config.input.image_size
        kwargs = {**preset.kwargs, "image_size": sample.shape[-1] if size is None else size}
        return preset.model_copy(update={"kwargs": kwargs})
    return preset


def _build_model(
    preset: ModelPreset,
    config: ProfileConfig,
    train_dataset: Dataset,
    selected_bands: list[BandSpec],
) -> nn.Module:
    """Inject runtime band metadata and empirical filters' source dataset."""
    options: dict[str, Any] = {
        "bands": selected_bands,
        "normalization": NORMALIZATIONS[config.input.normalization],
    }
    if (
        preset.target
        in {
            "geo_eval_harness.models.RCFBench",
            "geo_eval_harness.models.rcf.RCFBench",
        }
        and preset.kwargs.get("mode") == "empirical"
    ):
        options["dataset"] = train_dataset
    return build_model(preset, **options)


def _record(
    config: ProfileConfig,
    preset: ModelPreset,
    selected_bands: list[BandSpec],
    sample: torch.Tensor,
    result: ProfileResult,
) -> dict[str, Any]:
    """Attach reproducibility metadata without serializing runtime objects."""
    device = sample.device
    normalization = NORMALIZATIONS[config.input.normalization]
    resolved_model_config = {
        "name": preset.name,
        "target": preset.target,
        "kwargs": preset.model_dump(mode="json")["kwargs"],
        "input": config.input.model_dump(mode="json"),
    }
    model_json = json.dumps(resolved_model_config, sort_keys=True)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": config.model.name,
        "dataset": config.dataset,
        "seed": config.runtime.seed,
        "bands": [spec.name for spec in selected_bands],
        "normalization": normalization,
        "input_normalization": preset.kwargs.get("input_normalization", normalization),
        "dataset_partition": config.input.partition,
        "sample_sha256": hashlib.sha256(sample.detach().cpu().numpy().tobytes()).hexdigest(),
        "model_config": resolved_model_config,
        "model_config_hash": hashlib.sha256(model_json.encode()).hexdigest(),
        "device_index": device.index,
        "input_shape": list(sample.shape),
        "image_size": config.input.image_size,
        "interpolation": config.input.interpolation,
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.platform()
        ),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "scope": "encoder inference on one real dataset batch",
        "profile": asdict(result),
    }


def run(config: ProfileConfig) -> None:
    """Load one real dataset batch, measure it, and write one JSON record."""
    config, preset = resolve_profile_config(config)
    try:
        device = resolve_device(config.runtime.device)
    except ValueError as error:  # allow-except: report unavailable or invalid requested devices
        raise SystemExit(f"error: {error}") from error
    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)  # noqa: NPY002 - Dataset transforms use NumPy's global RNG.
    # Upstream dataset/model constructors may print diagnostics; stdout is one JSON record.
    with redirect_stdout(sys.stderr):
        train_dataset, batch, selected_bands = _load_batch(config)
        preset = _construction_preset(preset, config, batch)
        model = _build_model(preset, config, train_dataset, selected_bands).to(device).eval()
        sample = batch.to(device)
        result = profile_inference(
            model,
            sample,
            device=device,
            precision=config.precision,
            n_warmup=config.warmup,
            n_measure=config.measurements,
            count_flops=config.count_flops,
        )
    payload = json.dumps(_record(config, preset, selected_bands, sample, result), allow_nan=False)
    if config.output.directory is None and config.output.file is None:
        print(payload)
    else:
        path = Path(
            resolve_output_path(config.output.directory, config.output.file, "profile.json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
