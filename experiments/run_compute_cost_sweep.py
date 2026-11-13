#!/usr/bin/env python3
"""Measure compute cost across the current model catalog, image sizes, and decoders."""

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch
import yaml
from timm.models import get_pretrained_cfg, is_model_in_modules

from geo_eval_harness.config.catalog import list_model_configs
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import ModelPreset, load_model_preset
from geo_eval_harness.config.schema import ModelConfig, load_yaml
from geo_eval_harness.devices import resolve_device
from geo_eval_harness.flops_identity import measurement_metadata, source_fingerprint
from geo_eval_harness.models.terrafm import TERRAFM_INPUT_SIZE
from geo_eval_harness.models.torchgeo_models import _PANOPTICON_PATCH_SIZE

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


@cache
def benchmark_targets() -> dict[str, str]:
    """Read names and constructors that have checked-in benchmark measurements."""
    targets = {}
    for path in sorted((ROOT / "results" / "models").glob("*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                targets[row.get("name") or path.stem] = row["model"]
    return targets


def model_selection(name: str) -> ModelConfig:
    """Resolve catalog entries and the parameter-free ImageStats study control."""
    if name in list_model_configs():
        return ModelConfig(name=name)
    if benchmark_targets().get(name) in {
        "geo_eval_harness.models.ImageStatsBench",
        "torchgeo_bench.models.ImageStatsBench",
    }:
        return ModelConfig(name=name, target="geo_eval_harness.models.ImageStatsBench")
    raise ValueError(f"No model configuration for benchmark name {name!r}")


@dataclass(frozen=True)
class Job:
    """One model, input resolution, and source-band selection."""

    preset: str
    size: int
    bands: str
    source: str
    heads: tuple[str, ...]

    @property
    def name(self) -> str:
        """Return a stable filename component."""
        return f"{self.preset.replace('/', '__')}__{self.bands}__{self.size}"


def image_grid_size(preset: ModelPreset, size: int) -> int:
    """Resolve a supported image grid without changing learned patch or pooling weights."""
    if size <= 0:
        return size
    if preset.target == "geo_eval_harness.models.TorchGeoEarthLocBench":
        return int(preset.kwargs["target_size"])
    if preset.target == "geo_eval_harness.models.TorchGeoPanopticonBench":
        return max(
            _PANOPTICON_PATCH_SIZE, round(size / _PANOPTICON_PATCH_SIZE) * _PANOPTICON_PATCH_SIZE
        )
    return size


def build_jobs(models: list[str], sizes: list[int], heads: list[str]) -> list[Job]:
    """Discover image models and their declared modality without loading weights."""
    jobs = []
    names = models or list_model_configs()
    if not models:
        names += [
            name
            for name, target in benchmark_targets().items()
            if target.endswith(".ImageStatsBench")
            and name not in {load_model_preset(ModelConfig(name=n)).name for n in names}
        ]
    for name in names:
        preset = load_model_preset(model_selection(name))
        if preset.track != "image":
            continue
        modality = preset.kwargs.get("modality")
        source = "kuro_siwo" if modality in ("s1", "S1GRD") else "cloudsen12"
        if modality in ("s1", "S1GRD"):
            selections = ["all"]
        elif modality == "RGB":
            selections = ["rgb"]
        else:
            selections = ["rgb", "all"]
        selected = preset.for_dataset(source)
        selected_heads = tuple(heads) if selected.segmentation.layers else ()
        for size in dict.fromkeys(image_grid_size(selected, value) for value in sizes):
            jobs.extend(Job(name, size, band, source, selected_heads) for band in selections)
    return jobs


def resolution_kwargs(preset: ModelPreset, size: int) -> dict[str, Any]:
    """Set the actual wrapper resize and rebuild fixed encoder grids at this size."""
    kwargs: dict[str, Any] = {"target_size": size} if "target_size" in preset.kwargs else {}
    if preset.target == "geo_eval_harness.models.TimmPatchBenchModel":
        kwargs["target_size"] = size
        model_name = preset.kwargs["model_name"]
        if is_model_in_modules(model_name, {"swin_transformer", "swin_transformer_v2"}):
            kwargs["dynamic_img_size"] = True
        pretrained = get_pretrained_cfg(model_name)
        if pretrained is not None and pretrained.fixed_input_size:
            kwargs["img_size"] = size
    elif preset.target in {
        "geo_eval_harness.models.TorchGeoDOFABench",
        "geo_eval_harness.models.TorchGeoPanopticonBench",
    }:
        kwargs["img_size"] = size
    elif preset.target == "geo_eval_harness.models.TorchGeoCromaBench":
        kwargs["factory_kwargs"] = {**preset.kwargs.get("factory_kwargs", {}), "image_size": size}
    elif preset.target == "geo_eval_harness.models.TerraFMBench":
        kwargs.update(target_size=TERRAFM_INPUT_SIZE, auto_resize=False)
    if preset.target == "geo_eval_harness.models.AnySatBenchModel":
        kwargs["input_gsd"] = None
    elif preset.target == "geo_eval_harness.models.UniverSatBenchModel":
        kwargs["input_res"] = None
    return kwargs


def job_config(job: Job, args: argparse.Namespace) -> FlopsConfig:
    """Vary image dimensions while preserving the encoder's learned patch stride."""
    selection = model_selection(job.preset)
    preset = load_model_preset(selection).for_dataset(job.source)
    kwargs = resolution_kwargs(preset, job.size)
    kwargs.update(args.model_kwargs.get(job.preset, {}))
    return FlopsConfig.model_validate(
        {
            "model": {**selection.model_dump(exclude_unset=True), "kwargs": kwargs},
            "runtime": {"device": args.device, "seed": 0},
            "input": {
                "band_source": job.source,
                "band_configs": [job.bands],
                "image_size": job.size,
                **({"normalization": "none"} if selection.target is not None else {}),
            },
            "segmentation": {
                "heads": list(job.heads),
                "band_configs": [job.bands],
                "num_classes": args.num_classes,
                "neck_policy": "auto",
            },
            "timing": {
                "batch_size": args.batch_size,
                "n_warmup": args.warmup,
                "n_measure": args.measurements,
            },
            "output": {"file": str(args.output / "compute_cost.csv"), "resume": True},
        }
    )


def _valid_measurement(row: dict[str, str]) -> bool:
    values = (row.get("gflops_total"), row.get("throughput_samples_per_sec"))
    if any(value in (None, "") for value in values):
        return False
    gflops, throughput = (float(value) for value in values)
    return math.isfinite(gflops) and gflops >= 0 and math.isfinite(throughput) and throughput > 0


def measured_cells(path: Path, config_hash: str) -> set[tuple[str, str]]:
    """Read completed cells for this exact implementation, configuration, and hardware."""
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {
            (row["task"], row["head_type"])
            for row in csv.DictReader(stream)
            if row.get("config_hash") == config_hash and _valid_measurement(row)
        }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def sync_results(args: argparse.Namespace) -> None:
    """Copy durable results and diagnostics to the requested private S3 prefix."""
    if args.s3_prefix:
        subprocess.run(
            ["aws", "s3", "sync", str(args.output), args.s3_prefix, "--only-show-errors"],
            check=True,
        )


def run_job(job: Job, args: argparse.Namespace, deadline: float) -> dict[str, Any]:
    """Measure a job and attempt remaining heads independently after a partial failure."""
    config = job_config(job, args)
    effective, preset = config.resolve()
    python = sys.executable
    if args.olmoearth_v13_python and preset.kwargs.get("version") == "v1_3":
        python = args.olmoearth_v13_python
    path = args.output / "configs" / f"{job.name}.yaml"
    log_path = args.output / "logs" / f"{job.name}.log"
    path.write_text(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False))
    expected = {("classification", ""), *(("segmentation", head) for head in job.heads)}
    metadata = measurement_metadata(effective, preset, resolve_device(args.device))
    if python != sys.executable:
        script = (
            "import json, sys; from geo_eval_harness.config.flops import FlopsConfig; "
            "from geo_eval_harness.config.schema import load_yaml; "
            "from geo_eval_harness.devices import resolve_device; "
            "from geo_eval_harness.flops_identity import measurement_metadata; "
            "c,p=FlopsConfig.model_validate(load_yaml(sys.argv[1])).resolve(); "
            "print(json.dumps(measurement_metadata(c,p,resolve_device(c.runtime.device))))"
        )
        result = subprocess.run(
            [python, "-c", script, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=args.cell_timeout,
            check=True,
        )
        metadata = json.loads(result.stdout)
    completed = measured_cells(args.output / "compute_cost.csv", metadata["config_hash"])
    started = time.monotonic()
    codes = []
    selections = [list(job.heads)]
    for selected in selections:
        remaining = int(deadline - time.monotonic())
        if expected <= completed or remaining <= 0:
            break
        config.segmentation.heads = selected
        path.write_text(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False))
        command = [python, "-m", "geo_eval_harness", "flops", "--config", str(path)]
        with log_path.open("a") as log:
            log.write(f"\ncommand: {command!r}\n")
            log.flush()
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(args.cell_timeout, remaining),
                    check=False,
                )
                codes.append(result.returncode)
            except subprocess.TimeoutExpired:  # allow-except: retain timed-out cells as failed
                codes.append(124)
        completed = measured_cells(args.output / "compute_cost.csv", metadata["config_hash"])
        if len(selections) == 1 and ("classification", "") in completed:
            selections.extend(
                [head] for head in job.heads if ("segmentation", head) not in completed
            )
    return {
        "job": job.name,
        "preset": job.preset,
        "image_size": job.size,
        "band_config": job.bands,
        "band_source": job.source,
        "config_hash": metadata["config_hash"],
        "status": "complete" if expected <= completed else "incomplete",
        "expected": sorted(expected),
        "measured": sorted(expected & completed),
        "missing": sorted(expected - completed),
        "exit_codes": codes,
        "seconds": round(time.monotonic() - started, 3),
        "log": str(log_path.relative_to(args.output)),
    }


def main() -> int:
    """Write the complete job manifest, then run one GPU measurement at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[224, 256, 512])
    parser.add_argument("--heads", nargs="+", default=["linear", "upernet", "unet"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measurements", type=int, default=20)
    parser.add_argument("--cell-timeout", type=int, default=600)
    parser.add_argument("--max-runtime-minutes", type=int, default=420)
    parser.add_argument("--olmoearth-v13-python")
    parser.add_argument("--model-kwargs-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "compute_sweep")
    parser.add_argument("--s3-prefix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model_kwargs = load_yaml(args.model_kwargs_file) if args.model_kwargs_file else {}
    args.output = args.output.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for directory in (args.output, args.output / "configs", args.output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.models, args.image_sizes, args.heads)
    configs = [job_config(job, args).model_dump_yaml() for job in jobs]
    configured = {load_model_preset(model_selection(job.preset)).name for job in jobs}
    unmapped = sorted(set(benchmark_targets()) - configured)
    _write_json(
        args.output / "manifest.json",
        {
            "source_hash": source_fingerprint(),
            "jobs": configs,
            "precision": "float32",
            "requested_image_sizes": args.image_sizes,
            "effective_image_sizes": sorted({job.size for job in jobs}),
            "benchmark_names": len(benchmark_targets()),
            "unmapped_benchmark_names": unmapped,
            "requested_model_subset": bool(args.models),
        },
    )
    logger.info("Prepared %d model/band/size jobs", len(jobs))
    if args.dry_run:
        return 0
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    deadline = time.monotonic() + args.max_runtime_minutes * 60
    reports = []
    try:
        for index, job in enumerate(jobs):
            if time.monotonic() >= deadline:
                break
            report = run_job(job, args, deadline)
            reports.append(report)
            _write_json(args.output / "coverage.json", reports)
            logger.info(
                "%d/%d %s: %s (%d/%d cells)",
                index + 1,
                len(jobs),
                job.name,
                report["status"],
                len(report["measured"]),
                len(report["expected"]),
            )
            if (index + 1) % 5 == 0:
                sync_results(args)
    finally:
        _write_json(
            args.output / "summary.json",
            {
                "jobs": len(jobs),
                "attempted": len(reports),
                "complete": sum(r["status"] == "complete" for r in reports),
                "pending": len(jobs) - len(reports),
                "unmapped_benchmark_names": unmapped,
            },
        )
        sync_results(args)
    return 0 if len(reports) == len(jobs) and all(r["status"] == "complete" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
