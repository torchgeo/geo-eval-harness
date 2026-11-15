"""Plan a classification refresh using the current model and dataset catalogs.

Write the manifest with ``python experiments/plan_classification_refresh.py --output plan.json``.
Each job contains a public CLI configuration; save it as YAML and run
``geo-eval-harness run --config job.yaml --output-dir <fresh-directory>``.
"""

import argparse
import hashlib
import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from geo_eval_harness.config.catalog import list_model_configs
from geo_eval_harness.config.presets import ModelPreset, load_model_preset, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_dataset_task, list_datasets
from geo_eval_harness.resume import resume_config_hash

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MEAN_POOL_TARGETS = (
    "SmartiesBench",
    "TerraFMBench",
    "TerraTorchClayBench",
    "TerraTorchPrithviBench",
    "TerraTorchTerraMindBench",
    "TorchGeoScaleMAEBench",
)
LARGE_MODELS = (
    "large",
    "600",
    "sam3",
    "vgg",
    "maxvit",
    "terramind",
    "olmoearth_base",
    "olmoearth_v1_1_base",
    "olmoearth_v1_2_base",
    "olmoearth_v1_3_base",
)
DEFERRED_DATASETS = {
    ("olmoearth_nano_t6", "aid"),
    ("universat_base", "m-brick-kiln"),
}


def pooling_kwargs(preset: ModelPreset) -> dict[str, str]:
    """Select mean patch pooling through existing wrapper interfaces."""
    if preset.target.endswith("TerraFMBench"):
        return {
            "pool": "mean",
            "checkpoint_path": "checkpoints/TerraFM-B.pth",
            "checkpoint_md5": "4f38e4ad30f908c0aa8534d473e99f3f",
        }
    if preset.target.endswith(MEAN_POOL_TARGETS):
        return {"pool": "mean"}
    if preset.target.endswith("AnySatBenchModel"):
        return {"pooling": "mean"}
    if preset.target.endswith("OlmoEarthBenchModel"):
        return {"embedding": "patch_tokens"}
    return {}


def input_variants(model: str) -> list[dict[str, str]]:
    """Return the study's band and normalization comparisons."""
    if model.startswith("handcrafted_level"):
        return [{}]
    norms = ["dataset", "model"]
    if model.startswith("olmoearth"):
        norms = ["model"]
    elif model.startswith(("anysat", "universat")) or model == "imagestats":
        norms = ["dataset"]
    return [{"bands": band, "normalization": norm} for band in ["rgb", "all"] for norm in norms]


def hardware_tier(model: str) -> str:
    """Route memory-intensive wrappers separately from the regular GPU queue."""
    if model.startswith(("anysat", "universat")):
        return "heavy"
    if any(name in model for name in LARGE_MODELS):
        return "medium"
    return "regular"


def make_job(model: str, dataset: str, preset: ModelPreset, inputs: dict[str, str]) -> dict:
    """Build one immutable scientific configuration and its result identity."""
    tier = hardware_tier(model)
    raw = {
        "model": {"name": model, "kwargs": pooling_kwargs(preset)},
        "datasets": [dataset],
        "input": inputs,
        "classification": {"methods": ["knn", "linear"]},
        "runtime": {
            "batch_size": 4 if tier == "heavy" else 64,
            "workers": 4,
            "sample_cache": "disk",
            "sample_cache_dir": "/opt/geh-cache",
            "seed": 0,
        },
    }
    config, resolved = resolve_run_config(RunConfig.model_validate(raw), dataset)
    config_hash = resume_config_hash(config, resolved, task="classification")
    identity = hashlib.sha256(f"{model}|{dataset}|{config_hash}".encode()).hexdigest()[:20]
    return {
        "id": identity,
        "tier": tier,
        "model": model,
        "dataset": dataset,
        "config_hash": config_hash,
        "config": raw,
    }


def model_jobs(model: str, preset: ModelPreset, datasets: list[str]) -> tuple[list, list]:
    """Respect declared dataset restrictions without loading any model weights."""
    jobs, exclusions = [], []
    for dataset in datasets:
        if preset.supported_datasets is not None and dataset not in preset.supported_datasets:
            exclusions.append(
                {"model": model, "dataset": dataset, "reason": "outside preset supported_datasets"}
            )
            continue
        effective = preset.for_dataset(dataset)
        if effective.kwargs.get("refuse"):
            exclusions.append(
                {"model": model, "dataset": dataset, "reason": effective.kwargs["refuse"]}
            )
            continue
        for inputs in input_variants(model):
            if (
                inputs.get("normalization") == "model"
                and effective.supports_model_normalization is False
            ):
                continue
            jobs.append(make_job(model, dataset, preset, inputs))
    return jobs, exclusions


def build_manifest() -> dict[str, Any]:
    """Discover all image presets and classification datasets in this checkout."""
    datasets = [
        name
        for name in list_datasets()
        if get_dataset_task(name) == "classification" and name != "infrabench-cls"
    ]
    jobs, exclusions = [], []
    for model in list_model_configs():
        preset = load_model_preset(ModelConfig(name=model))
        if model == "eosfm" or model.startswith("tessera_v2/"):
            exclusions.append({"model": model, "reason": "excluded from this study by request"})
            continue
        if (
            preset.track != "image"
            or model.endswith(("_seg", "_cls", "_student128", "_patch_tokens"))
            or model in {"handcrafted", "universat_base_dense"}
        ):
            exclusions.append(
                {
                    "model": model,
                    "reason": "coordinate/segmentation preset, alternate pooling, or duplicate alias",
                }
            )
            continue
        selected, omitted = model_jobs(model, preset, datasets)
        jobs.extend(selected)
        exclusions.extend(omitted)
    deferred = [job for job in jobs if (job["model"], job["dataset"]) in DEFERRED_DATASETS]
    jobs = [job for job in jobs if (job["model"], job["dataset"]) not in DEFERRED_DATASETS]
    for job in deferred:
        job["reason"] = "Deferred by request: projected runtime exceeds 20 hours on this fleet."
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "source_commit": commit,
        "jobs": jobs,
        "deferred_jobs": deferred,
        "exclusions": exclusions,
    }


def main() -> None:
    """Write a reproducible manifest without running or downloading experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    counts = Counter(job["tier"] for job in manifest["jobs"])
    logging.basicConfig(level=logging.INFO)
    logger.info("Planned %d configurations: %s", len(manifest["jobs"]), dict(counts))


if __name__ == "__main__":
    main()
