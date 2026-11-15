"""Plan CLS companions from the refresh's immutable mean-pooling manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.resume import resume_config_hash

SOURCE_COMMIT = "4c8693320629fec9df030bbb04062ebdde6a6f3c"
BATCHES = {
    1: {
        "terratorch/clay_v1": "terratorch/clay_v1_cls",
        "terratorch/clay_v1_5": "terratorch/clay_v1_5_cls",
        "torchgeo/scalemae_large_fmow": "torchgeo/scalemae_large_fmow_cls",
        "smarties/base": "smarties/base",
        "smarties/large": "smarties/large",
        "terrafm/base_s2": "terrafm/base_s2",
    },
    2: {
        "terratorch/prithvi_eo_v2_100_tl": "terratorch/prithvi_eo_v2_100_tl_cls",
        "terratorch/prithvi_eo_v2_300": "terratorch/prithvi_eo_v2_300_cls",
        "terratorch/prithvi_eo_v2_300_tl": "terratorch/prithvi_eo_v2_300_tl_cls",
        "terratorch/prithvi_eo_v2_600": "terratorch/prithvi_eo_v2_600_cls",
    },
}
LEDGER_FIELDS = (
    "job_id",
    "model",
    "dataset",
    "bands",
    "normalization",
    "planned_config_hash",
    "effective_config_hash",
    "batch_size",
    "state",
    "reason",
)


def companion_job(mean: dict[str, Any], model: str) -> dict[str, Any]:
    """Freeze all resolved mean settings before changing only pooling and preset name."""
    config, preset = resolve_run_config(RunConfig.model_validate(mean["config"]), mean["dataset"])
    if resume_config_hash(config, preset, task="classification") != mean["config_hash"]:
        raise ValueError(f"Refresh configuration drift: {mean['id']}")
    if preset.kwargs.get("pool") != "mean":
        raise ValueError(f"Expected mean pooling: {mean['id']}")
    raw = config.model_dump(mode="json")
    raw["model"] = {"name": model, "kwargs": preset.kwargs | {"pool": "cls"}}
    raw["output"] = {"resume": False}
    cls_config, cls_preset = resolve_run_config(RunConfig.model_validate(raw), mean["dataset"])
    expected = preset.model_dump(exclude={"name", "segmentation", "classification"})
    expected["kwargs"]["pool"] = "cls"
    if cls_preset.model_dump(exclude={"name", "segmentation", "classification"}) != expected:
        raise ValueError(f"Non-pooling preset difference: {model}/{mean['dataset']}")
    config_hash = resume_config_hash(cls_config, cls_preset, task="classification")
    identity = hashlib.sha256(f"{model}|{mean['dataset']}|{config_hash}".encode()).hexdigest()[:20]
    return {
        "id": identity,
        "tier": mean["tier"],
        "model": model,
        "dataset": mean["dataset"],
        "config_hash": config_hash,
        "config": raw,
        "mean_job_id": mean["id"],
    }


def build_companion(
    refresh: dict[str, Any], coverage: list[dict[str, str]], batch: int
) -> dict[str, Any]:
    """Include only inputs with complete or partial mean-run evidence."""
    if refresh["source_commit"] != SOURCE_COMMIT:
        raise ValueError("The companion requires the pinned refresh evaluation source.")
    states = {row["job_id"]: row for row in coverage}
    models = BATCHES[batch]
    jobs, excluded = [], []
    for mean in refresh["jobs"]:
        if mean["model"] not in models:
            continue
        job = companion_job(mean, models[mean["model"]])
        state = states[mean["id"]]
        if state["state"] in {"complete", "partial"}:
            jobs.append(job)
        elif state["state"] == "excluded":
            job["reason"] = "Mean counterpart excluded: " + state["reason"]
            excluded.append(job)
        else:
            raise ValueError(f"Mean counterpart is unresolved: {mean['id']}")
    if {job["model"] for job in jobs + excluded} != set(models.values()):
        raise ValueError("Refresh manifest is missing a requested model.")
    return {"source_commit": SOURCE_COMMIT, "batch": batch, "jobs": jobs, "excluded_jobs": excluded}


def write_coverage(manifest: dict[str, Any], path: Path) -> None:
    """Write the companion ledger in the refresh coverage schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for job in manifest["jobs"] + manifest["excluded_jobs"]:
            writer.writerow(
                {
                    "job_id": job["id"],
                    "model": job["model"],
                    "dataset": job["dataset"],
                    "bands": job["config"]["input"]["bands"],
                    "normalization": job["config"]["input"]["normalization"],
                    "planned_config_hash": job["config_hash"],
                    "effective_config_hash": "",
                    "batch_size": job["config"]["runtime"]["batch_size"],
                    "state": "excluded" if "reason" in job else "pending",
                    "reason": job.get("reason", ""),
                }
            )


def main() -> None:
    """Generate immutable jobs and initial coverage without loading weights or data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-manifest", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=int, choices=BATCHES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-coverage", required=True, type=Path)
    args = parser.parse_args()
    with args.coverage.open() as stream:
        coverage = list(csv.DictReader(stream))
    manifest = build_companion(json.loads(args.refresh_manifest.read_text()), coverage, args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    write_coverage(manifest, args.output_coverage)


if __name__ == "__main__":
    main()
