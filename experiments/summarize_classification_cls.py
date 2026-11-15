"""Compare audited CLS results with their exact mean-sweep counterparts."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.results import model_results_path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a coverage ledger or result file."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def result_rows(job: dict, effective_hash: str, results: Path) -> dict[str, dict[str, str]]:
    """Select the audited run without mixing hashes, pooling, datasets, or probes."""
    _, preset = resolve_run_config(RunConfig.model_validate(job["config"]), job["dataset"])
    path = model_results_path(results, preset.name)
    selected = [
        row
        for row in read_csv(path)
        if row["config_hash"] == effective_hash and row["dataset"] == job["dataset"]
    ]
    if len({row["method"] for row in selected}) != len(selected):
        raise ValueError(f"Duplicate comparison rows: {path}/{job['id']}")
    return {row["method"]: row for row in selected}


def compare(
    manifests: tuple[dict, dict], ledgers: tuple[list[dict], list[dict]], results: Path
) -> list[dict]:
    """Average only paired configurations, separately for each model and method."""
    refresh, companion = manifests
    mean_jobs = {job["id"]: job for job in refresh["jobs"]}
    mean_coverage, cls_coverage = ({row["job_id"]: row for row in rows} for rows in ledgers)
    pairs: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    for job in companion["jobs"]:
        cls_state = cls_coverage[job["id"]]
        if cls_state["state"] not in {"complete", "partial"}:
            continue
        parent = mean_jobs[job["mean_job_id"]]
        mean_state = mean_coverage[parent["id"]]
        mean_rows = result_rows(parent, mean_state["effective_config_hash"], results)
        cls_rows = result_rows(job, cls_state["effective_config_hash"], results)
        for method in mean_rows.keys() & cls_rows.keys():
            pairs.setdefault((job["model"], method), []).append(
                (mean_rows[method], cls_rows[method])
            )
    return [
        {
            "model": model,
            "method": method,
            "matched_configurations": len(rows),
            "mean_pool_score": mean(float(a["metric_value"]) for a, _ in rows),
            "cls_pool_score": mean(float(b["metric_value"]) for _, b in rows),
            "cls_ci_above_mean": sum(float(b["ci_lower"]) > float(a["ci_upper"]) for a, b in rows),
            "mean_ci_above_cls": sum(float(a["ci_lower"]) > float(b["ci_upper"]) for a, b in rows),
        }
        for (model, method), rows in sorted(pairs.items())
    ]


def main() -> None:
    """Write matched comparisons; pending and unpaired methods are not scored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-manifest", required=True, type=Path)
    parser.add_argument("--refresh-coverage", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--results", default=Path("results/models"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = compare(
        (json.loads(args.refresh_manifest.read_text()), json.loads(args.manifest.read_text())),
        (read_csv(args.refresh_coverage), read_csv(args.coverage)),
        args.results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
