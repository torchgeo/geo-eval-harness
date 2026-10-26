"""Read merged fleet campaigns into canonical run and epoch tables.

Only already-merged campaign directories are read (``jobs.tsv``, ``fleet/merged.csv``, ``results/``,
``curves/``); nothing is fetched from or written to the fleet store.
"""

import csv
import fnmatch
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Campaigns with fewer planned epochs are gates: valid for memory/throughput, not for accuracy.
MIN_ACCURACY_EPOCHS = 5
STRATEGY_ORDER = ("frozen", "lora-last6", "lora", "full")
DECODER_ORDER = ("neck", "neck-ln", "anyup", "native")
DEFAULT_CAMPAIGNS = (
    "seg-decoder-pilot-2026-10-25",
    "overfit-s1-2026-10-25",
    "ws3-check-2026-10-25",
    "ws3-2026-10-25",
    "olmoearth-sar-check-2026-10-26",
    "olmoearth-sar-2026-10-26",
)
DEFAULT_CAMPAIGN_GLOBS = ("seg-fullft-*",)


@dataclass(frozen=True)
class Exclusion:
    """One campaign row left out of the analysis, with the reason."""

    campaign: str
    job_id: str
    reason: str


def decoder_of(job: dict) -> str:
    """Normalise a job's decoder arm: ``neck`` (plain), ``neck-ln``, ``anyup`` or ``native``."""
    arm = job.get("arm", "")
    if job.get("upsampler", "none") == "anyup" or arm == "anyup":
        return "anyup"
    if job.get("neck") == "vit" or arm in {"neck", "necknorm"}:
        return "neck-ln" if job.get("neck_norm") == "layernorm" or arm == "necknorm" else "neck"
    return "native"


def strategy_of(job: dict) -> str:
    """Return ``frozen``, ``lora-last<k>``, ``lora`` or ``full`` for a job row."""
    mode = job.get("mode", "")
    last = job.get("lora_last_blocks", "-")
    if mode == "lora" and last not in {"", "-", None}:
        return f"lora-last{last}"
    return mode


def family_of(model_key: str) -> str:
    """Architecture family: ``conv`` (ConvNeXt), ``swin`` or ``vit`` (plain ViT-style)."""
    if model_key.startswith("convnext"):
        return "conv"
    if model_key.startswith("swin"):
        return "swin"
    return "vit"


def read_tsv(path: Path) -> list[dict]:
    """Read a tab-separated table; a missing file gives an empty list."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> list[dict]:
    """Read a comma-separated table; a missing file gives an empty list."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_superseded(path: Path) -> list[tuple[str, str, str]]:
    """Read ``(campaign, job pattern, reason)`` rows from ``SUPERSEDED.tsv``."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "superseded"))
    return rows


def superseded_reason(campaign: str, job_id: str, rules: list[tuple[str, str, str]]) -> str | None:
    """Return the superseding reason for a job, if any rule matches it."""
    for rule_campaign, pattern, reason in rules:
        if rule_campaign == campaign and fnmatch.fnmatch(job_id, pattern):
            return f"superseded: {reason}"
    return None


def read_result(path: Path) -> dict | None:
    """Return the mIoU row of a result CSV, or None."""
    return next((r for r in read_csv(path) if r.get("metric_name") == "mIoU"), None)


def read_curve(path: Path) -> list[dict]:
    """Return the epoch events of the curve's selected (or else last) fit."""
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [e["run_id"] for e in events if e.get("event") == "selected"]
    epochs = [e for e in events if e.get("event") == "epoch"]
    if not epochs:
        return []
    run_id = selected[-1] if selected else epochs[-1]["run_id"]
    return [e for e in epochs if e["run_id"] == run_id]


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # allow-except: empty or non-numeric CSV fields mean missing
        return None


def curve_summary(epochs: list[dict]) -> dict:
    """Steady-state timing and memory from per-epoch records (epoch 1 may build caches)."""
    if not epochs:
        return {}
    seconds = [e["epoch_seconds"] for e in epochs]
    steady = seconds[1:] or seconds
    return {
        "n_epochs": len(epochs),
        "epoch1_s": seconds[0],
        "epoch_s_median": statistics.median(steady),
        "train_s_total": sum(seconds),
        "peak_gib": max(e.get("peak_gpu_mem_mb") or 0 for e in epochs) / 1024,
        "train_steps": epochs[0].get("train_steps"),
        "skipped_batches": epochs[0].get("skipped_batches"),
        "final_train_miou": epochs[-1].get("train_miou"),
        "final_val_miou": epochs[-1].get("val_miou"),
    }


def result_summary(row: dict | None) -> dict:
    """Accuracy fields from a result row, as fractions in [0, 1]."""
    if row is None:
        return {}
    return {
        "test_miou": _float(row.get("metric_value")),
        "ci_lower": _float(row.get("ci_lower")),
        "ci_upper": _float(row.get("ci_upper")),
        "boundary_iou": _float(row.get("boundary_iou")),
        "best_val_epoch": _float(row.get("best_val_epoch")),
        "best_val_miou": _float(row.get("best_val_miou")),
        "n_train": _float(row.get("n_train")),
        "n_val": _float(row.get("n_val")),
        "n_test": _float(row.get("n_test")),
        "method": row.get("method"),
    }


def _job_fields(campaign: str, job: dict) -> dict:
    planned = int(job.get("epochs") or 0)
    return {
        "campaign": campaign,
        "job_id": job["job_id"],
        "role": "accuracy" if planned >= MIN_ACCURACY_EPOCHS else "resource",
        "dataset": job["dataset"],
        "model_key": job["model_key"],
        "preset": job.get("preset"),
        "bands": job.get("bands"),
        "family": family_of(job["model_key"]),
        "strategy": strategy_of(job),
        "decoder": decoder_of(job),
        "seed": int(job.get("seed", 0)),
        "epochs_planned": planned,
        "config": job.get("config") or "base",
        "backbone_lr": _float(job.get("backbone_lr")),
        "weight_decay": _float(job.get("weight_decay")) or 0.0,
        "head_dropout": _float(job.get("head_dropout")) or 0.0,
    }


def _merge_fields(merged: dict) -> dict:
    started = pd.Timestamp(merged["started"]) if merged.get("started") else None
    finished = pd.Timestamp(merged["finished"]) if merged.get("finished") else None
    wall = None
    if started is not None and finished is not None:
        wall = (finished - started).total_seconds()
    return {
        "hub": merged.get("hub"),
        "gpu": merged.get("gpu"),
        "attempt": merged.get("attempt"),
        "bundle_sha": (merged.get("bundle_sha") or "")[:12],
        "started": started,
        "finished": finished,
        "wall_s": wall,
    }


def merge_exclusion(merged: dict | None) -> str | None:
    """Reason a merged outcome is unusable, or None for a verified, unstopped success."""
    if merged is None:
        return "not finished (no merged outcome)"
    if merged.get("status") != "done":
        return f"status {merged.get('status')}"
    if merged.get("verified") != "True":
        return "unverified outcome"
    if merged.get("stopped") == "True":
        return "stopped"
    return None


def load_campaign(
    root: Path, campaign: str, superseded: list[tuple[str, str, str]]
) -> tuple[list[dict], dict[tuple[str, str], list[dict]], list[Exclusion]]:
    """Return (runs, epoch records keyed by (campaign, job), exclusions) for one campaign."""
    cdir = root / campaign
    merged = {r["job_id"]: r for r in read_csv(cdir / "fleet" / "merged.csv")}
    runs, epochs, excluded = [], {}, []
    for job in read_tsv(cdir / "jobs.tsv"):
        job_id = job["job_id"]
        reason = superseded_reason(campaign, job_id, superseded) or merge_exclusion(
            merged.get(job_id)
        )
        result = read_result(cdir / "results" / f"{job_id}.csv")
        curve = read_curve(cdir / "curves" / f"{job_id}.jsonl")
        if reason is None and (result is None or not curve):
            reason = "missing result or curve"
        if reason is not None:
            excluded.append(Exclusion(campaign, job_id, reason))
            continue
        runs.append(
            _job_fields(campaign, job)
            | _merge_fields(merged[job_id])
            | curve_summary(curve)
            | result_summary(result)
        )
        epochs[(campaign, job_id)] = curve
    return runs, epochs, excluded


def discover_campaigns(root: Path, names: tuple[str, ...], globs: tuple[str, ...]) -> list[str]:
    """Return the named campaigns plus glob matches that have a jobs.tsv, in a stable order."""
    found = [n for n in names if (root / n / "jobs.tsv").exists()]
    for pattern in globs:
        found += sorted(
            p.name for p in root.glob(pattern) if (p / "jobs.tsv").exists() and p.name not in found
        )
    return found


def add_concurrency(runs: pd.DataFrame) -> pd.DataFrame:
    """Count jobs running on the same hub at each job's midpoint (co-tenancy, incl. itself)."""
    runs = runs.copy()
    runs["hub_jobs_at_mid"] = pd.NA
    timed = runs.dropna(subset=["hub", "started", "finished"])
    for _, group in timed.groupby("hub"):
        starts, ends = group["started"], group["finished"]
        for idx, row in group.iterrows():
            mid = row["started"] + (row["finished"] - row["started"]) / 2
            runs.at[idx, "hub_jobs_at_mid"] = int(((starts <= mid) & (ends >= mid)).sum())
    return runs


def _epoch_rows(campaign: str, job_id: str, records: list[dict]) -> list[dict]:
    keys = (
        "epoch_seconds",
        "train_loss",
        "val_loss",
        "train_miou",
        "val_miou",
        "train_steps",
        "skipped_batches",
    )
    return [
        {
            "campaign": campaign,
            "job_id": job_id,
            "epoch": e["epoch"],
            "peak_gib": (e.get("peak_gpu_mem_mb") or 0) / 1024,
        }
        | {k: e.get(k) for k in keys}
        for e in records
    ]


def load_runs(root: Path, campaigns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (runs, epochs, exclusions) tables for the given campaigns under ``root``."""
    superseded = read_superseded(root / "SUPERSEDED.tsv")
    all_runs, all_epochs, all_excluded = [], [], []
    for campaign in campaigns:
        runs, epochs, excluded = load_campaign(root, campaign, superseded)
        all_runs += runs
        all_excluded += excluded
        for (camp, job_id), records in epochs.items():
            all_epochs += _epoch_rows(camp, job_id, records)
    runs_df = add_concurrency(pd.DataFrame(all_runs)) if all_runs else pd.DataFrame()
    excluded_df = pd.DataFrame(
        [e.__dict__ for e in all_excluded], columns=["campaign", "job_id", "reason"]
    )
    return runs_df, pd.DataFrame(all_epochs), excluded_df


def reuse_map(root: Path, campaigns: list[str]) -> pd.DataFrame:
    """Return reuse.tsv rows: cells a campaign takes from an identical earlier job."""
    rows = [{"campaign": c} | r for c in campaigns for r in read_tsv(root / c / "reuse.tsv")]
    return pd.DataFrame(rows, columns=["campaign", "job_id", "source_campaign", "source_job_id"])
