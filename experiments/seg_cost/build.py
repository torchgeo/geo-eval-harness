"""Assemble all canonical tables for one report snapshot and write them to disk."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from experiments.seg_cost import analysis, ingest, resources

FOOTPRINT_CAMPAIGN = "seg-footprint-2026-10-25"
MATRIX_CAMPAIGN = "ws3-check-2026-10-25"  # the 268-cell WS3 definition, one epoch per cell
SEGMENTATION_DATASETS = (
    "burn_scars",
    "caffe",
    "cloudsen12",
    "dynamic_earthnet",
    "flair2",
    "fotw",
    "hotosm_buildings",
    "kuro_siwo",
    "pastis",
    "spacenet2",
    "spacenet7",
)
# Native band selections for datasets the loader profile did not cover (single-date S2 PASTIS).
NATIVE_BANDS_FALLBACK = {"pastis": "b02,b03,b04,b05,b06,b07,b08,b8a,b11,b12"}


@dataclass
class Bundle:
    """Every table of one snapshot, plus provenance."""

    runs: pd.DataFrame
    epochs: pd.DataFrame
    exclusions: pd.DataFrame
    reuse: pd.DataFrame
    accuracy: pd.DataFrame
    strategy_pairs: pd.DataFrame
    decoder_pairs: pd.DataFrame
    seed_noise: pd.DataFrame
    pareto: pd.DataFrame
    footprint: pd.DataFrame
    memory: pd.DataFrame
    data_paths: pd.DataFrame
    speedups: pd.DataFrame
    caches: pd.DataFrame
    peaks: pd.DataFrame
    fit: pd.DataFrame
    scenarios: pd.DataFrame
    full_ratio: float
    meta: dict = field(default_factory=dict)


def _git_commit(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def matrix_cells(root: Path, acc: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """WS3 cells (dataset x model x decoder) with a frozen epoch time for budget projections.

    The time is the 20-epoch steady-state median where a matching run exists, else the WS3
    one-epoch gate's frozen epoch (which may include a first-epoch cache build: an upper bound).
    """
    jobs = pd.DataFrame(ingest.read_tsv(root / MATRIX_CAMPAIGN / "jobs.tsv"))
    jobs["decoder"] = [ingest.decoder_of(r) for r in jobs.to_dict("records")]
    jobs["family"] = jobs["model_key"].map(ingest.family_of)
    cells = jobs.drop_duplicates(["dataset", "model_key", "decoder"])[
        ["dataset", "model_key", "bands", "decoder", "family"]
    ]
    keys = ["dataset", "model_key", "decoder"]
    frozen = acc[(acc["strategy"] == "frozen") & (acc["seed"] == 0)][[*keys, "epoch_s_median"]]
    gate = runs[(runs["campaign"] == MATRIX_CAMPAIGN) & (runs["strategy"] == "frozen")][
        [*keys, "epoch1_s"]
    ]
    cells = cells.merge(frozen.drop_duplicates(keys), on=keys, how="left").merge(
        gate.drop_duplicates(keys), on=keys, how="left"
    )
    cells["frozen_epoch_s"] = cells["epoch_s_median"].fillna(cells["epoch1_s"])
    cells["frozen_time_source"] = (
        cells["epoch_s_median"]
        .notna()
        .map({True: "20-epoch median", False: "1-epoch gate (upper bound)"})
    )
    return cells.dropna(subset=["frozen_epoch_s"])


def build(root: Path, repo: Path, campaigns: list[str] | None = None) -> Bundle:
    """Read every source under ``root`` and compute all tables."""
    names = campaigns or ingest.discover_campaigns(
        root, ingest.DEFAULT_CAMPAIGNS, ingest.DEFAULT_CAMPAIGN_GLOBS
    )
    runs, epochs, exclusions = ingest.load_runs(root, names)
    acc = analysis.accuracy_runs(runs)
    fp_dir = root / FOOTPRINT_CAMPAIGN / "results"
    footprint = resources.read_footprint(fp_dir)
    memory = resources.decompose_memory(footprint)
    paths = resources.data_path_table(footprint)

    splits = resources.split_census(root)
    splits = splits[splits["dataset"].isin(SEGMENTATION_DATASETS)]
    native = resources.native_channels(fp_dir)
    ladder_bands = pd.concat(
        [
            runs[["dataset", "bands"]],
            pd.DataFrame({"dataset": splits["dataset"], "bands": "rgb"}),
            pd.DataFrame(
                {"dataset": [d for d in splits["dataset"] if d in native], "bands": "all"}
            ),
            pd.DataFrame([{"dataset": d, "bands": b} for d, b in NATIVE_BANDS_FALLBACK.items()]),
        ]
    )
    caches = resources.dataset_cache_table(
        splits, ladder_bands, resources.measured_caches(fp_dir), native
    )
    ratios = resources.full_over_lora_ratio(memory)
    full_ratio = float(ratios["full_over_lora"].median())
    peaks = resources.project_full(resources.training_peaks(runs, memory), full_ratio)
    fit = resources.fit_matrix(peaks, caches)
    cells = matrix_cells(root, acc, runs)
    scenarios = analysis.budget_scenarios(cells, acc, analysis.footprint_epoch_ratios(paths))

    meta = {
        "snapshot_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "outputs_root": str(root),
        "analysis_commit": _git_commit(repo),
        "campaigns": {
            c: int((runs["campaign"] == c).sum()) if not runs.empty else 0 for c in names
        },
        "excluded": len(exclusions),
        "footprint_records": len(footprint),
        "full_over_lora_peak_ratio_median": full_ratio,
        "matrix_cells": len(cells),
    }
    return Bundle(
        runs=runs,
        epochs=epochs,
        exclusions=exclusions,
        reuse=ingest.reuse_map(root, names),
        accuracy=acc,
        strategy_pairs=analysis.strategy_pairs(acc),
        decoder_pairs=analysis.decoder_pairs(acc),
        seed_noise=analysis.seed_noise(acc),
        pareto=analysis.pareto_table(acc),
        footprint=footprint,
        memory=memory,
        data_paths=paths,
        speedups=resources.residency_speedups(paths),
        caches=caches,
        peaks=peaks,
        fit=fit,
        scenarios=scenarios,
        full_ratio=full_ratio,
        meta=meta,
    )


TABLES = (
    "runs",
    "epochs",
    "exclusions",
    "reuse",
    "accuracy",
    "strategy_pairs",
    "decoder_pairs",
    "seed_noise",
    "pareto",
    "footprint",
    "memory",
    "data_paths",
    "speedups",
    "caches",
    "peaks",
    "fit",
    "scenarios",
)


def write_tables(bundle: Bundle, out: Path) -> dict[str, str]:
    """Write every table as CSV under ``out/tables``; return their SHA-256 digests."""
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in TABLES:
        path = tdir / f"{name}.csv"
        getattr(bundle, name).to_csv(path, index=False)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    bundle.meta["table_sha256"] = digests
    (out / "manifest.json").write_text(json.dumps(bundle.meta, indent=2, default=str) + "\n")
    return digests
