"""Save dated result snapshots and refresh the Results Explorer.

Read per-model results and ``results/compute_cost.csv`` into a dated snapshot.

Write ``docs/_static/results-explorer.html`` and retain dated snapshots for comparison.

Include classification, segmentation, profiling, and intrinsic-dimension measurements.

Usage::

    python scripts/regen_results_explorer.py [--label 2026-05-08]
"""

import argparse
import csv
import json
import logging
import re
from datetime import date
from pathlib import Path

from geo_eval_harness.results import load_results

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "models"
PROFILE_RESULTS_DIR = ROOT / "results" / "profiles"
INTRINSIC_DIM_RESULTS_DIR = ROOT / "results" / "intrinsic_dim"
# Cost/throughput moved out of results/profiles/ into one wide table; only
# imagestats was ever migrated, which is why the efficiency figure emptied out.
COMPUTE_COST_CSV = ROOT / "results" / "compute_cost.csv"
HTML_PATH = ROOT / "docs" / "_static" / "results-explorer.html"
SNAPSHOT_DIR = ROOT / "docs" / "_static" / "_results_snapshots"
SEG_METHODS = ("seg-linear", "seg-conv_block", "seg-fpn", "seg-dpt")
ACCURACY_METHODS = ("knn5", "linear")
ALLOWED_METHODS = (*ACCURACY_METHODS, *SEG_METHODS, "profile", "intrinsic_dim")

# compute_cost.csv is wide -- one row per measured config, one column per
# quantity.  The explorer wants the long ``method="profile"`` rows that
# results/profiles/*.csv used to supply, so melt these columns into rows.
COMPUTE_COST_METRICS = (
    "gflops_backbone",
    "gflops_head",
    "gflops_probe",
    "gflops_total",
    "params_backbone_m",
    "params_head_m",
    "params_probe_m",
    "throughput_samples_per_sec",
    "latency_ms_per_batch_p50",
    "peak_gpu_mem_gb",
    "reserved_gpu_mem_gb",
    "n_tokens",
)
# The results CSVs call the multispectral run "all"; compute_cost measures it as
# the 12-band S2 stack.  Pair them so the efficiency join lands -- n_channels
# rides along on the row so a dataset whose "all" is not 12 bands stays visible.
BAND_CONFIG_ALIASES = {"s2": "all"}

# Cloud pricing and grid carbon intensity: page-owned reference data that lives
# inside the replaced <script> span and must survive regeneration.
REFERENCE_DATA_RE = r"const GPU_PRICES = \[.*?\];\s*const CARBON_INTENSITY = \[.*?\];"

COLUMNS = [
    "dataset",
    "method",
    "metric_name",
    "metric_value",
    "ci_lower",
    "ci_upper",
    "feature_dim",
    "best_c",
    "best_lr",
    "best_batch_size",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "model",
    "name",
    "normalization",
    "image_size",
    "interpolation",
    "partition",
    "bands",
    "c_range_start",
    "c_range_stop",
    "c_range_num",
    "merge_val",
    "bootstrap",
    "fw_iou",
    "precision",
    "recall",
    "f1",
    "task",
    "head_type",
    "n_channels",
    "snapshot",
]
NUMERIC = {
    "metric_value",
    "ci_lower",
    "ci_upper",
    "feature_dim",
    "best_c",
    "best_lr",
    "best_batch_size",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "image_size",
    "c_range_start",
    "c_range_stop",
    "c_range_num",
    "bootstrap",
    "fw_iou",
    "precision",
    "recall",
    "f1",
    "n_channels",
}
BOOL = {"merge_val"}


def _iter_compute_cost_rows():
    """Melt ``results/compute_cost.csv`` into long ``method="profile"`` rows.

    Cost is measured per (model, band config, task, head) at a fixed 224px
    input -- it carries no ``dataset``, because a frozen backbone costs the
    same whichever dataset is probed.  The explorer joins these to accuracy on
    (name, bands) for that reason.
    """
    if not COMPUTE_COST_CSV.exists():
        return
    with COMPUTE_COST_CSV.open() as fh:
        for r in csv.DictReader(fh):
            bands = BAND_CONFIG_ALIASES.get(r["band_config"], r["band_config"])
            for metric in COMPUTE_COST_METRICS:
                if not r.get(metric):
                    continue
                yield {
                    "dataset": "",
                    "method": "profile",
                    "metric_name": metric,
                    "metric_value": r[metric],
                    "model": r["model"],
                    "name": r["name"],
                    "bands": bands,
                    "n_channels": r["n_channels"],
                    "image_size": r["image_size"],
                    "feature_dim": r["feature_dim"],
                    "task": r["task"],
                    "head_type": r["head_type"],
                }


def _iter_result_rows():
    """Yield rows from every per-model results CSV, plus melted compute costs."""
    for directory in (RESULTS_DIR, PROFILE_RESULTS_DIR, INTRINSIC_DIM_RESULTS_DIR):
        frame = load_results(directory)
        yield from frame.astype(object).where(frame.notna(), None).to_dict("records")
    yield from _iter_compute_cost_rows()


def _load_csv_rows(label: str) -> list[dict]:
    rows = []
    for r in _iter_result_rows():
        if r["method"] not in ALLOWED_METHODS:
            continue
        if r.get("metric_value") is None or r["metric_value"] == "":
            continue
        row = {}
        for k in COLUMNS:
            if k == "snapshot":
                continue
            v = r.get(k, "")
            if v is None or v == "":
                row[k] = None
            elif k in NUMERIC:
                row[k] = float(v)
            elif k in BOOL:
                row[k] = str(v).lower() in ("true", "1")
            else:
                row[k] = v
        row["snapshot"] = label
        rows.append(row)
    return rows


def _snapshot_label_sort_key(label: str) -> tuple:
    """Sort dated labels chronologically and other labels alphabetically."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", label)
    return (m.group(1) if m else "", label)


def _sub_once(pattern: str, repl: str, text: str) -> str:
    """Substitute exactly once, failing loudly if the anchor has drifted.

    A silent no-match here leaves stale copy on a page whose data has been
    refreshed -- the failure mode that left the masthead quoting May's numbers.
    """
    out, n = re.subn(pattern, lambda _: repl, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise SystemExit(f"Could not locate {pattern!r} in {HTML_PATH.name}")
    return out


def _replace_snapshot_data(
    text: str, rows: list[dict], snapshot_meta: list[dict], latest_label: str
) -> str:
    """Replace generated rows while preserving page-owned cost reference data."""
    js_columns = "const COLUMNS = " + json.dumps(COLUMNS) + ";"
    js_numeric = "const NUMERIC_COLS = " + json.dumps(sorted(NUMERIC)) + ";"
    js_snaps = "const SNAPSHOTS = " + json.dumps(snapshot_meta) + ";"
    js_default = "const DEFAULT_SNAPSHOT = " + json.dumps(latest_label) + ";"
    js_data = "const DATA = " + json.dumps(rows, separators=(",", ":")) + ";"

    pattern = re.compile(
        r"const COLUMNS = \[.*?\];\s*const NUMERIC_COLS = \[.*?\];"
        r"(?:\s*const SNAPSHOTS = \[.*?\];)?(?:\s*const DEFAULT_SNAPSHOT = \"[^\"]*\";)?"
        r"\s*const DATA = \[.*?\];"
        r"(?:\s*const GPU_PRICES = \[.*?\];)?(?:\s*const CARBON_INTENSITY = \[.*?\];)?",
        re.DOTALL,
    )
    carried = re.search(REFERENCE_DATA_RE, text, re.DOTALL)
    if not carried:
        raise SystemExit(
            "Could not locate GPU_PRICES/CARBON_INTENSITY in "
            f"{HTML_PATH.name}; the efficiency figure needs both."
        )
    new_block = "\n".join([js_columns, js_numeric, js_snaps, js_default, js_data, carried.group()])
    if not pattern.search(text):
        raise SystemExit("Could not locate COLUMNS/NUMERIC_COLS/DATA block in HTML.")
    return pattern.sub(lambda _: new_block, text, count=1)


def main() -> None:
    """Regenerate the Results Explorer data snapshots from the tracked result CSVs."""
    today = date.today()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=today.isoformat(),
        help="Label for the snapshot generated from the current CSV (default: today).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    today_rows = _load_csv_rows(args.label)
    snapshot_path = SNAPSHOT_DIR / f"{args.label}.json"
    snapshot_path.write_text(json.dumps(today_rows, separators=(",", ":")))

    snapshots: dict[str, list[dict]] = {}
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        label = path.stem
        rows = json.loads(path.read_text())
        for r in rows:
            # The filename defines the snapshot, even if rows contain an old label.
            r["snapshot"] = label
        snapshots[label] = rows

    ordered_labels = sorted(snapshots, key=_snapshot_label_sort_key, reverse=True)
    latest_label = ordered_labels[0]
    latest_rows = snapshots[latest_label]
    flat_rows = [r for label in ordered_labels for r in snapshots[label]]
    snapshot_meta = [{"label": label, "rows": len(snapshots[label])} for label in ordered_labels]

    accuracy_rows = [r for r in latest_rows if r["method"] in ACCURACY_METHODS]
    seg_rows = [r for r in latest_rows if r["method"] in SEG_METHODS]
    n_models = len({r["name"] for r in accuracy_rows if r["name"]}) or len(
        {r["name"] for r in latest_rows if r["name"]}
    )
    # Profile rows carry no dataset, so count datasets per task rather than
    # over every row.
    n_datasets = len({r["dataset"] for r in accuracy_rows if r["dataset"]})
    n_seg_datasets = len({r["dataset"] for r in seg_rows if r["dataset"]})
    n_seg_models = len({r["name"] for r in seg_rows if r["name"]})
    best = max(accuracy_rows or latest_rows, key=lambda r: r["metric_value"] or 0)

    text = _replace_snapshot_data(
        HTML_PATH.read_text(encoding="utf-8"), flat_rows, snapshot_meta, latest_label
    )

    seg_sentence = (
        f" A further {len(seg_rows):,} segmentation measurements cover "
        f"{n_seg_models} backbones on {n_seg_datasets} datasets (mIoU, four "
        f"decoder heads)."
        if seg_rows
        else ""
    )

    text = _sub_once(
        r'<h1 class="headline" id="headline-text">.*?</h1>',
        '<h1 class="headline" id="headline-text">Geo Eval Harness results explorer</h1>',
        text,
    )
    text = _sub_once(
        r'<p class="standfirst" id="standfirst-text">.*?</p>',
        (
            f'<p class="standfirst" id="standfirst-text">'
            f"This snapshot contains {len(latest_rows):,} measurements, including "
            f"{n_models} frozen-backbone variants on {n_datasets} classification datasets."
            f"{seg_sentence}"
            f" Use the controls to filter; the figures and table follow the selection."
            f"</p>"
        ),
        text,
    )
    text = _sub_once(
        r'<b id="row-shown">\d+</b> of <b id="row-total">\d+</b>',
        f'<b id="row-shown">{len(latest_rows)}</b> of <b id="row-total">{len(latest_rows)}</b>',
        text,
    )
    text = _sub_once(
        r"Source: <b>[^<]*</b>",
        "Source: <b>results/{models,profiles,intrinsic_dim}/*.csv, results/compute_cost.csv</b>",
        text,
    )
    text = _sub_once(
        r"Published <b>[^<]*</b>",
        f"Published <b>{today.day} {today:%B %Y}</b>",
        text,
    )
    text = _sub_once(
        r"documented in <code>[^<]*</code>\.\s+Confidence intervals\s+are 95%\s+bootstrap"
        r"\s+on test predictions\s+\(default \d+ resamples\)\.",
        (
            "documented in <code>docs/user/methodology.rst</code>. "
            "Confidence intervals are 95% bootstrap on test predictions "
            "(default 200 resamples)."
        ),
        text,
    )

    HTML_PATH.write_text(text, encoding="utf-8")
    logger.info(
        "Wrote %s: %d snapshots, latest=%s (%d rows, %d models, %d datasets) — best %.4f (%s on %s)",
        HTML_PATH.relative_to(ROOT),
        len(snapshot_meta),
        latest_label,
        len(latest_rows),
        n_models,
        n_datasets,
        best["metric_value"],
        best["name"],
        best["dataset"],
    )


if __name__ == "__main__":
    main()
