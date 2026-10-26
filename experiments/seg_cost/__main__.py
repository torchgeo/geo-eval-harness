"""Build the segmentation cost-benefit report from merged campaign outputs.

    python -m experiments.seg_cost build --root ~/geo-eval-harness/outputs --out DIR
        [--campaign NAME ...] [--no-explorer]

Reads campaigns (jobs.tsv, fleet/merged.csv, results/, curves/) and the footprint benchmark; writes
DIR/tables/*.csv, DIR/manifest.json, DIR/figures/*.{png,svg,pdf}, DIR/explorer.html and
DIR/REPORT.md. It never contacts the fleet store; run ``fleet merge`` first to refresh.
"""

import argparse
import sys
from pathlib import Path

from experiments.seg_cost.build import build, write_tables
from experiments.seg_cost.explorer import write_explorer
from experiments.seg_cost.figures import write_figures
from experiments.seg_cost.report import write_report

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--root", type=Path, default=Path.home() / "geo-eval-harness" / "outputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        action="append",
        default=None,
        help="campaign directory name (repeatable); default: the known set",
    )
    parser.add_argument("--no-explorer", action="store_true")
    args = parser.parse_args(argv)
    bundle = build(args.root.expanduser(), REPO, args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    write_tables(bundle, args.out)
    write_figures(bundle, args.out / "figures")
    if not args.no_explorer:
        write_explorer(bundle, args.out)
    report = write_report(bundle, args.out)
    sys.stdout.write(f"{len(bundle.runs)} runs, {len(bundle.exclusions)} excluded -> {report}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
