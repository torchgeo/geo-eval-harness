"""Inspect segmentation training curves across campaigns.

    python -m experiments.seg_runs report ROOT [ROOT ...] [--out DIR]
        Writes DIR/curves.html (interactive viewer) and DIR/findings.tsv (rule hits with a
        proposed action). DIR defaults to the first ROOT.
    python -m experiments.seg_runs live ROOT [ROOT ...]
        Prints running fits (unfinished curves touched in the last 30 min) with early flags.

A ROOT is any directory; every ``curves/*.jsonl`` below it is read (``source/`` snapshots skipped).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from experiments.seg_runs.curves import discover, live_table, write_findings
from experiments.seg_runs.viewer import write_html


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["report", "live"])
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = discover(args.roots)
    if args.command == "live":
        sys.stdout.write(f"{live_table(runs)}\n")
        return 0
    out = args.out or args.roots[0]
    n = write_findings(runs, out / "findings.tsv")
    write_html(runs, out / "curves.html", args.roots)
    status = Counter(r.status for r in runs)
    rules = Counter(f.rule for r in runs for f in r.findings)
    sys.stdout.write(
        f"{len(runs)} runs ({dict(status)}), {n} findings -> {out / 'findings.tsv'}, "
        f"viewer -> {out / 'curves.html'}\n"
    )
    for rule, count in rules.most_common():
        sys.stdout.write(f"  {rule:28} {count}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
