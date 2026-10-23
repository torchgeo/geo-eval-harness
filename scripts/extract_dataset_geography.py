"""Write geographic metadata for every registered dataset.

Write dataset records and ``index.json`` to ``docs/_static/_dataset_geography/``.

Use the registry so newly added datasets appear in the store.

Usage::

    # regenerate the whole store
    python scripts/extract_dataset_geography.py --all

    # regenerate a single dataset without rescanning everything
    python scripts/extract_dataset_geography.py --dataset m-eurosat

    # report coverage without touching the store
    python scripts/extract_dataset_geography.py --check
"""

import argparse
import logging
import os
import warnings

from geo_eval_harness.datasets import list_datasets
from geo_eval_harness.geography import (
    STORE_DIR,
    build_index,
    extract_geography,
    list_geography,
    missing_datasets,
    write_record,
)

logger = logging.getLogger(__name__)


def _check() -> int:
    """Report the current store's coverage; non-zero if a dataset is missing."""
    store = list_geography()
    missing = missing_datasets()

    for name in sorted(store):
        record = store[name]
        detail = record.reason or f"n={record.n}"
        logger.info("  %-20s %-15s %s", name, record.status, detail)

    if missing:
        logger.warning("%d registered dataset(s) with no record: %s", len(missing), sorted(missing))
        logger.info("Run with --all (or --dataset <name>) to generate them.")
        return 1
    logger.info("All %d registered datasets have a record.", len(store))
    return 0


def main() -> int:
    """Regenerate the cached dataset geography metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="regenerate every registered dataset")
    group.add_argument("--dataset", help="regenerate a single dataset by name")
    group.add_argument("--check", action="store_true", help="report coverage, write nothing")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 8)),
        help="processes used for the V1 metadata scan",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")

    if args.check:
        return _check()

    names = list_datasets() if args.all else [args.dataset]
    if not args.all and args.dataset not in list_datasets():
        logger.error("Unknown dataset %r. Available: %s", args.dataset, ", ".join(list_datasets()))
        return 1

    for name in names:
        record = extract_geography(name, workers=args.workers)
        write_record(record)
        detail = record.reason or f"n={record.n}, {len(record.bins)} bins"
        logger.info("  %-20s %-15s %s", name, record.status, detail)

    index = build_index()
    totals = index["totals"]
    logger.info(
        "Wrote %s: %d records (%d extracted, %d samples)",
        STORE_DIR,
        totals["datasets"],
        totals["extracted"],
        totals["samples"],
    )
    for continent, share in list(totals["continents"].items())[:6]:
        logger.info("  %-20s %5.1f%%", continent, share)

    missing = missing_datasets()
    if missing:
        logger.warning("No record for %s", sorted(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
