#!/usr/bin/env python3
"""Rewrite every ``results/models/<name>.csv`` with the current result header.

The header becomes the field order of :class:`geo_eval_harness.results.EvaluationResult`,
which adds the segmentation setup columns (``adaptation`` ... ``early_stopping_patience``)
after ``pool`` and the ``boundary_iou*`` / ``best_val_*`` columns where files lack them.
Columns the file does not have yet get empty cells; existing cells are copied verbatim.
Unknown extra columns are kept after the known ones.

Existing segmentation rows keep empty setup columns: their epochs and learning rates are not
recorded in the row, so nothing is backfilled. The leaderboard treats them as legacy rows.

The rewrite is idempotent and keeps every row in order.

Usage::

    python scripts/migrations/migrate_setup_columns.py [RESULTS_DIR]
"""

import csv
import dataclasses
import logging
import sys
from pathlib import Path

from geo_eval_harness.results import EvaluationResult

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "results" / "models"

HEADER = [field.name for field in dataclasses.fields(EvaluationResult)]


def migrate_one(path: Path) -> tuple[int, bool]:
    """Rewrite one CSV with :data:`HEADER`; return ``(rows, changed)``."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    extra = [name for name in fieldnames if name not in HEADER]
    if extra:
        logger.warning("%s: keeping unknown columns %s after the known ones", path.name, extra)
    header = HEADER + extra
    if fieldnames == header:
        return len(rows), False
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), True


def migrate(models_dir: Path) -> tuple[int, int, int]:
    """Migrate every CSV in ``models_dir``; return ``(files, changed, rows)``."""
    files = changed = total = 0
    for path in sorted(models_dir.glob("*.csv")):
        n_rows, was_changed = migrate_one(path)
        files += 1
        changed += was_changed
        total += n_rows
    return files, changed, total


def main(argv: list[str]) -> int:
    """Migrate ``argv[0]`` (default ``results/models``) and log a summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    models_dir = Path(argv[0]) if argv else MODELS_DIR
    files, changed, rows = migrate(models_dir)
    logger.info("%d CSVs (%d rewritten), %d rows in %s", files, changed, rows, models_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
