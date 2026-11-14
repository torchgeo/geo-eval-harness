#!/usr/bin/env python3
"""Relabel Clay v1 Base results and generated snapshots."""

import csv
import hashlib
import io
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
IDENTIFIERS = {
    "tt_clay_v1_5_base": "tt_clay_v1_base",
    "tt_clay_v1_5_base_cls": "tt_clay_v1_base_cls",
    "terratorch/clay_v1_5": "terratorch/clay_v1",
    "terratorch/clay_v1_5_cls": "terratorch/clay_v1_cls",
}


def _csv(path: Path) -> int:
    original = path.read_bytes()
    reader = csv.reader(io.StringIO(original.decode()))
    rows = list(reader)
    if not rows or "name" not in rows[0]:
        return 0
    name_column = rows[0].index("name")
    changed = 0
    for row in rows[1:]:
        if row[name_column] not in IDENTIFIERS:
            continue
        updated = [IDENTIFIERS.get(cell, cell) for cell in row]
        changed += updated != row
        row[:] = updated
    if changed:
        stream = io.StringIO(newline="")
        newline = "\r\n" if b"\r\n" in original else "\n"
        csv.writer(stream, lineterminator=newline).writerows(rows)
        path.write_bytes(stream.getvalue().encode())
    return changed


def _snapshot(path: Path) -> bool:
    original = path.read_text()
    if "tt_clay_v1_5_base" not in original:
        return False
    if "tt_clay_v1_5_large" in original:
        raise ValueError(f"Regenerate mixed Clay v1/v1.5 snapshot from migrated CSVs: {path}")
    updated = original
    replacements = dict(IDENTIFIERS)
    for old, new in IDENTIFIERS.items():
        if old.startswith("tt_"):
            for directory in ("models", "profiles", "intrinsic_dim"):
                replacements[f"results/{directory}/{old}.csv"] = f"results/{directory}/{new}.csv"
        else:
            replacements[f"src/geo_eval_harness/conf/model/{old}.yaml"] = (
                f"src/geo_eval_harness/conf/model/{new}.yaml"
            )
    for old, new in replacements.items():
        updated = updated.replace(json.dumps(old), json.dumps(new))
    for old in ("Clay v1.5 base", "Clay v1.5 Base", "Clay-v1.5"):
        updated = updated.replace(old, "Clay v1 Base")
    if updated != original:
        path.write_text(updated)
        return True
    return False


def _provenance(root: Path, before: str, changed: int) -> None:
    path = root / "results/compute_cost_provenance.json"
    if not changed or not path.exists():
        return
    digest = hashlib.sha256((root / "results/compute_cost.csv").read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    data["combined_data_sha256"] = digest
    data["postprocessing"]["clay_v1_identifiers"] = {
        "input_combined_sha256": before,
        "output_combined_sha256": digest,
        "mapping": IDENTIFIERS,
        "changed_compute_rows": changed,
        "metrics_and_run_hashes_modified": False,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _renames(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for directory in ("models", "profiles", "intrinsic_dim"):
        for old, new in IDENTIFIERS.items():
            if not old.startswith("tt_"):
                continue
            source = root / "results" / directory / f"{old}.csv"
            destination = source.with_name(f"{new}.csv")
            if source.exists():
                if destination.exists():
                    raise FileExistsError(destination)
                pairs.append((source, destination))
    return pairs


def migrate(root: Path) -> dict[str, int]:
    """Relabel canonical CSVs and generated snapshots under a repository root."""
    renames = _renames(root)
    cost = root / "results/compute_cost.csv"
    before = hashlib.sha256(cost.read_bytes()).hexdigest() if cost.exists() else ""
    counts = {}
    paths = [*sorted((root / "results").rglob("*.csv"))]
    fixture = root / "tests/fixtures/accuracy_baselines.csv"
    if fixture.exists():
        paths.append(fixture)
    for path in paths:
        changed = _csv(path)
        if changed:
            counts[path.relative_to(root).as_posix()] = changed
    for source, destination in renames:
        source.rename(destination)
    snapshots = root / "docs/_static/_results_snapshots"
    assets = [*sorted(snapshots.glob("*.json"))]
    assets += [
        root / f"docs/_static/{name}" for name in ("ranking_explorer.html", "results-explorer.html")
    ]
    for path in assets:
        if path.exists() and _snapshot(path):
            counts[path.relative_to(root).as_posix()] = 1
    _provenance(root, before, counts.get("results/compute_cost.csv", 0))
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for filename, rows in migrate(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT).items():
        logger.info("%s: %d relabeled rows/objects", filename, rows)
