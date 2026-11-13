"""Reproduce the fixed seed-0 10k manifest from the committed Open Cities chip index."""

import gzip
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = Path(__file__).resolve().parents[1] / "src/geo_eval_harness/datasets"
logger = logging.getLogger(__name__)


def main() -> None:
    """Write the manifest and report its uncompressed SHA-256 digest."""
    index = pd.read_csv(DATASETS / "open_cities_index.csv.gz", comment="#")
    train = index[index.split == "train"].reset_index(drop=True)
    if len(train) != 76818:
        raise ValueError(f"Expected 76,818 train chips, got {len(train)}.")
    rows = np.sort(np.random.default_rng(0).permutation(len(train))[:10000])
    subset = pd.DataFrame({"row": rows, "chip_id": train.iloc[rows].chip_id.to_numpy()})
    raw = subset.to_csv(index=False, lineterminator="\n").encode()
    (DATASETS / "open_cities_train_10k.csv.gz").write_bytes(gzip.compress(raw, mtime=0))
    logger.info("Manifest SHA-256: %s", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
