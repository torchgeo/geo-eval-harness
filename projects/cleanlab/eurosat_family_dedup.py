#!/usr/bin/env python
"""Detect overlapping tiles across the EuroSAT family.

Hashes RGB tiles from ``m-eurosat`` (GeoBench V1), ``eurosat-spatial`` (the
spatial-split variant), and torchgeo's stock ``eurosat`` using a
perceptual hash (``imagehash.phash``). Reports collision groups so we can
flag tiles that appear in train of one variant and test of another — an
overlap that turns IID-test accuracy into a leakage signal.

Outputs:
- ``results/cleanlab/dedup_eurosat_family.csv``: one row per (dataset, split,
  index) with its phash.
- ``results/cleanlab/dedup_eurosat_family_collisions.csv``: cross-dataset
  collision groups.
- summary logged to stderr.
"""

import argparse
import logging
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from geo_eval_harness.datasets import get_datasets

logger = logging.getLogger(__name__)

DATASETS = ["m-eurosat", "eurosat-spatial", "eurosat"]


def _load_splits(dataset: str) -> dict[str, Dataset]:
    result = get_datasets(
        dataset_name=dataset,
        partition_name="default",
        batch_size=1,
        num_workers=0,
        return_val=True,
        image_size=None,
        bands="rgb",
        interpolation="bicubic",
    )
    assert result is not None
    train_ds, _train_loader, val_loader, test_loader = result
    return {
        "train": train_ds,
        "val": val_loader.dataset,
        "test": test_loader.dataset,
    }


def _phash_image(arr: torch.Tensor) -> str:
    img = arr.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    lo = np.percentile(img, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(img, 98, axis=(0, 1), keepdims=True)
    img = np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    return str(imagehash.phash(pil, hash_size=16))


def hash_dataset(dataset: str) -> pd.DataFrame:
    """Hash each RGB tile in the train, validation, and test splits."""
    splits = _load_splits(dataset)
    rows = []
    for split, ds in splits.items():
        n = len(ds)
        logger.info("[%s/%s] hashing %d samples", dataset, split, n)
        for i in range(n):
            sample = ds[i]
            img = sample["image"]
            label = sample.get("label")
            label_val = int(label) if label is not None and np.ndim(label) == 0 else None
            ph = _phash_image(img)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "index": i,
                    "label": label_val,
                    "phash": ph,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Write EuroSAT-family tile hashes and cross-dataset collisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("results/cleanlab/dedup_eurosat_family.csv")
    )
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    frames = [hash_dataset(d) for d in args.datasets]
    if not frames:
        raise SystemExit("nothing hashed")
    df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    grouped = df.groupby("phash")
    cross = []
    for ph, sub in grouped:
        ds_set = set(sub["dataset"].tolist())
        if len(ds_set) > 1:
            cross.append({"phash": ph, "n": len(sub), "datasets": ",".join(sorted(ds_set))})
    cross_df = pd.DataFrame(cross).sort_values("n", ascending=False) if cross else pd.DataFrame()
    logger.info("Hashed %d tiles across %d datasets", len(df), df["dataset"].nunique())
    logger.info("Cross-dataset collision groups: %d", len(cross_df))
    if not cross_df.empty:
        logger.info("\n%s", cross_df.head(20).to_string(index=False))
    cross_path = args.out.with_name(args.out.stem + "_collisions.csv")
    cross_df.to_csv(cross_path, index=False)
    logger.info("Wrote %s and %s", args.out, cross_path)


if __name__ == "__main__":
    main()
