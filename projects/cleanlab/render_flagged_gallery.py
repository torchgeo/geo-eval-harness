#!/usr/bin/env python
"""Render top-K flagged tiles per dataset for manual cleanlab review.

For each ``results/cleanlab/<dataset>_<split>.csv`` produced by
``run_cleanlab_audit.py``, take the top-K samples ranked by ``issue_score``,
fetch their RGB tiles from the dataset, and stitch them into a single PNG
grid. Captions show ``given_label -> guessed_label (score)``.

Output: ``results/cleanlab/galleries/<dataset>_<split>.png``.
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from geo_eval_harness.datasets import get_bench_dataset_class, get_datasets

logger = logging.getLogger(__name__)


def _load_split_dataset(dataset: str, split: str, partition: str = "default") -> Dataset:
    result = get_datasets(
        dataset_name=dataset,
        partition_name=partition,
        batch_size=1,
        num_workers=0,
        return_val=True,
        image_size=None,
        bands="all",
        interpolation="bicubic",
    )
    assert result is not None
    train_dataset, _train_loader, val_loader, test_loader = result
    if split == "train":
        return train_dataset
    if split == "val":
        return val_loader.dataset
    if split == "test":
        return test_loader.dataset
    raise ValueError(split)


def _to_rgb(image: torch.Tensor, rgb_idx: list[int]) -> np.ndarray:
    """Take (C, H, W) tensor → (H, W, 3) uint8 with per-image min-max contrast."""
    if image.ndim != 3:
        raise ValueError(f"expected (C, H, W); got {tuple(image.shape)}")
    arr = image[rgb_idx].detach().cpu().float().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    lo = np.percentile(arr, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(arr, 98, axis=(0, 1), keepdims=True)
    arr = np.clip((arr - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def render_gallery(
    dataset: str,
    split: str,
    flagged: pd.DataFrame,
    out_path: Path,
    cols: int = 10,
) -> None:
    """Save a grid of flagged RGB tiles for manual review."""
    if flagged.empty:
        logger.warning("[%s/%s] no flagged samples", dataset, split)
        return

    ds_cls = get_bench_dataset_class(dataset)
    ds = _load_split_dataset(dataset, split)
    bench = ds_cls()
    rgb_idx = bench.rgb_indices or [0, 1, 2]

    rows = (len(flagged) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    multilabel = "given_label" not in flagged.columns

    for ax in axes.ravel():
        ax.axis("off")

    for k, (_, row) in enumerate(flagged.iterrows()):
        idx = int(row["index"])
        sample = ds[idx]
        img = sample["image"]
        rgb = _to_rgb(img, rgb_idx)
        ax = axes[k // cols, k % cols]
        ax.imshow(rgb)
        if multilabel:
            caption = f"i={idx}\nscore={row['issue_score']:.2f}"
        else:
            g = int(row["given_label"])
            p = int(row["guessed_label"])
            caption = f"{g}->{p}\n{row['issue_score']:.2f}"
        ax.set_title(caption, fontsize=6)

    fig.suptitle(f"{dataset} / {split} — top {len(flagged)} flagged", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.warning("[%s/%s] wrote %s", dataset, split, out_path)


def main() -> None:
    """Render galleries from the selected per-sample issue reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issues-dir",
        type=Path,
        default=Path("results/cleanlab"),
        help="Directory containing <dataset>_<split>.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cleanlab/galleries"),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets to render (default: all CSVs in --issues-dir).",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    csvs = sorted(args.issues_dir.glob("*_*.csv"))
    csvs = [c for c in csvs if c.stem != "summary"]
    pairs = []
    for c in csvs:
        if "_" not in c.stem:
            continue
        dataset, _, split = c.stem.rpartition("_")
        if split not in args.splits:
            continue
        if args.datasets and dataset not in args.datasets:
            continue
        pairs.append((dataset, split, c))

    for dataset, split, csv_path in pairs:
        out = args.out_dir / f"{dataset}_{split}.png"
        frame = pd.read_csv(csv_path)
        flagged = (
            frame[frame["is_issue"]].sort_values("issue_score", ascending=False).head(args.top_k)
        )
        render_gallery(dataset, split, flagged, out, args.cols)


if __name__ == "__main__":
    main()
