#!/usr/bin/env python3
"""Render embedding downloads next to the source RGB and target as PNG sheets.

Reads ``<root>/<product>/<dataset>/<split>/<index>.npz`` as written by
``geo-eval-harness embeddings download``, joins products by sample index, and writes sheets of
a few samples each to ``--output-dir``. Each row is labelled with its sample ID and each
product's coverage. PCA colours are fitted per sheet and product for display only, so they are
not comparable across sheets or products. Datasets are read from ``data/``.

Usage::

    python scripts/visualize_embeddings.py --products aef tessera --datasets treesatai spacenet7
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.embeddings import DOWNLOADS_ROOT
from geo_eval_harness.embeddings.pca import project

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
GRAY = (0.5, 0.5, 0.5)


def stretch(pixels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``(..., 3)`` pixels to [0, 1] by the 2nd/98th percentiles of ``(N, 3)`` values."""
    low, high = np.percentile(reference, [2, 98], axis=0)
    return np.clip((pixels - low) / np.maximum(high - low, 1e-6), 0, 1)


def load(path: Path | None) -> np.ndarray | None:
    """Return the ``(C, H, W)`` embedding stored at *path*, or None if it was not extracted."""
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as sample:
        return sample["embedding"].astype(np.float32)


def pca_images(embeddings: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Colour one product's embeddings on a sheet by a display-only PCA of their valid cells."""
    present = [e.reshape(len(e), -1).T for e in embeddings if e is not None]
    cells = np.concatenate(present) if present else np.empty((0, 1))
    cells = cells[np.isfinite(cells).all(axis=1)]
    if len(cells) < 3:
        return [None] * len(embeddings)
    pca = PCA(n_components=3).fit(cells)
    reference = pca.transform(cells)
    images = []
    for embedding in embeddings:
        rgb = None
        if embedding is not None:
            rgb = stretch(project(embedding, pca).transpose(1, 2, 0), reference)
            rgb[np.isnan(rgb).any(axis=-1)] = GRAY
        images.append(rgb)
    return images


def draw_target(axis: Axes, target: np.ndarray, class_names: tuple[str, ...]) -> None:
    """Show a mask as an image and a classification label as its positive classes."""
    if target.ndim == 2:
        # Fixed limits give class i the same tab20 colour on every row.
        axis.imshow(target, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
        return
    positive = [int(target)] if target.ndim == 0 else np.flatnonzero(target).tolist()
    text = "\n".join(class_names[i] if class_names else str(i) for i in positive)
    axis.set_facecolor("white")
    axis.text(0.5, 0.5, text or "no positive labels", ha="center", va="center", fontsize=8)


def render_sheet(
    rows: list[dict], products: list[str], class_names: tuple[str, ...], path: Path
) -> None:
    """Draw one row per sample: RGB, target and one PCA panel per product."""
    columns = ["RGB", "target", *products]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(2.6 * len(columns), 2.8 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    colours = [pca_images([row["embeddings"][p] for row in rows]) for p in range(len(products))]
    for r, (row, line) in enumerate(zip(rows, axes, strict=True)):
        for axis, title in zip(line, columns, strict=True):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor(GRAY)
            if r == 0:
                axis.set_title(title, fontsize=9)
        line[0].imshow(row["rgb"])
        line[0].set_ylabel(row["caption"].replace(", ", "\n"), fontsize=8)
        draw_target(line[1], row["target"], class_names)
        for p, axis in enumerate(line[2:]):
            embedding = row["embeddings"][p]
            if embedding is None:
                axis.text(0.5, 0.5, "not extracted", ha="center", va="center", fontsize=8)
            elif not np.isfinite(embedding).all(axis=0).any():
                axis.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8)
            elif colours[p][r] is not None:
                axis.imshow(colours[p][r], interpolation="nearest")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def sample_row(
    sample: dict, index: int, embeddings: list[np.ndarray | None], products: list[str]
) -> dict:
    """Collect the panels and caption of one sample."""
    rgb = np.asarray(sample["image"], dtype=np.float32).transpose(1, 2, 0)
    coverage = [
        f"{p} not extracted" if e is None else f"{p} {np.isfinite(e).all(axis=0).mean():.0%}"
        for p, e in zip(products, embeddings, strict=True)
    ]
    return {
        "rgb": stretch(rgb, rgb.reshape(-1, 3)),
        "target": np.asarray(sample["mask"] if "mask" in sample else sample["label"]),
        "embeddings": embeddings,
        "caption": f"{index:06d}: " + ", ".join(coverage),
    }


def render_dataset(args: argparse.Namespace, dataset: str) -> None:
    """Render the sheets of one dataset."""
    products, split, size = args.products, args.split, args.rows_per_sheet
    files = {
        p: {int(f.stem): f for f in (args.root / p / dataset / split).glob("*.npz")}
        for p in products
    }
    indices = sorted(set().union(*files.values()))
    if not indices:
        raise FileNotFoundError(f"No {split} downloads of {dataset} under {args.root}")
    bench = get_bench_dataset_class(dataset)()
    data = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
    class_names = tuple(getattr(bench, "class_names", ()))
    for start in range(0, len(indices), size):
        rows = [
            sample_row(data[i], i, [load(files[p].get(i)) for p in products], products)
            for i in indices[start : start + size]
        ]
        name = f"{dataset}_{start // size:03d}.png"
        render_sheet(rows, products, class_names, args.output_dir / name)
    logger.info("%s: %d samples", dataset, len(indices))


def main() -> None:
    """Write the sheets of every requested dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / DOWNLOADS_ROOT)
    parser.add_argument("--products", nargs="+", default=["aef", "tessera"])
    parser.add_argument("--datasets", nargs="+", default=["treesatai", "spacenet7"])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "embedding-visualizations"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows-per-sheet", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        render_dataset(args, dataset)
    logger.info("Sheets: %s", args.output_dir)


if __name__ == "__main__":
    main()
