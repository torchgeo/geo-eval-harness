"""Reduce a stored collection to fewer channels with a train-only PCA.

The PCA is fitted on valid training cells only, then applied to every split.
The result is stored as its own collection, ``<product>-pca<k>``, and can be
evaluated like any other product.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from . import EMBEDDINGS_ROOT, collection_dir
from .download import save_sample, write_metadata

if TYPE_CHECKING:
    from sklearn.decomposition import PCA

__all__ = ["FIT_CELLS", "fit_pca", "pca_collection", "project"]

logger = logging.getLogger(__name__)

#: Training cells sampled for the fit; 1M cells of 128 float32 channels take 512 MiB.
FIT_CELLS = 1_000_000


def _valid_cells(file: Path) -> np.ndarray:
    """Return a file's embedding cells with no missing value as ``(N, C)``."""
    with np.load(file) as sample:
        embedding = sample["embedding"].astype(np.float32)
    cells = embedding.reshape(embedding.shape[0], -1).T
    return cells[np.isfinite(cells).all(axis=1)]


def fit_pca(train_files: list[Path], components: int, max_cells: int, seed: int) -> "PCA":
    """Fit a PCA on up to *max_cells* valid cells, sampled evenly across *train_files*."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    per_file = max(1, max_cells // len(train_files))
    chunks = []
    for file in train_files:
        cells = _valid_cells(file)
        if len(cells) > per_file:
            cells = cells[rng.choice(len(cells), per_file, replace=False)]
        chunks.append(cells)
    cells = np.concatenate(chunks)
    if len(cells) < components:
        raise ValueError(f"Only {len(cells)} valid training cells for {components} components")
    logger.info("Fitting a %d-component PCA on %d training cells", components, len(cells))
    return PCA(n_components=components, random_state=seed).fit(cells)


def project(embedding: np.ndarray, pca: "PCA") -> np.ndarray:
    """Project ``(C, H, W)`` embeddings to ``(k, H, W)``; cells with missing values stay NaN."""
    channels, height, width = embedding.shape
    cells = embedding.reshape(channels, -1).T.astype(np.float32)
    valid = np.isfinite(cells).all(axis=1)
    out = np.full((len(cells), pca.n_components_), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = pca.transform(cells[valid])
    return out.T.reshape(-1, height, width)


def pca_collection(
    product: str,
    dataset: str,
    components: int = 16,
    *,
    root: Path = EMBEDDINGS_ROOT,
    seed: int = 0,
) -> Path:
    """Write ``<product>-pca<components>`` for *dataset* from a downloaded collection.

    Returns:
        The new collection directory.
    """
    source = collection_dir(product, dataset, root)
    train_files = sorted((source / "train").glob("*.npz"))
    if not train_files:
        raise FileNotFoundError(
            f"No training embeddings in {source}; run "
            f"`geo-eval-harness embeddings download --product {product} --dataset {dataset}`."
        )
    name = f"{product}-pca{components}"
    target = collection_dir(name, dataset, root)
    pca = fit_pca(train_files, components, FIT_CELLS, seed)
    for file in tqdm(sorted(source.glob("*/*.npz")), desc=f"{name}/{dataset}"):
        with np.load(file) as sample:
            arrays = dict(sample)
        arrays["embedding"] = project(arrays["embedding"], pca).astype(np.float16)
        save_sample(target / file.parent.name / file.name, **arrays)
    info = json.loads((source / "metadata.json").read_text())
    info.pop("splits", None)
    pca_info = {
        "source": product,
        "seed": seed,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    write_metadata(target, {**info, "product": name, "channels": components, "pca": pca_info})
    return target
