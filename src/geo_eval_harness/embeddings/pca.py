"""Reduce a collection to fewer channels with a train-only PCA.

The PCA is fitted on valid cells of the distinct training embeddings, so samples that
share one annual embedding (such as monthly samples of one area) count once. It is
applied to every split and stored as the sibling collection ``<dataset>_pca<k>``, with
the fitted parameters in ``pca.safetensors``. Components are not whitened, so distances
keep the geometry of the source embeddings.
"""

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from safetensors.numpy import save_file
from tqdm.auto import tqdm

from . import EMBEDDINGS_ROOT, collection_dir
from .codecs import CODECS
from .collection import CollectionWriter, collection_digest, iter_rows, read_metadata
from .download import SPLITS

if TYPE_CHECKING:
    from sklearn.decomposition import PCA

__all__ = ["FIT_CELLS", "fit_pca", "pca_collection", "project"]

logger = logging.getLogger(__name__)

#: Training cells sampled for the fit; 1M cells of 128 float32 channels take 512 MiB.
FIT_CELLS = 1_000_000


def _valid_cells(embedding: np.ndarray) -> np.ndarray:
    """Return the cells of a ``(C, H, W)`` embedding with no missing value as ``(N, C)``."""
    cells = embedding.reshape(embedding.shape[0], -1).T
    return cells[np.isfinite(cells).all(axis=1)]


def fit_pca(directory: Path, components: int, max_cells: int, seed: int) -> "PCA":
    """Fit a PCA on up to *max_cells* valid cells of the distinct training embeddings."""
    from sklearn.decomposition import PCA

    metadata = read_metadata(directory)
    codec = CODECS[metadata["codec"]]
    rng = np.random.default_rng(seed)
    per_sample = max(1, max_cells // metadata["splits"]["train"]["samples"])
    seen, chunks = set(), []
    for row in iter_rows(directory, "train"):
        scale = row["scale"].tobytes() if "scale" in row else b""
        key = hashlib.sha256(row["embedding"].tobytes() + scale).digest()
        if key in seen:
            continue
        seen.add(key)
        cells = _valid_cells(codec.decode(row))
        if len(cells) > per_sample:
            cells = cells[rng.choice(len(cells), per_sample, replace=False)]
        chunks.append(cells)
    cells = np.concatenate(chunks)
    if len(cells) < components:
        raise ValueError(f"Only {len(cells)} valid training cells for {components} components")
    logger.info(
        "Fitting a %d-component PCA on %d cells of %d distinct training embeddings",
        components,
        len(cells),
        len(seen),
    )
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


def pca_collection(  # noqa: PLR0913
    product: str,
    dataset: str,
    components: int = 16,
    *,
    root: Path = EMBEDDINGS_ROOT,
    seed: int = 0,
    max_cells: int = FIT_CELLS,
) -> Path:
    """Write ``<dataset>_pca<components>`` next to the *product* collection of *dataset*.

    Returns:
        The new collection directory.
    """
    source = collection_dir(product, dataset, root)
    if not (source / "SHA256SUMS").exists():
        raise FileNotFoundError(
            f"No {product!r} collection for {dataset!r} in {source}; run "
            f"`geo-eval-harness embeddings fetch --product {product} --dataset {dataset}`."
        )
    info = read_metadata(source)
    codec, stored = CODECS["float16"], CODECS[info["codec"]]
    pca = fit_pca(source, components, max_cells, seed)
    writer = CollectionWriter(collection_dir(product, f"{dataset}_pca{components}", root))
    for split in SPLITS:
        rows = tqdm(iter_rows(source, split), desc=f"{product}/{dataset}_pca{components}/{split}")
        writer.write_split(
            split,
            (
                {
                    **{key: value for key, value in row.items() if key != "scale"},
                    **codec.encode(project(stored.decode(row), pca)),
                }
                for row in rows
            ),
        )
    save_file(
        {
            "mean": pca.mean_.astype(np.float32),
            "components": pca.components_.astype(np.float32),
            "explained_variance": pca.explained_variance_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        },
        writer.staging / "pca.safetensors",
    )
    for key in ("arrays", "splits"):
        info.pop(key, None)
    pca_info = {
        "source": f"{product}/{dataset}",
        "source_digest": collection_digest(source),
        "fit": "train split, valid cells of distinct embeddings",
        "max_cells": max_cells,
        "seed": seed,
        "whiten": False,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return writer.finish({**info, "channels": components, "codec": codec.name, "pca": pca_info})
