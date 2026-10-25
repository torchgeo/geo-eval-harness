"""Precomputed embedding products and their on-disk collections.

Modules:

* :mod:`.products` — read a product (AEF, TESSERA) for one sample footprint.
* :mod:`.grid` — footprint geometry and resampling onto the embedding grid.
* :mod:`.download` — pair product embeddings with dataset labels, one npz per sample.
* :mod:`.codecs` — store embeddings in their product's native quantization.
* :mod:`.collection` — the published Parquet layout.
* :mod:`.export` — turn a completed download into a published collection.
* :mod:`.fetch` — download a published collection at a pinned revision.
* :mod:`.pca` — fit a train-only PCA and store the projected collection.
* :mod:`.dataset` — load a collection as a PyTorch dataset.

Downloads are resumable working copies in ``data/embedding-downloads/<product>/<dataset>/``.
Evaluation reads published collections from ``data/embeddings/<product>/<dataset>/``,
which mirrors the product's Hugging Face repository; a PCA variant is the sibling
collection ``<dataset>_pca<k>``.
"""

from pathlib import Path

EMBEDDINGS_ROOT = Path("data/embeddings")
DOWNLOADS_ROOT = Path("data/embedding-downloads")


def collection_dir(product: str, dataset: str, root: Path = EMBEDDINGS_ROOT) -> Path:
    """Return the directory holding *product* embeddings for *dataset*."""
    return root / product / dataset
