"""Precomputed embedding products and their on-disk collections.

Modules:

* :mod:`.products` — read a product (AEF, TESSERA) for one sample footprint.
* :mod:`.grid` — footprint geometry and resampling onto the embedding grid.
* :mod:`.download` — pair product embeddings with dataset labels on disk.
* :mod:`.pca` — fit a train-only PCA and store the projected collection.
* :mod:`.dataset` — load a stored collection as a PyTorch dataset.

A collection lives in ``data/embeddings/<product>/<dataset>/`` with a
``metadata.json`` and one ``<split>/<index>.npz`` file per sample.
"""

from pathlib import Path

EMBEDDINGS_ROOT = Path("data/embeddings")


def collection_dir(product: str, dataset: str, root: Path = EMBEDDINGS_ROOT) -> Path:
    """Return the directory holding *product* embeddings for *dataset*."""
    return root / product / dataset
