"""Precomputed embedding products, such as AEF, TESSERA and OlmoEarth, read onto sample footprints.

Downloads are resumable working copies in ``data/embedding-downloads/<product>/<dataset>/``.
Published collections live in ``data/embeddings/<product>/<dataset>/``, which mirrors the
product's Hugging Face repository; a PCA variant is the sibling collection ``<dataset>_pca<k>``.
"""

from pathlib import Path

EMBEDDINGS_ROOT = Path("data/embeddings")
DOWNLOADS_ROOT = Path("data/embedding-downloads")


def collection_dir(product: str, dataset: str, root: Path) -> Path:
    """Return the directory under *root* holding *product* embeddings for *dataset*."""
    return root / product / dataset
