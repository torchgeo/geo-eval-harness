"""Precomputed embedding products, such as AEF, TESSERA and OlmoEarth, read onto sample footprints.

Downloads are resumable working copies in ``data/embedding-downloads/<product>/<dataset>/``.
"""

from pathlib import Path

DOWNLOADS_ROOT = Path("data/embedding-downloads")


def collection_dir(product: str, dataset: str, root: Path) -> Path:
    """Return the directory under *root* holding *product* embeddings for *dataset*."""
    return root / product / dataset
