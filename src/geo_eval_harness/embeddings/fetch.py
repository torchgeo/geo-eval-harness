"""Download published collections from Hugging Face at a pinned revision.

Each product has one dataset repository with a directory per collection, mirrored
under ``data/embeddings/<product>/``. Every file is checked against the collection's
``SHA256SUMS``. For published products, that file is itself checked against the
digest packaged in ``checksums/embeddings-<product>.sha256``.
"""

import re
from pathlib import Path

from geo_eval_harness.hub import fetch_pinned, load_checksums, parse_checksums, verify_checksums

from . import EMBEDDINGS_ROOT, collection_dir

__all__ = ["PUBLISHED", "fetch"]

#: ``{product: (repository, commit)}`` of each published product.
PUBLISHED: dict[str, tuple[str, str]] = {
    "aef": ("nilsleh/aef-embeddings-eval-datasets", "8c632b7d4a5945c3101aecf2e18aa1a144c51694"),
    "tessera": (
        "nilsleh/tessera-embeddings-eval-datasets",
        "2c8239080a0cfc75cd186926634d149f6459d0f2",
    ),
}


def fetch(
    product: str,
    collection: str,
    *,
    root: Path = EMBEDDINGS_ROOT,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download and verify *collection* (e.g. ``benv2`` or ``benv2_pca16``) of *product*.

    Args:
        product: Product name, e.g. ``"aef"``.
        collection: Collection directory in the product repository.
        root: Collection root directory.
        repo_id: Repository to read instead of the published one; requires *revision*.
        revision: Full commit hash to read.

    Returns:
        The collection directory.
    """
    published = repo_id is None
    if published:
        if product not in PUBLISHED:
            raise ValueError(
                f"No published collections for {product!r}; choose from {sorted(PUBLISHED)}"
            )
        repo_id, revision = PUBLISHED[product]
    if revision is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"revision must be a full commit hash, got {revision!r}")
    expected = {}
    if published:
        manifest = load_checksums(f"embeddings-{product}.sha256")
        key = f"{collection}/SHA256SUMS"
        if key not in manifest:
            choices = sorted(name.split("/")[0] for name in manifest)
            raise ValueError(
                f"{collection!r} is not published for {product!r}; choose from {choices}"
            )
        expected = {key: manifest[key]}
    fetch_pinned(repo_id, revision, root / product, expected, allow_patterns=[f"{collection}/*"])
    directory = collection_dir(product, collection, root)
    checksums = parse_checksums((directory / "SHA256SUMS").read_text())
    verify_checksums(directory, checksums, complete_glob="*.parquet")
    return directory
