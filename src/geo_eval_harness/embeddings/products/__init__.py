"""Embedding products: read one product year onto an embedding grid.

Each product lives in its own module. Add one by subclassing :class:`EmbeddingProduct`
in a new module and registering it in :data:`PRODUCTS`.
:meth:`EmbeddingProduct.read` is the only method to implement.
"""

from .aef import AEF
from .base import EmbeddingProduct, decode_signed_square
from .olmo_earth import OlmoEarth, OlmoEarth64
from .tessera import Tessera, decode_tessera

__all__ = [
    "AEF",
    "PRODUCTS",
    "EmbeddingProduct",
    "OlmoEarth",
    "OlmoEarth64",
    "Tessera",
    "decode_signed_square",
    "decode_tessera",
]

PRODUCTS: dict[str, type[EmbeddingProduct]] = {
    "aef": AEF,
    "tessera": Tessera,
    "olmoearth-v1-3": OlmoEarth,
    "olmoearth-v1-3-64": OlmoEarth64,
}
