"""Embedding products, one module each, registered by name in :data:`PRODUCTS`."""

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
