"""Data-only GeoBench V1 metadata decoding."""

import json
import math
from collections.abc import Mapping


def decode_metadata(value: object) -> dict:
    """Decode JSON metadata containing band names and numeric classification labels."""
    if not isinstance(value, (str, bytes)):
        raise TypeError(f"Expected JSON metadata text or bytes, got {type(value).__name__}.")
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise TypeError("GeoBench metadata must be a JSON object.")

    bands = metadata.get("bands_order")
    if (
        not isinstance(bands, list)
        or not bands
        or any(not isinstance(name, str) or not name for name in bands)
    ):
        raise ValueError("GeoBench metadata bands_order must be a non-empty list of band names.")

    label = metadata.get("label")
    labels = label if isinstance(label, list) else [label]
    if not labels or any(
        not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        for value in labels
    ):
        raise ValueError("GeoBench metadata label must be a number or a non-empty list of numbers.")
    return metadata


def read_hdf5_metadata(attrs: Mapping[str, object]) -> dict:
    """Read JSON metadata without interpreting pickle attributes."""
    if "metadata_json" not in attrs:
        raise ValueError(
            "GeoBench V1 HDF5 samples require a 'metadata_json' attribute. "
            "Legacy pickle metadata is not supported. Replace old caches with "
            "'geo-eval-harness download geobench_v1' or supply JSON metadata."
        )
    return decode_metadata(attrs["metadata_json"])
