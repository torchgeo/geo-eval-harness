"""Save the global embeddings of a classification run with stable sample IDs."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from geo_eval_harness import __version__
from geo_eval_harness.results import sanitize_name


def write_embeddings(
    directory: Path,
    metadata: Mapping[str, object],
    splits: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """Write embeddings to ``<directory>/<model name>/<dataset>-<config_hash>.parquet``.

    Each split keeps its dataset order, so ``sample_id`` (``<split>/<index>``) names the same
    sample in every model's file. The schema metadata key ``geo_eval_harness`` holds the run
    metadata as JSON.

    Args:
        directory: Root directory for embedding files.
        metadata: Result metadata of the run, including ``name``, ``dataset`` and ``config_hash``.
        splits: Embeddings ``(N, D)`` and labels ``(N,)`` or multi-hot ``(N, C)`` by split name.

    Returns:
        Path of the written file.
    """
    x = np.concatenate([features for features, _ in splits.values()])
    y = np.concatenate([labels for _, labels in splits.values()])
    rows = [
        (split, index) for split, (features, _) in splits.items() for index in range(len(features))
    ]
    table = pa.table(
        {
            "split": [split for split, _ in rows],
            "sample_id": [f"{split}/{index}" for split, index in rows],
            "label": pa.array(y)
            if y.ndim == 1
            else pa.FixedSizeListArray.from_arrays(pa.array(y.reshape(-1)), y.shape[1]),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1]),
        }
    )
    record = {**metadata, "geo_eval_harness_version": __version__, "sample_ids": "position"}
    table = table.replace_schema_metadata({"geo_eval_harness": json.dumps(record, sort_keys=True)})
    path = (
        directory
        / sanitize_name(str(metadata["name"]))
        / f"{metadata['dataset']}-{metadata['config_hash']}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path
