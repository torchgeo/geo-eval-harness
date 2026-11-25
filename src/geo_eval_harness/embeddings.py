"""Save the global embeddings of a classification run with stable sample IDs and compare them."""

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from geo_eval_harness import __version__
from geo_eval_harness.results import sanitize_name

MAX_CKA_SAMPLES = 10_000


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


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Return the linear CKA of two embeddings of the same samples (Kornblith et al., 2019).

    Rows are samples and the widths may differ. The value lies in [0, 1], equals 1 for identical
    features, and does not change when either embedding is rescaled or rotated. It is NaN when an
    embedding is constant across samples.
    """
    x = x - x.mean(axis=0, dtype=np.float64)
    y = y - y.mean(axis=0, dtype=np.float64)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(y.T @ x) ** 2 / denominator) if denominator else float("nan")


def cka_matrix(
    paths: Sequence[Path], *, split: str = "test", max_samples: int = MAX_CKA_SAMPLES
) -> pd.DataFrame:
    """Return the linear CKA between every pair of embedding files on one split.

    Every file must hold the same dataset, the same sample IDs in the same order, and the same
    labels; otherwise a ``ValueError`` names the first file that differs. A split with more than
    ``max_samples`` rows keeps the rows whose sample IDs have the smallest SHA-256 digests, so all
    models are compared on the same subset.

    Args:
        paths: Files written by :func:`write_embeddings`.
        split: Split to compare.
        max_samples: Largest number of samples to compare.

    Returns:
        Square matrix indexed by ``<model name>@<config_hash>``.
    """
    records, tables = [], []
    for path in paths:
        table = pq.read_table(path)
        records.append(json.loads(table.schema.metadata[b"geo_eval_harness"]))
        tables.append(table.filter(pa.array(np.asarray(table["split"]) == split)))
    for path, record, table in zip(paths[1:], records[1:], tables[1:], strict=True):
        if (
            record["dataset"] != records[0]["dataset"]
            or not table["sample_id"].equals(tables[0]["sample_id"])
            or not table["label"].equals(tables[0]["label"])
        ):
            raise ValueError(f"{path} does not hold the same {split} samples as {paths[0]}")
    ids = tables[0]["sample_id"].to_pylist()
    if not ids:
        raise ValueError(f"{paths[0]} has no {split} rows")
    digests = [hashlib.sha256(sample_id.encode()).hexdigest() for sample_id in ids]
    keep = np.sort(np.argsort(digests)[:max_samples])
    features = []
    for table in tables:
        column = table["embedding"].combine_chunks()
        features.append(column.flatten().to_numpy().reshape(len(column), -1)[keep])
    values = np.empty((len(features), len(features)))
    for i, j in itertools.combinations_with_replacement(range(len(features)), 2):
        values[i, j] = values[j, i] = linear_cka(features[i], features[j])
    names = [f"{record['name']}@{record['config_hash']}" for record in records]
    return pd.DataFrame(values, index=names, columns=names)
