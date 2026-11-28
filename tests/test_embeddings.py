"""Saved classification embeddings keep sample order, labels and run metadata."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from geo_eval_harness.embeddings import write_embeddings

METADATA = {"name": "timm/resnet50", "dataset": "m-eurosat", "config_hash": "0123456789abcdef"}


@pytest.mark.parametrize("multilabel", [False, True])
def test_write_embeddings_round_trips_ids_labels_and_metadata(
    tmp_path: Path, *, multilabel: bool
) -> None:
    rng = np.random.default_rng(0)
    splits = {
        split: (
            rng.normal(size=(n, 4)).astype(np.float32),
            rng.integers(0, 2, size=(n, 3)).astype(np.float32)
            if multilabel
            else rng.integers(0, 5, size=n),
        )
        for split, n in (("val", 2), ("test", 3))
    }

    path = write_embeddings(tmp_path, METADATA, splits)

    assert path == tmp_path / "timm_resnet50" / "m-eurosat-0123456789abcdef.parquet"
    table = pq.read_table(path)
    assert table["sample_id"].to_pylist() == ["val/0", "val/1", "test/0", "test/1", "test/2"]
    assert table["split"].to_pylist() == ["val", "val", "test", "test", "test"]
    for column, index in (("embedding", 0), ("label", 1)):
        expected = np.concatenate([arrays[index] for arrays in splits.values()])
        np.testing.assert_array_equal(np.asarray(table[column].to_pylist()), expected)
    record = json.loads(table.schema.metadata[b"geo_eval_harness"])
    assert (record["config_hash"], record["sample_ids"]) == ("0123456789abcdef", "position")
