"""Saved classification embeddings keep sample order, labels and run metadata."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from geo_eval_harness.embeddings import cka_matrix, linear_cka, write_embeddings
from scripts import cka_matrix as cka_script

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


def test_linear_cka_is_one_for_identical_features_and_ignores_scale_and_rotation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6))
    y = x[:, :4] @ rng.normal(size=(4, 4)) + rng.normal(size=(50, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    centered_x, centered_y = x - x.mean(axis=0), y - y.mean(axis=0)
    gram_x, gram_y = centered_x @ centered_x.T, centered_y @ centered_y.T
    gram_cka = (gram_x * gram_y).sum() / np.sqrt((gram_x**2).sum() * (gram_y**2).sum())

    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, y) == pytest.approx(gram_cka)
    assert 0.0 < linear_cka(x, y) < 1.0
    assert linear_cka(3.0 * x, 0.5 * y) == pytest.approx(linear_cka(x, y))
    assert linear_cka(x @ rotation, y) == pytest.approx(linear_cka(x, y))


def test_cka_matrix_script_compares_files_and_rejects_shuffled_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    labels = np.arange(12)
    x = rng.normal(size=(12, 5)).astype(np.float32)
    models = {"a": x, "b": x.copy(), "c": rng.normal(size=(12, 3)).astype(np.float32)}
    paths = [
        write_embeddings(tmp_path, {**METADATA, "name": name}, {"test": (features, labels)})
        for name, features in models.items()
    ]
    output = tmp_path / "cka.csv"

    cka_script.main([*map(str, paths), "--max-samples", "8", "--output", str(output)])

    matrix = pd.read_csv(output, index_col=0)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix, matrix.T)
    assert matrix.loc["a@0123456789abcdef", "b@0123456789abcdef"] == pytest.approx(1.0)
    order = np.roll(labels, 1)
    shuffled = write_embeddings(
        tmp_path, {**METADATA, "name": "shuffled"}, {"test": (x[order], labels[order])}
    )
    with pytest.raises(ValueError, match="same test samples"):
        cka_matrix([paths[0], shuffled])
