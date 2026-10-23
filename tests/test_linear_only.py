"""Linear-only executes and resumes without requiring the KNN backend."""

import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from geo_eval_harness.main import main
from tests.support.runner import (
    _chainable_model_mock,
    _compose_cfg,
    _hash_for,
    _resume_row,
    _synthetic_loaders,
)


def _embeddings(kind: str) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(42)
    splits = []
    for count in (24, 12, 12):
        features = rng.standard_normal((count, 4), dtype=np.float32)
        labels = (
            rng.integers(0, 2, size=(count, 3)).astype(np.float32)
            if kind == "multilabel"
            else np.arange(count, dtype=np.int64) % (2 if kind == "binary" else 4)
        )
        splits.append((features, labels))
    return splits


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel"])
def test_only_linear_runs_and_resumes_without_knn(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "linear.csv"
    config = _compose_cfg(
        output,
        {
            "datasets": ["m-bigearthnet" if kind == "multilabel" else "m-eurosat"],
            "classification": {
                "methods": ["linear"],
                "knn_device": "cuda:999",
                "linear": {"refit_train_val": False},
                "calibration": {"temp_scale": True},
            },
        },
    )
    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=_chainable_model_mock()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_embeddings(kind)),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn", side_effect=AssertionError("KNN evaluated")
        ),
        mock.patch(
            "geo_eval_harness.knn.resolve_knn_device",
            side_effect=AssertionError("KNN device resolved"),
        ),
        mock.patch(
            "geo_eval_harness.knn.KNNClassifier", side_effect=AssertionError("KNN constructed")
        ),
    ):
        main(config, strict=True)
    rows = pd.read_csv(output)
    assert rows["method"].tolist() == ["linear"]
    assert rows["metric_name"].tolist() == ["micro_mAP" if kind == "multilabel" else "accuracy"]
    assert rows["metric_value"].between(0, 1).all()
    assert np.isfinite(rows["ece"]).all()
    assert rows["temperature"].gt(0).all()

    config.output.resume = True
    with mock.patch(
        "geo_eval_harness.main.get_datasets", side_effect=AssertionError("resume loaded data")
    ):
        main(config, strict=True)
    assert len(pd.read_csv(output)) == 1


def test_linear_resume_reuses_existing_linear_row_without_knn(tmp_path: Path) -> None:
    config = _compose_cfg(tmp_path / "linear.csv", {"classification": {"methods": ["linear"]}})
    combined = _compose_cfg(tmp_path / "linear.csv")
    assert _hash_for(config) == _hash_for(combined)
    pd.DataFrame([_resume_row(combined, method="linear", metric_name="accuracy")]).to_csv(
        config.output.file, index=False
    )
    config.output.resume = True
    with mock.patch(
        "geo_eval_harness.main.get_datasets", side_effect=AssertionError("resume loaded data")
    ):
        main(config)
    assert pd.read_csv(config.output.file)["method"].tolist() == ["linear"]


def test_completed_knn_alone_does_not_satisfy_linear_only_resume(tmp_path: Path) -> None:
    config = _compose_cfg(
        tmp_path / "partial.csv",
        {
            "classification": {"methods": ["linear"], "knn_device": "cuda:999"},
            "output": {"resume": True},
        },
    )
    pd.DataFrame([_resume_row(config, method="knn5", metric_name="accuracy")]).to_csv(
        config.output.file, index=False
    )
    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=_chainable_model_mock()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_embeddings("binary")),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn", side_effect=AssertionError("KNN evaluated")
        ),
    ):
        main(config, strict=True)
    assert pd.read_csv(config.output.file)["method"].tolist() == ["knn5", "linear"]


def test_linear_only_failure_never_writes_fabricated_knn_row(tmp_path: Path) -> None:
    output = tmp_path / "linear.csv"
    config = _compose_cfg(output, {"classification": {"methods": ["linear"]}})
    with (
        mock.patch("geo_eval_harness.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("geo_eval_harness.main.build_model", return_value=_chainable_model_mock()),
        mock.patch("geo_eval_harness.main.embed_split", side_effect=_embeddings("binary")),
        mock.patch(
            "geo_eval_harness.main.evaluate_logistic", side_effect=RuntimeError("linear failed")
        ),
        mock.patch(
            "geo_eval_harness.main.evaluate_knn", side_effect=AssertionError("KNN evaluated")
        ),
        pytest.raises(RuntimeError, match="linear failed"),
    ):
        main(config, strict=True)
    assert not output.exists()


def test_linear_only_does_not_import_faiss_or_knn_backend() -> None:
    code = """
import importlib
import importlib.abc
import sys
from unittest.mock import patch

class BlockKNN(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {'faiss', 'faissknn', 'geo_eval_harness.knn'}:
            raise AssertionError(f'unselected backend imported: {fullname}')
sys.meta_path.insert(0, BlockKNN())

from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.main import dataset_metadata, run_dataset
from geo_eval_harness.resume import ResumeState
from tests.support.runner import _synthetic_loaders, _synthetic_embeddings, _chainable_model_mock
config = RunConfig.model_validate({
    'model': {'name': 'rcf'}, 'datasets': ['m-eurosat'],
    'runtime': {'device': 'cpu'},
    'classification': {'methods': ['linear'], 'knn_device': 'cuda:999', 'bootstrap_samples': 2,
        'linear': {'c_log10_start': -2.0, 'c_log10_stop': -1.0, 'c_count': 2}},
})
module = importlib.import_module('geo_eval_harness.main')
with (
    patch.object(module, 'get_datasets', return_value=_synthetic_loaders()),
    patch.object(module, 'build_model', return_value=_chainable_model_mock()),
    patch.object(module, 'embed_split', side_effect=_synthetic_embeddings()),
):
    rows = list(run_dataset(config, 'm-eurosat', ResumeState(set(), {})))
assert [row['method'] for batch, _, _ in rows for row in batch] == ['linear']
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
