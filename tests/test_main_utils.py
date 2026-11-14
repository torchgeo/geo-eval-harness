"""Unit tests for utility helpers in ``geo_eval_harness.main``."""

from unittest import mock

import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader, Dataset

from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.main import _expand_dataset_list, evaluate_profile
from geo_eval_harness.resume import (
    _completed_run_keys,
    filter_completed_metric_rows,
    normalize_bands_value,
)
from geo_eval_harness.segmentation_task import build_seg_probe_and_solver


class _ImageOnlyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        return {"image": torch.ones(3, 8, 8)}


def test_expand_dataset_list_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("geo_eval_harness.main.list_datasets", lambda: ["m-eurosat", "benv2"])
    assert _expand_dataset_list("all") == ["m-eurosat", "benv2"]
    assert _expand_dataset_list(["all"]) == ["m-eurosat", "benv2"]


def test_normalize_bands_value_none_and_list() -> None:
    assert normalize_bands_value(None) == "all"
    cfg_list = ["red", "green"]
    assert normalize_bands_value(cfg_list) == "red,green"


def test_completed_run_keys_metric_name_absent_returns_empty() -> None:
    existing = pd.DataFrame([{"dataset": "m-eurosat", "method": "knn5"}])
    assert _completed_run_keys(existing, ["dataset", "method"], metric_name="accuracy") == set()


def test_filter_completed_metric_rows_partial_filtering() -> None:
    rows = [
        {"dataset": "m-eurosat", "method": "knn5", "metric_name": "accuracy"},
        {"dataset": "m-eurosat", "method": "knn5", "metric_name": "f1"},
    ]
    completed = {"accuracy": {("m-eurosat", "knn5")}}
    filtered = filter_completed_metric_rows(rows, completed, ["dataset", "method"])
    assert filtered == [{"dataset": "m-eurosat", "method": "knn5", "metric_name": "f1"}]


def test_build_seg_probe_and_solver_rejects_empty_layers() -> None:
    config = SegmentationConfig(layers=[])
    with pytest.raises(ValueError, match=r"requires segmentation\.layers"):
        build_seg_probe_and_solver(
            model=torch.nn.Identity(),
            num_classes=2,
            config=config,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "greater than 0"),
        ("batch_size", 0, "greater than 0"),
        ("learning_rate", 0.0, "greater than 0"),
        ("cache_features", "true", "valid boolean"),
        ("cache_dtype", "bfloat16", "Input should be"),
    ],
)
def test_segmentation_runtime_config_rejects_invalid_values(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SegmentationConfig.model_validate({field: value})


def test_evaluate_profile_adds_cpu_metrics_branch() -> None:
    loader = DataLoader(_ImageOnlyDataset(), batch_size=2, shuffle=False, num_workers=0)
    common_meta = {
        "dataset": "m-eurosat",
        "seed": 0,
        "model": "mock.Model",
        "name": "mock",
        "normalization": "identity",
        "image_size": 8,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "num_classes": 10,
        "config_hash": "test-config-hash",
        "c_range_start": -2,
        "c_range_stop": 2,
        "c_range_num": 3,
        "merge_val": False,
        "bootstrap": 10,
    }

    with (
        mock.patch(
            "geo_eval_harness.main.measure_profile",
            return_value={"params_m": 0.1, "throughput_samples_per_sec": 20.0},
        ),
        mock.patch(
            "geo_eval_harness.main.measure_cpu_throughput",
            return_value={
                "throughput_samples_per_sec_cpu": 3.0,
                "latency_ms_per_batch_p50_cpu": 12.0,
            },
        ),
    ):
        common_meta.update(feature_dim=8, n_train=2, n_val=2, n_test=2)
        rows = evaluate_profile(
            model=torch.nn.Identity(),
            sample_loader=loader,
            cfg=RunConfig.model_validate(
                {
                    "model": {"name": "rcf"},
                    "datasets": ["m-eurosat"],
                    "runtime": {"device": "cpu"},
                    "profile": {
                        "n_warmup": 0,
                        "n_measure": 1,
                        "cpu_throughput": {
                            "enabled": True,
                            "batch_size": 2,
                            "n_warmup": 0,
                            "n_measure": 1,
                            "time_budget_s": 1.0,
                        },
                    },
                }
            ),
            common_meta=common_meta,
        )

    metric_names = {row["metric_name"] for row in rows}
    assert "params_m" in metric_names
    assert "throughput_samples_per_sec" in metric_names
    assert "throughput_samples_per_sec_cpu" in metric_names
    assert "latency_ms_per_batch_p50_cpu" in metric_names
