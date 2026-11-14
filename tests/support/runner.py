"""Small deterministic inputs for runner orchestration tests."""

from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from geo_eval_harness.config.presets import merge_settings, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.main import dataset_metadata
from geo_eval_harness.resume import resume_config_hash


class _DictTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[index], "label": self._labels[index]}


def _compose_cfg(output_path: Path, overrides: dict | None = None) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings(
            {
                "model": {"name": "rcf"},
                "datasets": ["m-eurosat"],
                "runtime": {"batch_size": 4, "workers": 0, "device": "cpu"},
                "classification": {
                    "bootstrap_samples": 5,
                    "linear": {"c_log10_start": -2.0, "c_log10_stop": -1.0, "c_count": 2},
                },
                "output": {"file": str(output_path)},
            },
            overrides or {},
        )
    )


def _synthetic_loaders() -> tuple[_DictTensorDataset, DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(0)
    datasets = [
        _DictTensorDataset(
            torch.rand(size, 3, 8, 8, generator=generator) * 3000,
            torch.arange(size) % 10,
        )
        for size in (16, 8, 8)
    ]
    loaders = [DataLoader(dataset, batch_size=4, num_workers=0) for dataset in datasets]
    return datasets[0], *loaders


def _synthetic_embeddings() -> list[tuple[np.ndarray, np.ndarray]]:
    generator = np.random.default_rng(0)
    return [
        (
            generator.standard_normal((size, 8), dtype=np.float32),
            np.arange(size, dtype=np.int64) % 10,
        )
        for size in (16, 8, 8)
    ]


def _hash_for(cfg: RunConfig, ds_name: str | None = None) -> str:
    dataset = ds_name or cfg.datasets[0]
    resolved, preset = resolve_run_config(cfg, dataset)
    return resume_config_hash(resolved, preset, task=get_bench_dataset_class(dataset).task)


def _resume_row(cfg: RunConfig, *, method: str, metric_name: str) -> dict[str, object]:
    ds_name = cfg.datasets[0]
    resolved, preset = resolve_run_config(cfg, ds_name)
    return {
        **dataset_metadata(
            resolved,
            ds_name,
            get_bench_dataset_class(ds_name),
            preset,
            resume_config_hash(resolved, preset, task=get_bench_dataset_class(ds_name).task),
        ),
        "method": method,
        "metric_name": metric_name,
        "metric_value": 0.1,
    }


def _chainable_model_mock() -> mock.Mock:
    model = mock.Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.effective_normalization = "bandspec_zscore"
    return model
