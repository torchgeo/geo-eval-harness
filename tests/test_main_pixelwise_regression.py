"""Offline tests for the pixelwise regression runner."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tests.support.runner import _chainable_model_mock, _compose_cfg, _resume_row
from torchgeo_bench.config.presets import merge_settings
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.main import main, run_dense_task


class _RegressionDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.images = torch.zeros(size, 3, 8, 8)
        self.targets = torch.zeros(size, 8, 8)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "mask": self.targets[index]}


def _loaders() -> tuple[_RegressionDataset, DataLoader, DataLoader, DataLoader]:
    datasets = [_RegressionDataset(size) for size in (8, 4, 4)]
    loaders = [DataLoader(dataset, batch_size=2) for dataset in datasets]
    return datasets[0], *loaders


def _config(output: Path, overrides: dict | None = None) -> RunConfig:
    return _compose_cfg(
        output,
        overrides=merge_settings(
            {
                "datasets": ["biomassters"],
                "segmentation": {"cache_features": True, "head": "fpn"},
            },
            overrides or {},
        ),
    )


def test_cached_regression_records_physical_rmse(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    cfg = _config(output)
    probe = mock.Mock()
    probe.channels_list = [16, 32]
    probe.freeze_backbone = True
    cache = mock.Mock()
    probe.extract_segmentation_features.return_value = cache
    solver = mock.Mock()
    solver.evaluate_cached.return_value = ({"RMSE": 30.0}, torch.tensor([20.0, 40.0]))

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_loaders()),
        mock.patch(
            "torchgeo_bench.main.instantiate_dataset_model",
            return_value=_chainable_model_mock(),
        ),
        mock.patch(
            "torchgeo_bench.pixelwise_regression_task.build_pixelwise_regression_probe_and_solver",
            return_value=(probe, solver),
        ) as build,
    ):
        main(cfg)

    assert build.call_args.args[-1] == 289.89
    assert probe.extract_segmentation_features.call_count == 3
    for call in probe.extract_segmentation_features.call_args_list:
        assert call.kwargs["target_dtype"] is torch.float32
    solver.fit_cached.assert_called_once_with(
        train_cache=cache,
        val_cache=cache,
        batch_size=cfg.segmentation.batch_size,
        epochs=cfg.segmentation.epochs,
        verbose=cfg.runtime.verbose,
    )
    result = pd.read_csv(output).iloc[0]
    assert result["method"] == "reg-fpn"
    assert result["metric_name"] == "RMSE"
    assert result["metric_value"] == 30.0


def test_uncached_regression_trains_from_images(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    cfg = _config(output, {"segmentation": {"cache_features": False}})
    probe = mock.Mock(channels_list=[16, 32], freeze_backbone=True)
    solver = mock.Mock()
    solver.evaluate.return_value = ({"RMSE": 30.0}, torch.tensor([20.0, 40.0]))

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_loaders()),
        mock.patch(
            "torchgeo_bench.main.instantiate_dataset_model",
            return_value=_chainable_model_mock(),
        ),
        mock.patch(
            "torchgeo_bench.pixelwise_regression_task.build_pixelwise_regression_probe_and_solver",
            return_value=(probe, solver),
        ),
    ):
        main(cfg)

    solver.fit.assert_called_once()
    solver.evaluate.assert_called_once()
    assert pd.read_csv(output).iloc[0]["best_batch_size"] == 2


def test_dense_task_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unsupported dense task"):
        list(run_dense_task("unknown", mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), 1.0))


def test_regression_resume_skips_complete_run(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    cfg = _config(output, {"output": {"resume": True}})
    pd.DataFrame([_resume_row(cfg, method="reg-fpn", metric_name="RMSE")]).to_csv(
        output, index=False
    )

    with mock.patch("torchgeo_bench.main.get_datasets") as get_datasets:
        main(cfg)

    get_datasets.assert_not_called()
