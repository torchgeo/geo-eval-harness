"""Numerical contracts for pixelwise regression probes."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.config.schema import SegmentationConfig
from torchgeo_bench.pixelwise_regression_task import (
    PixelwiseRegressionSolver,
    build_pixelwise_regression_probe_and_solver,
)
from torchgeo_bench.results import bootstrap_mean


class _FixedProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Identity()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :1] + self.bias


class _SpatialBackbone(nn.Module):
    num_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(1, 2, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layer(images)


def test_evaluation_averages_per_image_rmse_in_reported_units() -> None:
    images = torch.zeros(2, 1, 4, 4)
    targets = torch.stack([torch.ones(4, 4), torch.full((4, 4), 2.0)])
    solver = PixelwiseRegressionSolver(
        _FixedProbe(), target_scale=10.0, device="cpu", lr_scheduler="none"
    )

    metrics, per_image = solver.evaluate(DataLoader(TensorDataset(images, targets), batch_size=2))

    torch.testing.assert_close(per_image, torch.tensor([10.0, 20.0]))
    assert metrics["RMSE"] == pytest.approx(15.0)


def test_fit_trains_on_images_and_reports_validation_rmse() -> None:
    samples = [{"image": torch.zeros(1, 4, 4), "mask": torch.ones(4, 4)} for _ in range(2)]
    loader = DataLoader(samples, batch_size=2)
    solver = PixelwiseRegressionSolver(
        _FixedProbe(), target_scale=1.0, lr=0.1, device="cpu", lr_scheduler="cosine"
    )
    assert solver.optimizer.param_groups[0]["weight_decay"] == 0.0

    val_rmse = solver.fit(loader, val_loader=loader, epochs=2, verbose=True)

    assert val_rmse is not None
    assert val_rmse < 1.0
    assert solver.model.bias.item() > 0


def test_cached_targets_remain_float_for_regression() -> None:
    config = SegmentationConfig(layers=["layer"], head="linear", scheduler="cosine")
    probe, solver = build_pixelwise_regression_probe_and_solver(
        _SpatialBackbone(), config, torch.device("cpu"), target_scale=10.0
    )
    for parameter in probe.head.parameters():
        nn.init.zeros_(parameter)
    images = torch.zeros(2, 1, 8, 8)
    targets = torch.stack([torch.full((8, 8), 0.25), torch.full((8, 8), 1.75)])
    cache = probe.extract_segmentation_features(
        DataLoader(TensorDataset(images, targets), batch_size=2),
        cache_dtype=torch.float32,
        target_dtype=torch.float32,
    )

    assert cache.masks.dtype == torch.float32
    metrics, per_image = solver.evaluate_cached(cache, batch_size=2)
    torch.testing.assert_close(per_image, torch.tensor([2.5, 17.5]))
    assert metrics["RMSE"] == pytest.approx(10.0)

    val_rmse = solver.fit_cached(cache, val_cache=cache, batch_size=2, epochs=1, verbose=True)
    assert val_rmse is not None


def test_solver_rejects_invalid_configuration_and_shapes() -> None:
    solver = PixelwiseRegressionSolver(
        _FixedProbe(), target_scale=1.0, device="cpu", lr_scheduler="invalid"
    )
    with pytest.raises(ValueError, match="Unknown lr_scheduler"):
        solver._make_scheduler(1)
    solver.lr_scheduler_type = "none"
    assert solver._make_scheduler(1) is None
    with pytest.raises(ValueError, match="predictions shaped"):
        solver._prepare_outputs(torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match="does not match target shape"):
        solver._prepare_outputs(torch.zeros(1, 1, 4, 4), torch.zeros(1, 3, 3))
    predictions, targets = solver._prepare_outputs(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert predictions.shape == targets.shape == (1, 4, 4)


def test_probe_builder_requires_spatial_layers() -> None:
    with pytest.raises(ValueError, match=r"requires segmentation\.layers"):
        build_pixelwise_regression_probe_and_solver(
            _SpatialBackbone(), SegmentationConfig(layers=[]), torch.device("cpu"), 1.0
        )


def test_bootstrap_mean_resamples_held_out_images() -> None:
    lower, upper = bootstrap_mean(torch.tensor([10.0, 20.0]), n_boot=200, seed=7)
    assert (lower, upper) == pytest.approx((10.0, 20.0))


def test_bootstrap_mean_can_be_disabled() -> None:
    lower, upper = bootstrap_mean(torch.tensor([10.0]), n_boot=0)
    assert torch.isnan(torch.tensor([lower, upper])).all()


def test_bootstrap_mean_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_mean(torch.tensor([]))
