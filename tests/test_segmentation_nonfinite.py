"""Non-finite backbone features (float16 overflow) must not poison the segmentation head."""

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from torch.utils.data import DataLoader, TensorDataset

from geo_eval_harness.config.schema import SegmentationConfig
from geo_eval_harness.segmentation_probe import NonFiniteFeaturesError, SegmentationProbe
from geo_eval_harness.segmentation_task import SegmentationSolver, build_seg_probe_and_solver
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")

MARK = 1e3


class OverflowBackbone(nn.Module):
    """Emit ``inf`` features for images whose first pixel is marked, like a float16 overflow."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        marked = (x[:, 0, 0, 0] > MARK / 2).view(-1, 1, 1, 1)
        return self.out(torch.where(marked, torch.full_like(features, float("inf")), features))


def _data(n: int, marked: tuple[int, ...] = ()) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(n, 3, 8, 8)
    for index in marked:
        images[index, 0, 0, 0] = MARK
    return images, torch.randint(0, 2, (n, 8, 8))


def _loader(images: torch.Tensor, masks: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)


def _solver(*, feature_standardize: bool = False) -> SegmentationSolver:
    probe = SegmentationProbe(
        OverflowBackbone(),
        ["out"],
        num_classes=2,
        head_type="linear",
        feature_standardize=feature_standardize,
    )
    return SegmentationSolver(probe, num_classes=2, device="cpu")


def _batchnorm_buffers(solver: SegmentationSolver) -> list[torch.Tensor]:
    norms = [m for m in solver.model.head.modules() if isinstance(m, nn.BatchNorm2d)]
    assert norms
    return [buffer for m in norms for buffer in (m.running_mean, m.running_var)]


@pytest.mark.parametrize("path", ["images", "cached"])
def test_nonfinite_batch_is_skipped_before_the_head(path: str) -> None:
    solver = _solver()
    train = _loader(*_data(42, marked=(0,)))
    val = _loader(*_data(6))
    if path == "cached":
        probe = solver.model
        train_cache = probe.extract_segmentation_features(train, cache_dtype=torch.float32)
        val_cache = probe.extract_segmentation_features(val, cache_dtype=torch.float32)
        solver.fit_cached(train_cache, val_cache, batch_size=2, epochs=2, verbose=False)
    else:
        solver.fit(train, val, epochs=2, verbose=False)

    for record in solver.epoch_history:
        assert (record.nonfinite_batches, record.train_steps, record.skipped_batches) == (1, 20, 0)
        assert record.val_miou is not None
    assert all(bool(torch.isfinite(buffer).all()) for buffer in _batchnorm_buffers(solver))


@pytest.mark.parametrize("path", ["images", "cached"])
def test_too_many_nonfinite_batches_fail_the_run(path: str) -> None:
    solver = _solver()
    train = _loader(*_data(8, marked=(0, 2)))
    cache = solver.model.extract_segmentation_features(train, cache_dtype=torch.float32)

    def fit() -> None:
        if path == "cached":
            solver.fit_cached(cache, None, batch_size=2, epochs=1, verbose=False)
        else:
            solver.fit(train, None, epochs=1, verbose=False)

    with pytest.raises(NonFiniteFeaturesError, match=r"skipped [12] of 4 training batches"):
        fit()


@pytest.mark.parametrize("path", ["images", "cached"])
def test_standardizer_leaves_out_nonfinite_samples(path: str) -> None:
    solver = _solver(feature_standardize=True)
    probe = solver.model
    train = _loader(*_data(42, marked=(0,)))
    train_cache = probe.extract_segmentation_features(train, cache_dtype=torch.float32)
    probe.fit_standardizer(train_cache if path == "cached" else train)
    assert all(bool(torch.isfinite(buffer).all()) for buffer in probe.buffers())

    val_cache = probe.extract_segmentation_features(_loader(*_data(6)), cache_dtype=torch.float32)
    solver.fit_cached(train_cache, val_cache, batch_size=2, epochs=1, verbose=False)
    assert solver.epoch_history[0].nonfinite_batches == 1
    assert all(bool(torch.isfinite(buffer).all()) for buffer in probe.buffers())


@pytest.mark.parametrize("path", ["images", "cached"])
def test_standardizer_fails_on_too_many_nonfinite_samples(path: str) -> None:
    probe = _solver(feature_standardize=True).model
    train = _loader(*_data(8, marked=(0,)))
    data = (
        probe.extract_segmentation_features(train, cache_dtype=torch.float32)
        if path == "cached"
        else train
    )
    with pytest.raises(NonFiniteFeaturesError, match="left out 1 of 8 samples"):
        probe.fit_standardizer(data)


def test_evaluation_fails_on_nonfinite_features() -> None:
    solver = _solver()
    loader = _loader(*_data(4, marked=(3,)))
    with pytest.raises(NonFiniteFeaturesError):
        solver.evaluate(loader)
    cache = solver.model.extract_segmentation_features(loader, cache_dtype=torch.float32)
    with pytest.raises(NonFiniteFeaturesError):
        solver.evaluate_cached(cache, batch_size=2)


def test_evaluation_fails_on_poisoned_batchnorm_statistics() -> None:
    solver = _solver()
    _batchnorm_buffers(solver)[0].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="logits are not finite"):
        solver.evaluate(_loader(*_data(4)))


def test_amp_defaults_to_fp16_and_reaches_probe_and_solver() -> None:
    assert SegmentationConfig().amp == "fp16"
    for amp, enabled in (("fp16", True), ("off", False)):
        config = SegmentationConfig(layers=["out"], head="linear", amp=amp, cache_dtype="float32")
        probe, solver = build_seg_probe_and_solver(
            OverflowBackbone(), 2, config, torch.device("cpu")
        )
        assert (probe.amp, solver.amp, solver.use_amp) == (enabled, enabled, False)


def test_amp_off_requires_a_float32_feature_cache() -> None:
    with pytest.raises(ValidationError, match="requires cache_dtype 'float32'"):
        SegmentationConfig(amp="off")
    SegmentationConfig(amp="off", cache_dtype="float32")
    SegmentationConfig(amp="off", cache_features=False)
    SegmentationConfig(amp="off", full_finetune=True, backbone_learning_rate=1e-5)
    with pytest.raises(ValidationError, match="Input should be"):
        SegmentationConfig(amp="bf16")
