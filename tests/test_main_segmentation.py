"""Offline tests for the segmentation runner."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tests.support.runner import _chainable_model_mock, _compose_cfg, _resume_row
from torchgeo_bench.config.presets import merge_settings
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.main import LoaderSplits, evaluate_segmentation, main, run_dataset
from torchgeo_bench.results import bootstrap_miou
from torchgeo_bench.resume import ResumeState
from torchgeo_bench.segmentation_probe import SegmentationProbe
from torchgeo_bench.segmentation_task import SegmentationSolver


class _SegmentationDataset(Dataset):
    def __init__(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        self._images = images
        self._masks = masks

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[index], "mask": self._masks[index]}


def _synthetic_segmentation_loaders(
    n_train: int = 8,
    n_val: int = 4,
    n_test: int = 4,
    channels: int = 3,
    n_classes: int = 3,
) -> tuple[_SegmentationDataset, DataLoader, DataLoader, DataLoader]:
    rng = torch.Generator().manual_seed(1)
    train_images = torch.rand(n_train, channels, 8, 8, generator=rng)
    val_images = torch.rand(n_val, channels, 8, 8, generator=rng)
    test_images = torch.rand(n_test, channels, 8, 8, generator=rng)

    train_masks = torch.randint(0, n_classes, (n_train, 8, 8), generator=rng)
    val_masks = torch.randint(0, n_classes, (n_val, 8, 8), generator=rng)
    test_masks = torch.randint(0, n_classes, (n_test, 8, 8), generator=rng)

    train_dataset = _SegmentationDataset(train_images, train_masks)
    val_dataset = _SegmentationDataset(val_images, val_masks)
    test_dataset = _SegmentationDataset(test_images, test_masks)

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, generator=rng, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_loader, test_loader


def _cfg_for_segmentation(out: Path, overrides: dict | None = None) -> RunConfig:
    return _compose_cfg(
        out,
        overrides=merge_settings(
            {
                "datasets": ["burn_scars"],
                "segmentation": {"cache_features": False, "head": "fpn"},
            },
            overrides or {},
        ),
    )


def _mock_probe_and_solver():
    probe = mock.Mock()
    probe.channels_list = [16, 32]
    solver = mock.Mock()
    return probe, solver


def test_dataset_eval_resolution_preserves_explicit_values_and_original_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg_for_segmentation(tmp_path / "out.csv")
    cfg.segmentation.learning_rate = 0.002
    cfg.segmentation.epochs = 3
    cfg.segmentation.batch_size = 6
    original = cfg.model_dump()
    captured: list[RunConfig] = []

    def capture_eval(config: RunConfig, *_args: object) -> list[dict]:
        captured.append(config)
        return []

    loaders = _synthetic_segmentation_loaders()
    monkeypatch.setattr("torchgeo_bench.main.get_datasets", lambda **_kwargs: loaders)
    monkeypatch.setattr(
        "torchgeo_bench.main.instantiate_dataset_model", lambda *_args: _chainable_model_mock()
    )
    monkeypatch.setattr("torchgeo_bench.main.run_segmentation", capture_eval)

    assert list(run_dataset(cfg, "burn_scars", ResumeState(set(), {}))) == []

    assert captured[0].segmentation.learning_rate == 0.002
    assert captured[0].segmentation.epochs == 3
    assert captured[0].segmentation.batch_size == 6
    assert cfg.model_dump() == original


def test_cached_segmentation_records_probe_batch_size(tmp_path: Path):
    """Cached probes use their own configured batch size, not loader batch size."""
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(
        out,
        overrides={"segmentation": {"cache_features": True, "batch_size": 3}},
    )
    probe, solver = _mock_probe_and_solver()
    cache = mock.Mock()
    probe.freeze_backbone = True
    probe.extract_segmentation_features.return_value = cache
    solver.evaluate_cached.return_value = (
        {
            "mIoU": 0.42,
            "fw_IoU": 0.55,
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
        },
        torch.tensor([[[0, 4], [0, 0]], [[0, 0], [0, 4]]]),
    )

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
            return_value=(probe, solver),
        ),
    ):
        main(cfg)

    solver.fit_cached.assert_called_once_with(
        train_cache=cache,
        val_cache=cache,
        batch_size=3,
        epochs=cfg.segmentation.epochs,
        verbose=cfg.runtime.verbose,
    )
    df = pd.read_csv(out)
    assert df.loc[0, "best_batch_size"] == 3


def test_segmentation_resume_skips_complete_run(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides={"output": {"resume": True}})
    pd.DataFrame([_resume_row(cfg, method="seg-fpn", metric_name="mIoU")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch("torchgeo_bench.main.get_datasets") as data_mock,
        mock.patch("torchgeo_bench.main.build_model", return_value=model) as instantiate_mock,
        mock.patch("torchgeo_bench.segmentation_task.build_seg_probe_and_solver") as build_mock,
    ):
        main(cfg)

    data_mock.assert_not_called()
    instantiate_mock.assert_not_called()
    build_mock.assert_not_called()
    df = pd.read_csv(out)
    assert int((df["method"] == "seg-fpn").sum()) == 1


class _ChangingHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.5))
        self.register_buffer("batches", torch.tensor(0))

    def forward(self, features: list[torch.Tensor], height: int, width: int) -> torch.Tensor:
        if self.training:
            self.batches.add_(1)
        score = self.bias.expand(len(features[0]), height, width)
        return torch.stack((score, -score), dim=1)


@pytest.mark.parametrize("cached", [False, True])
def test_runner_test_metrics_and_bootstrap_use_selected_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> None:
    cfg = _cfg_for_segmentation(
        tmp_path / "out.csv",
        {"segmentation": {"cache_features": cached, "epochs": 3, "batch_size": 2}},
    )
    images = torch.ones(2, 3, 8, 8)
    train = DataLoader(_SegmentationDataset(images, torch.ones(2, 8, 8).long()), batch_size=2)
    test = DataLoader(_SegmentationDataset(images, torch.zeros(2, 8, 8).long()), batch_size=2)
    probe = SegmentationProbe(
        torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1)), ["0"], 2, head_type="linear"
    )
    probe.head = _ChangingHead()
    solver = SegmentationSolver(probe, 2, lr=1.0, lr_scheduler="none", device="cpu")
    evaluate = solver._evaluate
    test_data = (
        probe.extract_segmentation_features(test, cache_dtype=torch.float32) if cached else test
    )
    if cached:
        from torchgeo_bench.segmentation_probe import GPUTensorCache

        test_data = GPUTensorCache.from_cached(test_data, "cpu")
    boundaries: list[tuple[dict[str, float], torch.Tensor]] = []

    def controlled_validation(data, batch_size=64, *, collect_confusions=False):
        if collect_confusions:
            return evaluate(data, batch_size, collect_confusions=True)
        result = evaluate(test_data, collect_confusions=True)
        assert isinstance(result, tuple)
        boundaries.append(result)
        return {"mIoU": [0.9, 0.5, 0.2][len(boundaries) - 1]}

    monkeypatch.setattr(solver, "_evaluate", controlled_validation)
    monkeypatch.setattr(
        "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
        lambda *_args: (probe, solver),
    )
    with mock.patch("torchgeo_bench.main.bootstrap_miou", wraps=bootstrap_miou) as bootstrap:
        metrics, *_ = evaluate_segmentation(probe.backbone, LoaderSplits(train, test, test), cfg, 2)

    assert len(boundaries) == 3
    assert boundaries[0][0]["mIoU"] == 1.0
    assert boundaries[-1][0]["mIoU"] == 0.0
    for name, value in boundaries[0][0].items():
        assert metrics[name] == value
    torch.testing.assert_close(bootstrap.call_args.args[0], boundaries[0][1], rtol=0, atol=0)
    assert (metrics["ci_lower"], metrics["ci_upper"]) == bootstrap_miou(
        boundaries[0][1], n_boot=cfg.classification.bootstrap_samples, seed=cfg.runtime.seed
    )
    assert solver.best_epoch == 1
    assert probe.head.batches.item() == 1


@pytest.mark.parametrize("cached", [False, True])
def test_runner_invalid_validation_does_not_write_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> None:
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(
        out,
        {
            "segmentation": {
                "cache_features": cached,
                "epochs": 1,
                "layers": ["0"],
                "head": "linear",
            }
        },
    )
    monkeypatch.setattr(
        "torchgeo_bench.main.get_datasets", lambda **_kwargs: _synthetic_segmentation_loaders()
    )
    backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))
    backbone.effective_normalization = "bandspec_zscore"
    monkeypatch.setattr(
        "torchgeo_bench.main.instantiate_dataset_model",
        lambda *_args: backbone,
    )
    monkeypatch.setattr(
        SegmentationSolver, "_compute_metrics", lambda _self: {"mIoU": float("nan")}
    )
    with pytest.raises(ValueError, match="must be finite"):
        main(cfg)
    assert not out.exists()
