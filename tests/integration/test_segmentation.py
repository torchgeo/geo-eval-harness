"""Learn a known segmentation target through the CLI on a tiny GeoBench V2 Taco."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from geo_eval_harness.cli import main
from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class, get_datasets
from geo_eval_harness.segmentation_task import build_seg_probe_and_solver
from tests.support.cli import cli_output, run_cli
from tests.support.data import write_caffe_files
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("isolated_torch_rng")]


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "uncached"])
def test_public_segmentation_with_random_weight_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> None:
    write_caffe_files(tmp_path, target_class=2)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "public-segmentation.csv"
    arguments = _segmentation_arguments(output, cached=cached)
    with torch.random.fork_rng(devices=[]):
        main(arguments)
        _assert_segmentation_results(output, cached=cached)
        before = output.read_bytes()
        shutil.rmtree(tmp_path / "data")
        main([*arguments, "--resume"])
    assert output.read_bytes() == before


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "uncached"])
def test_module_segmentation_training_and_resume(tmp_path: Path, *, cached: bool) -> None:
    write_caffe_files(tmp_path, target_class=2)
    output = tmp_path / "segmentation.csv"
    arguments = _segmentation_arguments(output, cached=cached)
    result = run_cli(*arguments, cwd=tmp_path)
    assert result.returncode == 0, cli_output(result)
    assert output.is_file(), cli_output(result)
    _assert_segmentation_results(output, cached=cached)
    logs = result.stdout + result.stderr
    assert ("Cached features for" in logs) is cached
    assert "Epoch 1 Val mIoU:" in logs
    assert "Epoch 10 Val mIoU:" in logs

    before = output.read_bytes()
    shutil.rmtree(tmp_path / "data")
    resumed = run_cli(*arguments, "--resume", cwd=tmp_path)
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before


def _segmentation_config(output: Path, *, cached: bool) -> RunConfig:
    return RunConfig.model_validate(
        {
            "model": {"name": "timm/resnet18", "kwargs": {"pretrained": False}},
            "datasets": ["caffe"],
            "input": {"image_size": 16},
            "runtime": {
                "device": "cpu",
                "batch_size": 4,
                "workers": 0,
                "seed": 7,
                "verbose": True,
            },
            "classification": {"bootstrap_samples": 5},
            "segmentation": {
                "head": "linear",
                "layers": ["layer1"],
                "epochs": 10,
                "learning_rate": 0.05,
                "scheduler": "none",
                "batch_size": 2,
                "cache_features": cached,
                "cache_dtype": "float32",
            },
            "output": {"file": str(output)},
        }
    )


def _segmentation_arguments(output: Path, *, cached: bool) -> list[str]:
    config = _segmentation_config(output, cached=cached)
    path = output.with_suffix(".yaml")
    path.write_text(yaml.safe_dump(config.model_dump_yaml()))
    return ["run", "--config", str(path)]


def test_real_solver_learns_target_with_consistent_cached_and_uncached_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check learned predictions and parameter updates, not merely executable epochs."""
    write_caffe_files(tmp_path, target_class=2)
    monkeypatch.chdir(tmp_path)
    _, train, val, test = get_datasets(
        "caffe", batch_size=6, num_workers=0, image_size=16, bands="rgb", return_val=True
    )
    final_heads = []
    for cached in (True, False):
        torch.manual_seed(7)
        cfg, preset = resolve_run_config(
            _segmentation_config(tmp_path / "unused.csv", cached=cached), "caffe"
        )
        model = build_model(
            preset,
            bands=get_bench_dataset_class("caffe").bands,
            normalization=NORMALIZATIONS[cfg.input.normalization],
        )
        probe, solver = build_seg_probe_and_solver(model, 4, cfg.segmentation, torch.device("cpu"))
        backbone_before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        head_before = {name: tensor.clone() for name, tensor in probe.head.named_parameters()}
        assert solver.evaluate(test)["mIoU"] < 1
        batch = next(iter(test))
        with torch.no_grad():
            loss_before = solver.criterion(probe(batch["image"]), batch["mask"]).item()
        if cached:
            caches = [
                probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
                for loader in (train, val, test)
            ]
            solver.fit_cached(caches[0], caches[1], batch_size=6, epochs=10, verbose=False)
            metrics = solver.evaluate_cached(caches[2], batch_size=6)
        else:
            solver.fit(train, val, epochs=10, verbose=False)
            metrics = solver.evaluate(test)
        assert metrics["mIoU"] == 1
        assert len(solver.val_history) == 10
        with torch.no_grad():
            logits = probe(batch["image"])
            loss_after = solver.criterion(logits, batch["mask"]).item()
        assert loss_after < loss_before
        valid = batch["mask"] != 255
        torch.testing.assert_close(logits.argmax(dim=1)[valid], batch["mask"][valid])
        assert any(
            not torch.equal(head_before[name], parameter)
            for name, parameter in probe.head.named_parameters()
        )
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, backbone_before[name], rtol=0, atol=0)
        final_heads.append(
            {name: parameter.detach().clone() for name, parameter in probe.head.named_parameters()}
        )
    # A full training batch makes sample-shuffle order irrelevant to the intended update.
    for name, parameter in final_heads[0].items():
        torch.testing.assert_close(parameter, final_heads[1][name], rtol=1e-5, atol=1e-6)


def _assert_segmentation_results(output: Path, *, cached: bool) -> None:
    rows = pd.read_csv(output)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["dataset"] == "caffe"
    assert row["method"] == "seg-linear"
    assert row["metric_name"] == "mIoU"
    assert row["num_classes"] == 4
    assert row["n_train"] == 6
    assert row["n_val"] == 4
    assert row["n_test"] == 4
    assert row["metric_value"] == 1
    assert row["boundary_iou"] == 1
    assert row["boundary_iou_width"] == 1
    assert np.isfinite(row[["metric_value", "ci_lower", "ci_upper"]].astype(float)).all()
    assert row["ci_lower"] == row["ci_upper"] == 1
    assert row["feature_dim"] == 64
    assert row["best_lr"] == 0.05
    assert row["best_batch_size"] == (2 if cached else 4)
