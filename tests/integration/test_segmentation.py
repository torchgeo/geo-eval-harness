"""Learn a known segmentation target through the CLI on a tiny GeoBench V2 Taco."""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.datasets import get_bench_dataset_class, get_datasets
from geo_eval_harness.segmentation_task import build_seg_probe_and_solver
from tests.support.data import (
    write_bright_post_sar_files,
    write_caffe_files,
    write_levir_cd_post_files,
    write_loveda_files,
    write_openearthmap_sar_files,
    write_oscd_post_files,
    write_sen1floods11_files,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("isolated_torch_rng")]


# Each entry: size-limiting fixture, file writer, input settings, feature caching, and the
# expected (num_classes, n_train, n_val, n_test).
# fmt: off
TOY_DATASETS = {
    "openearthmap-sar": ("small_openearthmap_sar", write_openearthmap_sar_files, {"bands": ["sar"]}, True, (8, 3, 2, 2)),
    "loveda": ("small_loveda", write_loveda_files, {"bands": ["red", "green", "blue"]}, True, (7, 2, 1, 2)),
    "levir-cd-post": ("small_levir_cd_post", write_levir_cd_post_files, {"image_size": 8}, False, (2, 12, 8, 8)),
    "bright-post-sar": ("small_bright_post_sar", write_bright_post_sar_files, {"bands": "rgb"}, True, (4, 3, 2, 2)),
    "bright-post-sar-rgb": ("small_bright_post_sar", write_bright_post_sar_files, {"bands": "rgb"}, True, (4, 3, 2, 2)),
    "oscd-post": ("small_oscd_post", write_oscd_post_files, {"bands": "all", "image_size": 8}, True, (2, 7, 4, 4)),
    # One Sen1Floods11 val chip is entirely ignored; metrics must stay finite.
    "sen1floods11-s1": ("small_sen1floods11", write_sen1floods11_files, {"bands": "all"}, True, (2, 3, 2, 2)),
    "sen1floods11-s2": ("small_sen1floods11", write_sen1floods11_files, {"bands": "all"}, True, (2, 3, 2, 2)),
}
# fmt: on


class TestToyDatasetSegmentation:
    @pytest.mark.parametrize("dataset", TOY_DATASETS)
    def test_run_writes_one_row(
        self,
        request: pytest.FixtureRequest,
        tmp_path: Path,
        cli_main: Callable[..., None],
        dataset: str,
    ) -> None:
        fixture, write_files, input_settings, cached, expected = TOY_DATASETS[dataset]
        request.getfixturevalue(fixture)
        write_files(tmp_path)
        output = tmp_path / "rows.csv"
        config = _segmentation_config(output, cached=cached)
        config.datasets = [dataset]
        config.segmentation.epochs = 1
        for name, value in input_settings.items():
            setattr(config.input, name, value)
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(config.model_dump_yaml()))
        cli_main(["run", "--config", str(path)])
        (row,) = pd.read_csv(output).to_dict("records")
        assert row["dataset"] == dataset
        assert row["method"] == "seg-linear"
        assert row["metric_name"] == "mIoU"
        assert (row["num_classes"], row["n_train"], row["n_val"], row["n_test"]) == expected
        assert np.isfinite([row["metric_value"], row["ci_lower"], row["ci_upper"]]).all()


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "uncached"])
def test_public_segmentation_with_random_weight_preset(
    tmp_path: Path,
    cli_main: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
    *,
    cached: bool,
) -> None:
    write_caffe_files(tmp_path, target_class=2)
    output = tmp_path / "public-segmentation.csv"
    arguments = _segmentation_arguments(output, cached=cached)
    with caplog.at_level(logging.INFO, logger="geo_eval_harness"):
        cli_main(arguments)
    _assert_segmentation_results(output, cached=cached)
    assert ("Cached features for" in caplog.text) is cached
    assert "Epoch 1 Val mIoU:" in caplog.text
    assert "Epoch 10 Val mIoU:" in caplog.text
    before = output.read_bytes()
    shutil.rmtree(tmp_path / "data")
    cli_main([*arguments, "--resume"])
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


def test_caffe_serves_gray_as_identical_rgb_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_caffe_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    train, loader, *_ = get_datasets(
        "caffe", num_workers=0, return_val=True, image_size=12, bands="rgb"
    )
    image = next(iter(loader))["image"]
    assert image.shape[1] == 3
    torch.testing.assert_close(image[:, 1], image[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(image[:, 2], image[:, 0], rtol=0, atol=0)
    specs = get_bench_dataset_class("caffe")().resolve_band_specs("rgb")
    assert [(b.name, b.sensor, b.wavelength_um) for b in specs] == [
        ("red", "aerial", None),
        ("green", "aerial", None),
        ("blue", "aerial", None),
    ]
    single, *_ = get_datasets("caffe", num_workers=0, return_val=True, image_size=12, bands=["red"])
    torch.testing.assert_close(single[0]["image"], train[0]["image"][:1], rtol=0, atol=0)
