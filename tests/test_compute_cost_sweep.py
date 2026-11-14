"""Catalog coverage and partial-failure recovery in the compute sweep."""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import torch
import yaml

from experiments import run_compute_cost_sweep as sweep
from geo_eval_harness.config.presets import build_model
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.models.terrafm import TerraFMBench


def options(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=tmp_path,
        device="cpu",
        num_classes=4,
        batch_size=1,
        warmup=0,
        measurements=1,
        cell_timeout=30,
        olmoearth_v13_python=None,
        model_kwargs={},
    )


def test_catalog_covers_new_models_and_declared_sar_input() -> None:
    jobs = sweep.build_jobs(
        [
            "timm/convnext_tiny_dinov3",
            "timm/vit/vit_large_patch16_dinov3",
            "terratorch/terramind_v1_base_s1",
        ],
        [224, 512],
        ["linear", "upernet", "unet"],
    )
    assert len(jobs) == 10
    assert {job.size for job in jobs} == {224, 512}
    assert all(job.heads == ("linear", "upernet", "unet") for job in jobs)
    sar = [job for job in jobs if job.preset.endswith("_s1")]
    assert all((job.source, job.bands) == ("kuro_siwo", "all") for job in sar)


def test_sweep_changes_wrapper_resolution_without_changing_encoder_patch_stride(
    tmp_path: Path,
) -> None:
    job = sweep.Job("torchgeo/dofa_base", 512, "all", "cloudsen12", ("unet",))
    args = options(tmp_path)
    args.model_kwargs = {"torchgeo/dofa_base": {"checkpoint_path": "/weights/dofa"}}
    config = sweep.job_config(job, args)
    effective, preset = config.resolve()
    assert effective.input.image_size == 512
    assert preset.kwargs["target_size"] == 512
    assert preset.kwargs["img_size"] == 512
    assert "patch_size" not in config.model.kwargs
    assert preset.kwargs["checkpoint_path"] == "/weights/dofa"


def test_failed_decoder_does_not_prevent_later_heads(tmp_path: Path, monkeypatch) -> None:
    for directory in ("configs", "logs"):
        (tmp_path / directory).mkdir()
    job = sweep.Job("timm/resnet50", 32, "rgb", "cloudsen12", ("linear", "upernet", "unet"))
    monkeypatch.setattr(sweep, "measurement_metadata", lambda *args: {"config_hash": "measured"})
    monkeypatch.setattr(sweep, "resolve_device", lambda _: torch.device("cpu"))
    calls = []

    def run(command, **kwargs):
        config = yaml.safe_load(Path(command[-1]).read_text())
        heads = config["segmentation"]["heads"]
        calls.append(heads)
        completed = [("classification", ""), ("segmentation", "linear")] if len(heads) > 1 else []
        if heads == ["unet"]:
            completed = [("segmentation", "unet")]
        path = tmp_path / "compute_cost.csv"
        exists = path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "config_hash",
                    "task",
                    "head_type",
                    "gflops_total",
                    "throughput_samples_per_sec",
                ],
            )
            if not exists:
                writer.writeheader()
            for task, head in completed:
                writer.writerow(
                    {
                        "config_hash": "measured",
                        "task": task,
                        "head_type": head,
                        "gflops_total": 1,
                        "throughput_samples_per_sec": 10,
                    }
                )
        return subprocess.CompletedProcess(command, 0 if heads == ["unet"] else 1)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    report = sweep.run_job(job, options(tmp_path), time.monotonic() + 60)
    assert calls == [["linear", "upernet", "unet"], ["upernet"], ["unet"]]
    assert report["status"] == "incomplete"
    assert report["missing"] == [("segmentation", "upernet")]
    assert ("segmentation", "unet") in report["measured"]


def test_incomplete_and_nonfinite_metrics_are_not_completed(tmp_path: Path) -> None:
    path = tmp_path / "compute_cost.csv"
    path.write_text(
        "config_hash,task,head_type,gflops_total,throughput_samples_per_sec\n"
        "same,segmentation,linear,1,10\n"
        "same,segmentation,unet,NaN,10\n"
        "same,segmentation,upernet,1,inf\n"
        "same,classification,,1,0\n"
        "other,segmentation,fpn,1,10\n"
    )
    assert sweep.measured_cells(path, "same") == {("segmentation", "linear")}


def test_sweep_overrides_inferred_resize_and_fixed_encoder_grid(tmp_path: Path) -> None:
    args = options(tmp_path)
    cnn = sweep.job_config(sweep.Job("timm/efficientnet_b3", 512, "rgb", "cloudsen12", ()), args)
    vit = sweep.job_config(
        sweep.Job("timm/vit/vit_base_patch16_224", 512, "rgb", "cloudsen12", ()), args
    )
    croma = sweep.job_config(sweep.Job("torchgeo/croma_base", 512, "all", "cloudsen12", ()), args)
    assert cnn.resolve()[1].kwargs["target_size"] == 512
    assert "img_size" not in cnn.model.kwargs
    assert vit.resolve()[1].kwargs["img_size"] == 512
    assert croma.resolve()[1].kwargs["factory_kwargs"]["image_size"] == 512
    swin = sweep.job_config(
        sweep.Job("timm/vit/swinv2_large_window12to16_192to256", 224, "rgb", "cloudsen12", ()),
        args,
    )
    assert swin.resolve()[1].kwargs["dynamic_img_size"] is True


def test_synthetic_sweep_uses_native_sensor_gsd_instead_of_fixed_tile_resize(
    tmp_path: Path,
) -> None:
    args = options(tmp_path)
    anysat = sweep.job_config(
        sweep.Job("anysat_base_seg", 512, "all", "cloudsen12", ("linear",)), args
    )
    universat = sweep.job_config(
        sweep.Job("universat_base_seg", 512, "all", "cloudsen12", ("linear",)), args
    )
    assert anysat.resolve()[1].kwargs["input_gsd"] is None
    assert anysat.resolve()[1].kwargs["patch_size"] == 80
    assert universat.resolve()[1].kwargs["input_res"] is None
    assert universat.resolve()[1].kwargs["patch_px"] == 4
    thor = sweep.job_config(sweep.Job("thor/base", 256, "all", "cloudsen12", ("linear",)), args)
    assert thor.resolve()[1].kwargs["native_size"] == 256


def test_sweep_labels_panopticon_with_its_actual_patch_aligned_image_sizes(tmp_path: Path) -> None:
    jobs = sweep.build_jobs(["torchgeo/panopticon"], [224, 256, 512], ["linear", "upernet", "unet"])
    assert {job.size for job in jobs} == {224, 252, 518}
    for job in jobs:
        config = sweep.job_config(job, options(tmp_path))
        assert config.input.image_size == job.size
        assert config.resolve()[1].kwargs["img_size"] == job.size
        assert config.resolve()[1].kwargs["target_size"] == job.size


def test_sweep_uses_earthloc_native_grid_once_without_resampling_trained_aggregation() -> None:
    jobs = sweep.build_jobs(
        ["torchgeo/earthloc_s2_resnet50"], [224, 256, 512], ["linear", "upernet", "unet"]
    )
    assert len(jobs) == 2
    assert {job.size for job in jobs} == {320}
    assert {job.bands for job in jobs} == {"rgb", "all"}


def test_terrafm_keeps_checkpoint_positions_and_interpolates_larger_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(TerraFMBench, "_VARIANTS", {"base": (32, 1, 4)})
    bands = get_bench_dataset_class("cloudsen12")().select_band_specs(None)
    checkpoint = tmp_path / "weights.pth"
    with torch.random.fork_rng(devices=[]):
        native = TerraFMBench(bands=bands, pretrained=False)
        torch.save(native.state_dict(), checkpoint)
        args = options(tmp_path)
        args.model_kwargs = {"terrafm/base_s2": {"checkpoint_path": str(checkpoint)}}
        config = sweep.job_config(sweep.Job("terrafm/base_s2", 512, "all", "cloudsen12", ()), args)
        _, preset = config.resolve()
        model = build_model(preset, bands=bands, normalization="identity").eval()
        with torch.inference_mode():
            tokens = model.forward_tokens(torch.randn(1, 12, 512, 512))
    assert config.input.image_size == 512
    assert model.auto_resize is False
    assert tokens.shape == (1, 1025, 32)
    torch.testing.assert_close(model.pos_embed, native.pos_embed, rtol=0, atol=0)
