"""Explicit FLOPs flags, lightweight validation, and synthetic CPU execution."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from geo_eval_harness import commands
from geo_eval_harness.cli import main as cli_main
from geo_eval_harness.commands import _flops
from geo_eval_harness.commands.flops_arguments import add_flops_arguments
from geo_eval_harness.config.flops import FlopsConfig


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    add_flops_arguments(result)
    return result


def test_explicit_flags_override_yaml_without_overriding_omitted_values(tmp_path: Path) -> None:
    path = tmp_path / "flops.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "rcf", "kwargs": {"features": 12, "kernel_size": 5}},
                "runtime": {"device": "cuda", "seed": 11, "verbose": True},
                "input": {"image_size": 112, "band_configs": ["s2"]},
                "segmentation": {"probe": {"layers": ["stem"]}},
                "timing": {"batch_size": 7, "n_measure": 2},
                "output": {"resume": True},
            }
        )
    )
    config = _flops.load_config(
        parser().parse_args(
            [
                "--config",
                str(path),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--model-kwargs",
                "{features: 8}",
                "--no-resume",
                "--no-verbose",
                "--seg-layers",
                "--n-warmup",
                "0",
            ]
        )
    )
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 0
    assert not config.runtime.verbose
    assert config.model.kwargs == {"features": 8, "kernel_size": 5}
    assert config.input.band_configs == ["s2"]
    assert config.input.image_size == 112
    assert config.segmentation.probe.layers == []
    assert config.timing.batch_size == 7
    assert config.timing.n_measure == 2
    assert config.timing.n_warmup == 0
    assert not config.output.resume


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(
            {"name": "custom-stats", "target": "geo_eval_harness.models.ImageStatsBench"},
            id="custom-target-to-preset",
        ),
        pytest.param(
            {
                "name": "timm/resnet18",
                "kwargs": {"pretrained": True, "model_name": "resnet18"},
            },
            id="incompatible-preset-kwargs",
        ),
        pytest.param(
            {"name": "rcf", "kwargs": {"features": 24, "mode": "empirical"}},
            id="same-preset-is-still-a-new-selection",
        ),
    ],
)
def test_model_flag_replaces_entire_yaml_selection(
    tmp_path: Path, selection: dict[str, Any]
) -> None:
    import torch

    from geo_eval_harness.config.presets import build_model
    from geo_eval_harness.datasets.cloudsen12 import CloudSEN12
    from geo_eval_harness.models import RCFBench

    path = tmp_path / "flops.yaml"
    path.write_text(
        yaml.safe_dump(
            {"model": selection, "runtime": {"device": "cpu"}, "input": {"image_size": 8}}
        )
    )
    config = _flops.load_config(
        parser().parse_args(
            ["--config", str(path), "--model", "rcf", "--model-kwargs", "{features: 8}"]
        )
    )
    resolved, preset = config.resolve()
    model = build_model(preset, bands=CloudSEN12.bands).to(resolved.runtime.device).eval()
    assert isinstance(model, RCFBench)
    assert resolved.model.name == "rcf"
    assert resolved.model.target is None
    assert resolved.model.kwargs == {"features": 8}
    images = torch.zeros(
        2,
        len(CloudSEN12.bands),
        resolved.input.image_size,
        resolved.input.image_size,
        device=resolved.runtime.device,
    )
    with torch.inference_mode():
        features = model(images)
    assert features.shape == (2, 8)
    assert features.device.type == "cpu"


def test_explicit_constructor_flags_win_after_model_switch(tmp_path: Path) -> None:
    import torch

    from geo_eval_harness.config.presets import build_model
    from geo_eval_harness.datasets.cloudsen12 import CloudSEN12
    from geo_eval_harness.models import RCFBench

    path = tmp_path / "flops.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "timm/resnet18",
                    "kwargs": {"pretrained": True, "model_name": "resnet18"},
                },
                "runtime": {"device": "cpu"},
                "input": {"image_size": 8},
            }
        )
    )
    config = _flops.load_config(
        parser().parse_args(
            [
                "--config",
                str(path),
                "--model",
                "imagestats",
                "--model-target",
                "geo_eval_harness.models.RCFBench",
                "--model-kwargs",
                "{features: 8, stats_mode: stdev}",
            ]
        )
    )
    resolved, preset = config.resolve()
    model = build_model(preset, bands=CloudSEN12.bands).to(resolved.runtime.device).eval()
    assert isinstance(model, RCFBench)
    assert resolved.model.name == "imagestats"
    assert resolved.model.target == "geo_eval_harness.models.RCFBench"
    assert resolved.model.kwargs == {"features": 8, "stats_mode": "stdev"}
    images = torch.zeros(
        2,
        len(CloudSEN12.bands),
        resolved.input.image_size,
        resolved.input.image_size,
        device=resolved.runtime.device,
    )
    with torch.inference_mode():
        features = model(images)
    assert features.shape == (2, 16)
    assert features.device.type == "cpu"


def test_all_argument_categories_dispatch_typed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[FlopsConfig] = []
    monkeypatch.setattr(commands, "run_flops", received.append)
    _flops.run(
        parser().parse_args(
            [
                "--model",
                "custom",
                "--model-target",
                "not_installed.Custom",
                "--model-kwargs",
                "{features: 8}",
                "--device",
                "auto",
                "--band-source",
                "so2sat",
                "--band-configs",
                "s2",
                "--normalization",
                "none",
                "--image-size",
                "32",
                "--probe-head",
                "mlp",
                "--probe-num-classes",
                "19",
                "--seg-heads",
                "linear",
                "conv_block",
                "fpn",
                "dpt",
                "patch_linear",
                "--seg-band-configs",
                "s2",
                "--seg-num-classes",
                "6",
                "--seg-layers",
                "four",
                "three",
                "two",
                "one",
                "--temporal-pool",
                "max",
                "--timing-batch-size",
                "2",
                "--n-warmup",
                "0",
                "--n-measure",
                "1",
                "--output",
                "other.csv",
                "--resume",
            ]
        )
    )
    config = received[0]
    assert config.model.target == "not_installed.Custom"
    assert config.model.kwargs == {"features": 8}
    assert config.input.band_source == "so2sat"
    assert config.classification.head == "mlp"
    assert config.classification.num_classes == 19
    assert len(config.segmentation.heads) == 5
    assert config.segmentation.num_classes == 6
    assert config.segmentation.probe.layers == ["four", "three", "two", "one"]
    assert config.segmentation.probe.temporal_pool == "max"
    assert config.timing.batch_size == 2
    assert config.output.file == "other.csv"


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "model: [",
        "model: {name: rcf, name: rcf}",
        "model: {name: rcf}\nruntime: {seed: true}",
        "model: {name: rcf}\nunknown: true",
        "model: !!python/object:os.system {}",
        "model: {name: rcf}\ninput: {band_source: missing}",
    ],
)
def test_bad_yaml_has_concise_errors(
    tmp_path: Path, text: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(SystemExit) as error:
        _flops.run(parser().parse_args(["--config", str(path), "--dry-run"]))
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_missing_yaml_is_a_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _flops.run(parser().parse_args(["--config", str(tmp_path / "missing"), "--dry-run"]))
    assert error.value.code == 2
    assert "No such file" in capsys.readouterr().err


@pytest.mark.parametrize("kwargs", ["[1]", "{x: 1, x: 2}", "{x: !!python/object:os.system {}}"])
def test_constructor_options_use_strict_safe_yaml(
    kwargs: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        _flops.run(parser().parse_args(["--model", "rcf", "--model-kwargs", kwargs, "--dry-run"]))
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["model=rcf"], ["+model.features=8"], ["++device=cpu"]])
def test_canonical_parser_rejects_old_override_syntax(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["flops", *arguments])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "overrides have been retired" in message
    assert "--config" in message


def test_canonical_parser_dispatches_yaml_and_explicit_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[FlopsConfig] = []
    monkeypatch.setattr(commands, "run_flops", received.append)
    path = tmp_path / "flops.yaml"
    path.write_text(
        "model: {name: rcf, kwargs: {features: 8}}\n"
        "runtime: {device: cuda}\noutput: {resume: true}\n"
    )
    cli_main(["flops", "--config", str(path), "--device", "cpu", "--no-resume"])
    assert isinstance(received[0], FlopsConfig)
    assert received[0].model.kwargs == {"features": 8}
    assert received[0].runtime.device == "cpu"
    assert not received[0].output.resume


@pytest.mark.parametrize(
    "arguments",
    [["--config-help"], ["--model", "rcf", "--dry-run"], ["--help"]],
)
def test_discovery_and_dry_run_do_not_import_runtime(arguments: list[str]) -> None:
    script = """
import argparse
import sys
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchgeo', 'timm', 'pandas', 'omegaconf'}:
            raise AssertionError(f'Unexpected heavy import: {fullname}')
sys.meta_path.insert(0, BlockRuntime())
from geo_eval_harness.commands.flops_arguments import add_flops_arguments
from geo_eval_harness.commands._flops import run
parser = argparse.ArgumentParser()
add_flops_arguments(parser)
run(parser.parse_args(sys.argv[1:]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout


def test_dry_run_round_trip_is_reusable(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    _flops.run(
        parser().parse_args(
            ["--model", "rcf", "--device", "cpu", "--seg-heads", "--no-resume", "--dry-run"]
        )
    )
    first = yaml.safe_load(capsys.readouterr().out)
    path = tmp_path / "printed.yaml"
    path.write_text(yaml.safe_dump(first))
    second = _flops.load_config(parser().parse_args(["--config", str(path)]))
    assert second.model_dump_yaml() == first
    assert second.segmentation.heads == []
    assert not second.output.resume


def test_packaged_flops_yaml_uses_the_typed_schema() -> None:
    path = Path(__file__).parents[1] / "src" / "geo_eval_harness" / "conf" / "flops_config.yaml"
    config = _flops.load_config(parser().parse_args(["--config", str(path), "--model", "rcf"]))
    resolved, preset = config.resolve()
    assert preset.kwargs["seed"] == 0
    assert resolved.input.image_size == 224
    assert resolved.segmentation.heads == ["fpn", "dpt"]


def test_real_rcf_synthetic_cpu_csv_and_resume(tmp_path: Path) -> None:
    import pandas as pd

    output = tmp_path / "synthetic.csv"
    arguments = [
        "--model",
        "rcf",
        "--model-kwargs",
        "{features: 8}",
        "--device",
        "cpu",
        "--image-size",
        "16",
        "--timing-batch-size",
        "2",
        "--n-warmup",
        "0",
        "--n-measure",
        "1",
        "--seg-heads",
        "--probe-head",
        "mlp",
        "--seed",
        "23",
        "--output",
        str(output),
    ]
    _flops.run(parser().parse_args(arguments))
    rows = pd.read_csv(output)
    assert rows["band_config"].tolist() == ["rgb", "s2"]
    assert rows["n_channels"].tolist() == [3, 12]
    assert (rows["gflops_total"] > 0).all()
    assert (rows["throughput_samples_per_sec"] > 0).all()
    assert rows["peak_gpu_mem_gb"].isna().all()
    before = output.read_bytes()
    _flops.run(parser().parse_args(arguments))
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    ("device", "available", "message"),
    [("cuda", False, "CUDA is unavailable"), ("cuda:2", True, "CUDA index 2")],
)
def test_invalid_cuda_errors_without_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, *, available: bool, message: str
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=message):
        _flops.run(
            parser().parse_args(
                ["--model", "rcf", "--device", device, "--output", str(tmp_path / "cuda.csv")]
            )
        )
    assert not (tmp_path / "cuda.csv").exists()
