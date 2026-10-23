# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the discoverable image CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from geo_eval_harness.cli import main
from geo_eval_harness.commands._config import set_path
from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.presets import resolve_run_config
from geo_eval_harness.config.run import validate_run_config


def test_dry_run_applies_explicit_flags_and_preserves_false_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "run",
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--dataset",
            "burn_scars",
            "--image-size",
            "none",
            "--no-resume",
            "--methods",
            "knn",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert "image_size: null" in output
    assert "resume: false" in output
    assert "- burn_scars" in output
    assert "methods:\n  - knn" in output


def test_config_values_are_overridden_by_explicit_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\nruntime: {seed: 8}\noutput: {resume: true}\n",
        encoding="utf-8",
    )
    main(["run", "--config", str(path), "--seed", "3", "--no-resume", "--dry-run"])
    output = capsys.readouterr().out
    assert "seed: 3" in output
    assert "resume: false" in output


def test_nested_flag_mapping() -> None:
    mapping = {}
    set_path(mapping, ("classification", "linear", "refit_train_val"), False)
    set_path(mapping, ("runtime", "workers"), 0)
    assert mapping == {
        "classification": {"linear": {"refit_train_val": False}},
        "runtime": {"workers": 0},
    }


def test_boolean_flags_override_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\n"
        "classification:\n  linear:\n    refit_train_val: true\n",
        encoding="utf-8",
    )
    main(
        [
            "run",
            "--config",
            str(path),
            "--no-refit-train-val",
            "--no-temp-scale",
            "--dry-run",
        ]
    )
    assert "refit_train_val: false" in capsys.readouterr().out


@pytest.mark.parametrize("selection", [["--dataset", "m-eurosat"], ["--model", "rcf"]])
def test_missing_model_or_dataset_fails_before_execution(selection: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", *selection, "--dry-run"])
    assert error.value.code == 2


def test_non_dry_run_calls_typed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.run", received.append)
    main(["run", "--model", "rcf", "--dataset", "m-eurosat"])
    assert received[0].model.name == "rcf"


def test_runtime_preserves_typed_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu", "batch_size": 2},
            "input": {
                "normalization": "none",
                "image_size": 8,
                "time_steps": 1,
                "interpolation": "nearest",
            },
            "classification": {"methods": ["knn"]},
        }
    )
    from geo_eval_harness.commands._run_runtime import run

    run(config)
    assert received[0].runtime.device == "cpu"
    assert received[0].input.normalization == "none"
    assert received[0].classification.methods == ["knn"]
    assert received[0] == config


@pytest.mark.parametrize(
    ("device", "available", "message"),
    [("cuda", False, "CUDA is unavailable"), ("cuda:2", True, "CUDA index 2")],
)
def test_runtime_rejects_invalid_cuda_before_loading(
    monkeypatch: pytest.MonkeyPatch, device: str, *, available: bool, message: str
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with (
        mock.patch("geo_eval_harness.main.get_datasets") as data,
        mock.patch("geo_eval_harness.main.build_model") as build,
        pytest.raises(ValueError, match=message),
    ):
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", device])
    data.assert_not_called()
    build.assert_not_called()


def test_runtime_preserves_model_and_segmentation_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "torchgeo/scalemae_large_fmow", "kwargs": {"res": 1.0}},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            "input": {"image_size": 8, "interpolation": "nearest"},
            "segmentation": {"layers": ["layer1"]},
        }
    )
    from geo_eval_harness.commands._run_runtime import run

    run(config)
    effective, preset = resolve_run_config(received[0], "m-eurosat")
    assert preset.kwargs["res"] == 1.0
    assert effective.input.image_size == 8
    assert effective.input.interpolation == "nearest"
    assert effective.segmentation.layers == ["layer1"]


def test_runtime_preserves_preset_layers_when_schema_omits_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "torchgeo/resnet50_s2rgb_satlas_si"},
            "datasets": ["burn_scars"],
            "runtime": {"device": "cpu"},
        }
    )
    from geo_eval_harness.commands._run_runtime import run

    run(config)
    effective, _ = resolve_run_config(received[0], "burn_scars")
    assert effective.segmentation.layers == [
        "layer4",
        "layer3",
        "layer2",
        "layer1",
    ]


def test_runtime_explicit_empty_layers_clear_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "torchgeo/resnet50_s2rgb_satlas_si"},
            "datasets": ["burn_scars"],
            "runtime": {"device": "cpu"},
            "segmentation": {"layers": []},
        }
    )
    from geo_eval_harness.commands._run_runtime import run

    run(config)
    effective, _ = resolve_run_config(received[0], "burn_scars")
    assert effective.segmentation.layers == []


def test_linear_only_reaches_typed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.run", received.append)
    main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--methods", "linear"])
    assert received[0].classification.methods == ["linear"]


def test_unknown_config_field_fails_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\nrunntim: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "error_type",
    [FileNotFoundError, PermissionError, ValueError, yaml.YAMLError, RuntimeError, TypeError],
)
def test_runtime_failure_propagates_from_cli(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    def fail(_: object, *, strict: bool) -> None:
        raise error_type("benchmark failed")

    monkeypatch.setattr("geo_eval_harness.commands._run_runtime.main", fail)
    with pytest.raises(error_type, match="benchmark failed"):
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu"])


@pytest.mark.parametrize(
    ("contents", "diagnostic"),
    [
        (None, "No such file"),
        ("model: [\n", "while parsing"),
        ("model: {name: rcf}\nmodel: {name: other}\n", "duplicate key"),
        ("[a, b]: value\n", "unhashable key"),
        ("model: !unknown rcf\n", "could not determine a constructor"),
        ("null\n", "top level"),
        ("runtime: null\n", "runtime"),
        ("runtime: []\n", "runtime"),
        ("runtime: {workers: -1}\n", "runtime.workers"),
    ],
)
def test_config_errors_exit_without_tracebacks(
    contents: str | None, diagnostic: str, tmp_path: Path
) -> None:
    path = tmp_path / "invalid.yaml"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "geo_eval_harness.cli",
            "run",
            "--config",
            str(path),
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert str(path) in result.stderr
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


def test_config_directory_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(tmp_path), "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(tmp_path) in message
    assert "directory" in message


def test_unreadable_config_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "unreadable.yaml"

    def deny_read(_: Path, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(Path, "open", deny_read)
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert "Permission denied" in message


def test_invalid_config_encoding_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid-encoding.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert "utf-8" in message


@pytest.mark.parametrize(
    ("section", "flags"),
    [
        ("model", ["--model", "rcf"]),
        ("runtime", ["--seed", "3"]),
        ("input", ["--image-size", "32"]),
        ("classification", ["--knn-k", "3"]),
        ("output", ["--no-resume"]),
        ("classification.linear", ["--no-refit-train-val"]),
        ("classification.calibration", ["--no-temp-scale"]),
    ],
)
@pytest.mark.parametrize("value", [None, [], [["seed", 8]], 3])
def test_invalid_sections_are_not_coerced_or_replaced_by_overrides(
    section: str,
    flags: list[str],
    value: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}
    if "." in section:
        parent, child = section.split(".")
        config[parent] = {child: value}
    else:
        config[section] = value
    path = tmp_path / "invalid-section.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), *flags, "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert section in message
    assert "mapping" in message


def test_flags_complete_partial_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("runtime: {seed: 8}\n", encoding="utf-8")
    main(
        [
            "run",
            "--config",
            str(path),
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--device",
            "cpu",
            "--seed",
            "3",
            "--dry-run",
        ]
    )
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["runtime"] == {"device": "cpu", "seed": 3}
    assert config["model"]["name"] == "rcf"
    assert config["datasets"] == ["m-eurosat"]


def test_nested_linear_override_preserves_sibling_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\n"
        "classification:\n  knn_k: 7\n  linear:\n    refit_train_val: true\n",
        encoding="utf-8",
    )
    main(["run", "--config", str(path), "--no-refit-train-val", "--dry-run"])
    output = capsys.readouterr().out
    assert "knn_k: 7" in output
    assert "refit_train_val: false" in output


def test_model_catalog_matches_packaged_presets(capsys: pytest.CaptureFixture[str]) -> None:
    main(["models"])
    assert capsys.readouterr().out.splitlines() == list_model_configs()


def test_catalog_name_selection_and_invalid_requests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["models"])
    assert "timm/resnet50" in capsys.readouterr().out
    main(["datasets"])
    assert "m-eurosat" in capsys.readouterr().out
    main(["models", "timm/resnet50"])
    assert "target:" in capsys.readouterr().out
    main(["datasets", "m-eurosat"])
    dataset_detail = capsys.readouterr().out
    assert "name: m-eurosat" in dataset_detail
    assert "task: classification" in dataset_detail
    with pytest.raises(SystemExit) as error:
        main(["profile"])
    assert error.value.code == 2
    with pytest.raises(SystemExit, match="unknown model"):
        main(["models", "unknown"])
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["datasets", "unknown"])


@pytest.mark.parametrize(("model", "dataset"), [("unknown", "m-eurosat"), ("rcf", "unknown")])
def test_unknown_catalog_entries_fail_before_execution(model: str, dataset: str) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", model, "--dataset", dataset])
    assert error.value.code == 2


def test_config_help_is_available_without_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config-help"])
    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out)["title"] == "RunConfig"


def test_main_rejects_unknown_parser_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    monkeypatch.setattr(
        "geo_eval_harness.cli._parse_args",
        lambda _: argparse.Namespace(command="other"),
    )
    with pytest.raises(SystemExit, match="not implemented"):
        main([])


@pytest.mark.parametrize("value", ["null", "3", "[]"])
def test_invalid_model_section_has_a_field_error(
    value: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"model: {value}\ndatasets: [m-eurosat]\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    assert "model" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--methods", "other"], ["--bands", "red,,blue"]])
def test_dry_run_rejects_unsupported_method_and_band_selections(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run", *flags])
    assert error.value.code == 2


def test_yaml_bands_require_a_list(tmp_path: Path) -> None:
    path = tmp_path / "bands.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\ninput:\n  bands: red,green\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2


def test_public_download_and_profile_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_eval_harness import commands

    received = []
    monkeypatch.setattr(commands, "download", received.append)
    monkeypatch.setattr(commands, "profile", received.append)
    main(["download", "m-eurosat", "burn_scars"])
    main(["profile", "--model", "rcf", "--dataset", "m-eurosat", "--batch-size", "4"])
    assert received[0].target == ["m-eurosat", "burn_scars"]
    assert received[1].model == "rcf"
    assert received[1].dataset == "m-eurosat"
    assert received[1].batch_size == 4


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:1")])
def test_auto_device_resolves_once_before_dataset_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, available: bool, expected: str
) -> None:
    import torch

    received = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def dataset_run(cfg: object, *args: object, **kwargs: object) -> tuple:
        received.append(cfg)
        return ()

    monkeypatch.setattr("geo_eval_harness.main.run_dataset", dataset_run)
    with mock.patch.object(torch.cuda, "is_available", return_value=available) as query:
        main(
            [
                "run",
                "--model",
                "rcf",
                "--dataset",
                "m-eurosat",
                "--device",
                "auto",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )
    assert received[0].runtime.device == expected
    query.assert_called_once_with()


@pytest.mark.parametrize(
    "model", ["torchgeo/scalemae_large_fmow", "torchgeo/resnet50_s2rgb_satlas_si"]
)
def test_dry_run_reload_preserves_preset_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    from geo_eval_harness.commands import _run_runtime

    received = []
    monkeypatch.setattr(
        _run_runtime,
        "main",
        lambda cfg, **kwargs: received.append(cfg.model_dump_yaml()),
    )
    arguments = ["run", "--model", model, "--dataset", "m-eurosat", "--device", "cpu"]
    main(arguments)
    main([*arguments, "--dry-run"])
    restored = validate_run_config(yaml.safe_load(capsys.readouterr().out))
    _run_runtime.run(restored)
    assert received[0] == received[1]
    assert not restored.input.model_fields_set
    assert not restored.segmentation.model_fields_set


def test_image_size_override_reaches_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_eval_harness.commands import _run_runtime

    received = []
    monkeypatch.setattr(_run_runtime, "main", lambda cfg, **kwargs: received.append(cfg))
    main(
        [
            "run",
            "--model",
            "torchgeo/scalemae_large_fmow",
            "--dataset",
            "m-eurosat",
            "--device",
            "cpu",
            "--image-size",
            "32",
        ]
    )
    config = received[0]
    assert config.input.image_size == 32
    assert resolve_run_config(config, "m-eurosat")[0].input.image_size == 32
