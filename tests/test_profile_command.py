# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Offline regression tests for standalone real-batch profiling."""

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from geo_eval_harness import commands
from geo_eval_harness.commands import _profile_runtime
from geo_eval_harness.commands._profile import profile
from geo_eval_harness.commands.profile_arguments import add_profile_arguments, load_profile_config
from geo_eval_harness.config.presets import NORMALIZATIONS, ModelPreset
from geo_eval_harness.config.profile import ProfileConfig
from geo_eval_harness.config.schema import ModelConfig


class _Loader:
    def __init__(self, batch: torch.Tensor | None = None) -> None:
        self.batch = torch.ones(4, 3, 8, 8) if batch is None else batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield {"image": self.batch}


class _ImageDataset(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index]}


@pytest.fixture
def profile_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    return parser.parse_args(
        [
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--partition",
            "default",
            "--device",
            "cpu",
            "--bands",
            "rgb",
            "--image-size",
            "8",
            "--interpolation",
            "bilinear",
            "--batch-size",
            "4",
            "--warmup",
            "0",
            "--measurements",
            "1",
            "--precision",
            "float32",
            "--no-count-flops",
            "--seed",
            "0",
            "--normalization",
            "dataset",
        ]
    )


@pytest.fixture
def profile_config(profile_args: argparse.Namespace) -> ProfileConfig:
    return load_profile_config(profile_args)


@pytest.mark.parametrize("count_flops", [False, True])
def test_profile_emits_fixed_real_batch_metadata(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    count_flops: bool,
) -> None:
    profile_config.count_flops = count_flops
    profile_config.warmup = 2
    profile_config.measurements = 3
    batch = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    calls: list[torch.Tensor] = []

    class CountingModel(nn.Conv2d):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            calls.append(images)
            return super().forward(images)

    monkeypatch.setattr(
        _profile_runtime, "get_datasets", lambda **_: (None, _Loader(batch), None, None)
    )
    monkeypatch.setattr(_profile_runtime, "build_model", lambda *_, **__: CountingModel(3, 3, 1))

    _profile_runtime.run(profile_config)

    record = json.loads(capsys.readouterr().out)
    assert record["model"] == "rcf"
    assert record["dataset"] == "m-eurosat"
    assert record["bands"] == ["red", "green", "blue"]
    assert record["input_shape"] == record["profile"]["input_shape"] == [4, 3, 8, 8]
    assert record["profile"]["batch_size"] == 4
    assert record["profile"]["warmup"] == 2
    assert record["profile"]["measurements"] == 3
    assert record["profile"]["throughput_samples_per_sec"] > 0
    assert record["profile"]["flops"]["status"] == ("measured" if count_flops else "disabled")
    assert record["sample_sha256"] == hashlib.sha256(batch.numpy().tobytes()).hexdigest()
    assert (
        record["model_config_hash"]
        == hashlib.sha256(json.dumps(record["model_config"], sort_keys=True).encode()).hexdigest()
    )
    assert record["device"] == "cpu"
    assert record["device_index"] is None
    assert record["hardware"]
    assert record["torch_version"]
    assert record["python_version"]
    assert record["timestamp_utc"].endswith("+00:00")
    assert record["scope"].startswith("encoder inference")
    assert len(calls) == 5 + count_flops
    assert all(images is batch for images in calls[:5])
    if count_flops:
        assert calls[-1].shape == (1, 3, 8, 8)


@pytest.mark.parametrize("destination", ["directory", "file", "both"])
def test_profile_writes_json_to_requested_destination(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    destination: str,
) -> None:
    root = tmp_path / "nested" / "profiles"
    explicit = tmp_path / "selected" / "measurement.json"
    if destination != "file":
        profile_args.output_dir = str(root)
    if destination != "directory":
        profile_args.output = str(explicit)
    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(_profile_runtime, "build_model", lambda *_, **__: nn.Conv2d(3, 3, 1))

    profile(profile_args)

    path = root / "profile.json" if destination == "directory" else explicit
    assert capsys.readouterr().out == ""
    assert set(tmp_path.rglob("*.json")) == {path}
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["model"] == "rcf"
    assert record["dataset"] == "m-eurosat"
    assert record["profile"]["throughput_samples_per_sec"] > 0
    if destination == "both":
        assert not root.exists()


@pytest.mark.parametrize(
    ("image_size_flags", "expected_size", "loader_size"),
    [([], 64, 64), (["--image-size", "32"], 32, 32), (["--image-size", "none"], 64, None)],
)
def test_profile_applies_image_size_to_model_and_loader(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    image_size_flags: list[str],
    expected_size: int,
    loader_size: int | None,
) -> None:
    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    args = parser.parse_args(
        [
            "--model",
            "torchgeo/scalemae_large_fmow",
            "--dataset",
            "m-eurosat",
            "--batch-size",
            "4",
            "--warmup",
            "0",
            "--measurements",
            "1",
            "--normalization",
            "none",
            *image_size_flags,
        ]
    )
    inputs: dict[str, Any] = {}
    construction: dict[str, Any] = {}

    def load(**kwargs: Any) -> tuple:
        inputs.update(kwargs)
        size = kwargs["image_size"] or 64
        batch = torch.ones(kwargs["batch_size"], 3, size, size)
        return _ImageDataset(batch), _Loader(batch), None, None

    def build(preset: ModelPreset, **kwargs: Any) -> nn.Module:
        construction["preset"] = preset
        construction["options"] = kwargs
        return nn.Conv2d(3, 2, 1)

    monkeypatch.setattr(_profile_runtime, "get_datasets", load)
    monkeypatch.setattr(_profile_runtime, "build_model", build)

    _profile_runtime.run(load_profile_config(args))

    record = json.loads(capsys.readouterr().out)
    model_kwargs = record["model_config"]["kwargs"]
    assert inputs["image_size"] == record["image_size"] == loader_size
    assert (
        construction["preset"].kwargs["image_size"] == model_kwargs["image_size"] == expected_size
    )
    assert record["model_config"]["input"]["image_size"] == loader_size
    assert record["input_shape"] == [4, 3, expected_size, expected_size]
    assert inputs["interpolation"] == record["interpolation"] == "area"
    assert inputs["partition_name"] == "default"
    assert model_kwargs["res"] == 3.5
    assert model_kwargs["pool"] == "cls"
    assert model_kwargs["auto_resize"] is False
    assert construction["options"]["normalization"] == record["normalization"] == "identity"
    assert record["input_normalization"] == "identity"
    assert [band.name for band in construction["options"]["bands"]] == record["bands"]
    assert "dataset" not in construction["options"]
    assert "dataset_overrides" not in record["model_config"]
    assert "segmentation" not in model_kwargs


@pytest.mark.parametrize("seed", [7, 19])
def test_profile_resolves_requested_seed_before_building_rcf(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seed: int,
) -> None:
    profile_config.runtime.seed = seed
    models: list[Any] = []
    build_model = _profile_runtime.build_model

    def build(preset: ModelPreset, **kwargs: Any) -> nn.Module:
        model = build_model(preset, **kwargs)
        models.append(model)
        return model

    def load(**_: Any) -> tuple:
        offset = np.random.random()  # noqa: NPY002 - Verify the seeded dataset-transform RNG.
        return None, _Loader(torch.randn(4, 3, 8, 8) + offset), None, None

    monkeypatch.setattr(_profile_runtime, "get_datasets", load)
    monkeypatch.setattr(_profile_runtime, "build_model", build)

    _profile_runtime.run(profile_config)
    record = json.loads(capsys.readouterr().out)
    _profile_runtime.run(profile_config)
    repeated = json.loads(capsys.readouterr().out)

    assert record["seed"] == record["model_config"]["kwargs"]["seed"] == seed
    assert "image_size" not in record["model_config"]["kwargs"]
    assert record["sample_sha256"] == repeated["sample_sha256"]
    assert record["model_config_hash"] == repeated["model_config_hash"]
    expected_filters = torch.randn(256, 3, 3, 3, generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(models[0].rcf.weights, expected_filters, rtol=0, atol=0)


def test_profile_supplies_selected_training_dataset_to_empirical_rcf(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_config.model = ModelConfig(name="rcf", kwargs={"mode": "empirical", "features": 8})
    profile_config.runtime.seed = 23
    batch = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    train_dataset = _ImageDataset(batch)
    build_model = _profile_runtime.build_model
    construction: dict[str, Any] = {}

    def build(preset: ModelPreset, **kwargs: Any) -> nn.Module:
        construction.update(kwargs)
        return build_model(preset, **kwargs)

    monkeypatch.setattr(
        _profile_runtime,
        "get_datasets",
        lambda **_: (train_dataset, _Loader(batch), None, None),
    )
    monkeypatch.setattr(_profile_runtime, "build_model", build)

    _profile_runtime.run(profile_config)

    record = json.loads(capsys.readouterr().out)
    assert construction["dataset"] is train_dataset
    assert record["model_config"]["kwargs"]["mode"] == "empirical"
    assert record["model_config"]["kwargs"]["features"] == 8
    assert record["model_config"]["kwargs"]["seed"] == 23
    assert "dataset" not in record["model_config"]["kwargs"]
    assert record["profile"]["throughput_samples_per_sec"] > 0


@pytest.mark.parametrize("normalization", list(NORMALIZATIONS))
def test_profile_preserves_custom_constructor_and_normalization_metadata(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    normalization: str,
) -> None:
    profile_config.model = ModelConfig(
        name="custom",
        target="custom.Model",
        kwargs={"input_normalization": "imagenet", "options": {"_target_": "ordinary.mapping"}},
    )
    profile_config.input = profile_config.input.model_copy(update={"normalization": normalization})
    construction: dict[str, Any] = {}

    def build(preset: ModelPreset, **kwargs: Any) -> nn.Module:
        construction.update(kwargs)
        assert preset.target == "custom.Model"
        assert preset.kwargs == profile_config.model.kwargs
        return nn.Identity()

    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(_profile_runtime, "build_model", build)

    _profile_runtime.run(profile_config)

    record = json.loads(capsys.readouterr().out)
    assert construction["normalization"] == record["normalization"] == NORMALIZATIONS[normalization]
    assert record["input_normalization"] == "imagenet"
    assert record["model_config"]["kwargs"] == profile_config.model.kwargs
    assert record["model_config"]["input"]["normalization"] == normalization


@pytest.mark.parametrize("bands", ["all", ["nir", "red"]])
def test_profile_passes_input_options_and_band_order(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bands: str | list[str],
) -> None:
    profile_config.input.bands = bands
    profile_config.input.partition = "0.01x_train"
    profile_config.input.time_steps = 2
    profile_config.runtime.workers = 1
    options: dict[str, Any] = {}

    def load(**kwargs: Any) -> tuple:
        options.update(kwargs)
        channels = 13 if bands == "all" else len(bands)
        return None, _Loader(torch.ones(4, channels, 8, 8)), None, None

    monkeypatch.setattr(_profile_runtime, "get_datasets", load)
    monkeypatch.setattr(_profile_runtime, "build_model", lambda *_, **__: nn.Identity())

    _profile_runtime.run(profile_config)

    record = json.loads(capsys.readouterr().out)
    assert options["bands"] == bands
    assert options["partition_name"] == record["dataset_partition"] == "0.01x_train"
    assert options["time_steps"] == 2
    assert options["num_workers"] == 1
    if isinstance(bands, list):
        assert record["bands"] == bands
    else:
        assert len(record["bands"]) == 13
        assert record["bands"][0] == "coastal_aerosol"


def test_profile_rejects_short_batches_before_construction(
    profile_config: ProfileConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _profile_runtime,
        "get_datasets",
        lambda **_: (None, _Loader(torch.ones(2, 3, 8, 8)), None, None),
    )
    with pytest.raises(RuntimeError, match="dataset returned batch size 2, requested 4"):
        _profile_runtime.run(profile_config)


def test_profile_keeps_upstream_diagnostics_off_json_stdout(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class NoisyModel(nn.Identity):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            sys.stdout.write("upstream forward\n")
            return images

    def load(**_: Any) -> tuple:
        sys.stdout.write("upstream dataset\n")
        return None, _Loader(), None, None

    def build(*_: Any, **__: Any) -> nn.Module:
        sys.stdout.write("upstream model\n")
        return NoisyModel()

    monkeypatch.setattr(_profile_runtime, "get_datasets", load)
    monkeypatch.setattr(_profile_runtime, "build_model", build)

    _profile_runtime.run(profile_config)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["input_shape"] == [4, 3, 8, 8]
    assert captured.err == "upstream dataset\nupstream model\nupstream forward\n"


def test_profile_rejects_unavailable_cuda_before_loading_data(
    profile_config: ProfileConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_config.runtime.device = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with (
        mock.patch.object(_profile_runtime, "_load_batch") as load,
        mock.patch.object(_profile_runtime, "_build_model") as build,
        pytest.raises(SystemExit, match=r"error: .*CUDA is unavailable"),
    ):
        _profile_runtime.run(profile_config)
    load.assert_not_called()
    build.assert_not_called()


def test_profile_validates_cuda_indices(
    profile_config: ProfileConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    profile_config.runtime.device = "cuda:2"
    with (
        mock.patch.object(_profile_runtime, "_load_batch") as load,
        pytest.raises(SystemExit, match=r"error: .*CUDA index 2"),
    ):
        _profile_runtime.run(profile_config)
    load.assert_not_called()


def test_profile_auto_uses_cpu_without_cuda(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(_profile_runtime, "build_model", lambda *_, **__: nn.Identity())
    profile_config.runtime.device = "auto"
    _profile_runtime.run(profile_config)
    record = json.loads(capsys.readouterr().out)
    assert record["device"] == record["profile"]["device"] == "cpu"
    assert record["profile"]["throughput_samples_per_sec"] > 0


@pytest.mark.parametrize(
    ("requested", "expected"), [("auto", "cuda:1"), ("cuda", "cuda:1"), ("cuda:0", "cuda:0")]
)
def test_profile_records_resolved_cuda_index(
    profile_config: ProfileConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requested: str,
    expected: str,
) -> None:
    device = torch.device(expected)
    batch = mock.Mock(
        spec=torch.Tensor, device=device, dtype=torch.float32, shape=(4, 3, 8, 8), ndim=4
    )
    batch.to.return_value = batch
    batch.detach.return_value = torch.ones(4, 3, 8, 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Mock GPU")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * 1024**3)
    monkeypatch.setattr(
        _profile_runtime, "get_datasets", lambda **_: (None, _Loader(batch), None, None)
    )
    monkeypatch.setattr(_profile_runtime, "build_model", lambda *_, **__: nn.Identity())
    profile_config.runtime.device = requested
    with mock.patch.object(torch.cuda, "synchronize") as synchronize:
        _profile_runtime.run(profile_config)
    record = json.loads(capsys.readouterr().out)
    assert record["device"] == record["profile"]["device"] == expected
    assert record["device_index"] == device.index
    assert record["profile"]["throughput_samples_per_sec"] > 0
    assert record["profile"]["peak_gpu_mem_gb"] == 1
    batch.to.assert_called_once_with(device)
    assert synchronize.call_args_list == [mock.call(device), mock.call(device)]


def test_profile_propagates_model_failures(
    profile_config: ProfileConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build(*_: Any, **__: Any) -> nn.Module:
        raise RuntimeError("unexpected model failure")

    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(_profile_runtime, "build_model", build)
    with pytest.raises(RuntimeError, match="unexpected model failure"):
        _profile_runtime.run(profile_config)


def test_profile_dispatches_typed_config_through_lazy_runtime(
    profile_args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[ProfileConfig] = []
    monkeypatch.setattr(commands, "_profile_runtime", SimpleNamespace(run=calls.append))

    profile(profile_args)

    assert len(calls) == 1
    assert isinstance(calls[0], ProfileConfig)
    assert calls[0].model.name == "rcf"
    assert calls[0].input.normalization == "dataset"


def test_profile_reports_validation_errors_without_loading_runtime(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delattr(commands, "_profile_runtime")
    profile_args.batch_size = 0

    with pytest.raises(SystemExit) as error:
        profile(profile_args)
    assert error.value.code == 2
    assert "batch_size" in capsys.readouterr().err


@pytest.mark.parametrize(
    "invalid_request",
    [
        ("batch_size", 0, "batch_size"),
        ("warmup", -1, "warmup"),
        ("measurements", 0, "measurements"),
        ("seed", -1, "seed"),
        ("image_size", 0, "image_size"),
        ("model", "missing-model", "Unknown model config"),
        ("dataset", "missing-dataset", "unknown dataset"),
        ("bands", "red,,blue", "bands must contain non-empty names"),
    ],
)
def test_invalid_profile_request_fails_before_loading(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    invalid_request: tuple[str, object, str],
) -> None:
    field, value, message = invalid_request
    profile_args.model = "rcf"
    profile_args.dataset = "m-eurosat"
    setattr(profile_args, field, value)

    def unexpected_load(**_kwargs: object) -> None:
        pytest.fail("Invalid profile request reached dataset loading")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_profile_runtime, "get_datasets", unexpected_load)
    with pytest.raises(SystemExit) as error:
        profile(profile_args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
