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
from geo_eval_harness.config.presets import NORMALIZATIONS, ModelPreset, build_model
from geo_eval_harness.config.profile import ProfileConfig
from geo_eval_harness.config.schema import ModelConfig
from tests.support.torchgeo_models import install_tiny_scalemae_factory


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


class _Runtime:
    """Mocked dataset loading and model construction for ``_profile_runtime.run``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        self.load = mock.Mock(return_value=(None, _Loader(), None, None))
        self.build = mock.Mock(return_value=nn.Identity())
        self._capsys = capsys
        monkeypatch.setattr(_profile_runtime, "get_datasets", self.load)
        monkeypatch.setattr(_profile_runtime, "build_model", self.build)

    def record(self, config: ProfileConfig) -> dict[str, Any]:
        """Run the profile and parse the JSON record printed to stdout."""
        _profile_runtime.run(config)
        return json.loads(self._capsys.readouterr().out)


def _parse_profile_args(flags: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    return parser.parse_args(flags.split())


@pytest.fixture
def profile_args() -> argparse.Namespace:
    return _parse_profile_args(
        "--model rcf --dataset m-eurosat --partition default --device cpu --bands rgb "
        "--image-size 8 --interpolation bilinear --batch-size 4 --warmup 0 --measurements 1 "
        "--precision float32 --no-count-flops --seed 0 --normalization dataset"
    )


@pytest.fixture
def profile_config(profile_args: argparse.Namespace) -> ProfileConfig:
    return load_profile_config(profile_args)


@pytest.fixture
def profile_runtime(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> _Runtime:
    return _Runtime(monkeypatch, capsys)


class TestProfileCommand:
    @pytest.mark.parametrize("count_flops", [False, True])
    def test_profile_emits_fixed_real_batch_metadata(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime, *, count_flops: bool
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

        profile_runtime.load.return_value = (None, _Loader(batch), None, None)
        profile_runtime.build.return_value = CountingModel(3, 3, 1)

        record = profile_runtime.record(profile_config)

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
            == hashlib.sha256(
                json.dumps(record["model_config"], sort_keys=True).encode()
            ).hexdigest()
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
    @pytest.mark.usefixtures("profile_runtime")
    def test_profile_writes_json_to_requested_destination(
        self,
        profile_args: argparse.Namespace,
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
        [("", 64, 64), ("--image-size 32", 32, 32), ("--image-size none", 64, None)],
    )
    def test_profile_applies_image_size_to_model_and_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
        profile_runtime: _Runtime,
        image_size_flags: str,
        expected_size: int,
        loader_size: int | None,
    ) -> None:
        args = _parse_profile_args(
            "--model torchgeo/scalemae_large_fmow --dataset m-eurosat --batch-size 4 --warmup 0 "
            f"--measurements 1 --normalization none {image_size_flags}"
        )
        calls = install_tiny_scalemae_factory(monkeypatch)

        def load(**kwargs: Any) -> tuple:
            size = kwargs["image_size"] or 64
            batch = torch.ones(kwargs["batch_size"], 3, size, size)
            return _ImageDataset(batch), _Loader(batch), None, None

        profile_runtime.load.side_effect = load
        profile_runtime.build.side_effect = build_model

        record = profile_runtime.record(load_profile_config(args))

        inputs = profile_runtime.load.call_args.kwargs
        options = profile_runtime.build.call_args.kwargs
        model_kwargs = record["model_config"]["kwargs"]
        assert inputs["image_size"] == record["image_size"] == loader_size
        assert calls[0][0]["img_size"] == expected_size
        assert record["model_config"]["input"]["image_size"] == loader_size
        assert record["input_shape"] == [4, 3, expected_size, expected_size]
        assert inputs["interpolation"] == record["interpolation"] == "area"
        assert inputs["partition_name"] == "default"
        assert model_kwargs["res"] == 3.5
        assert model_kwargs["pool"] == "cls"
        assert model_kwargs["auto_resize"] is False
        assert options["normalization"] == record["normalization"] == "identity"
        assert record["input_normalization"] == "identity"
        assert [band.name for band in options["bands"]] == record["bands"]
        assert "dataset" not in options
        assert "dataset_overrides" not in record["model_config"]
        assert "segmentation" not in model_kwargs

    @pytest.mark.parametrize("seed", [7, 19])
    def test_profile_resolves_requested_seed_before_building_rcf(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime, seed: int
    ) -> None:
        profile_config.runtime.seed = seed
        models: list[Any] = []

        def build(preset: ModelPreset, **kwargs: Any) -> nn.Module:
            models.append(build_model(preset, **kwargs))
            return models[-1]

        def load(**_: Any) -> tuple:
            offset = np.random.random()  # noqa: NPY002 - Verify the seeded dataset-transform RNG.
            return None, _Loader(torch.randn(4, 3, 8, 8) + offset), None, None

        profile_runtime.load.side_effect = load
        profile_runtime.build.side_effect = build

        record = profile_runtime.record(profile_config)
        repeated = profile_runtime.record(profile_config)

        assert record["seed"] == record["model_config"]["kwargs"]["seed"] == seed
        assert "image_size" not in record["model_config"]["kwargs"]
        assert record["sample_sha256"] == repeated["sample_sha256"]
        assert record["model_config_hash"] == repeated["model_config_hash"]
        expected_filters = torch.randn(256, 3, 3, 3, generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(models[0].rcf.weights, expected_filters, rtol=0, atol=0)

    def test_profile_supplies_selected_training_dataset_to_empirical_rcf(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime
    ) -> None:
        profile_config.model = ModelConfig(name="rcf", kwargs={"mode": "empirical", "features": 8})
        profile_config.runtime.seed = 23
        batch = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        train_dataset = _ImageDataset(batch)
        profile_runtime.load.return_value = (train_dataset, _Loader(batch), None, None)
        profile_runtime.build.side_effect = build_model

        record = profile_runtime.record(profile_config)

        assert profile_runtime.build.call_args.kwargs["dataset"] is train_dataset
        assert record["model_config"]["kwargs"]["mode"] == "empirical"
        assert record["model_config"]["kwargs"]["features"] == 8
        assert record["model_config"]["kwargs"]["seed"] == 23
        assert "dataset" not in record["model_config"]["kwargs"]
        assert record["profile"]["throughput_samples_per_sec"] > 0

    @pytest.mark.parametrize("normalization", list(NORMALIZATIONS))
    def test_profile_preserves_custom_constructor_and_normalization_metadata(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime, normalization: str
    ) -> None:
        profile_config.model = ModelConfig(
            name="custom",
            target="custom.Model",
            kwargs={"input_normalization": "imagenet", "options": {"_target_": "ordinary.mapping"}},
        )
        profile_config.input = profile_config.input.model_copy(
            update={"normalization": normalization}
        )

        record = profile_runtime.record(profile_config)

        preset = profile_runtime.build.call_args.args[0]
        options = profile_runtime.build.call_args.kwargs
        assert preset.target == "custom.Model"
        assert preset.kwargs == profile_config.model.kwargs
        assert options["normalization"] == record["normalization"] == NORMALIZATIONS[normalization]
        assert record["input_normalization"] == "imagenet"
        assert record["model_config"]["kwargs"] == profile_config.model.kwargs
        assert record["model_config"]["input"]["normalization"] == normalization

    @pytest.mark.parametrize("bands", ["all", ["nir", "red"]])
    def test_profile_passes_input_options_and_band_order(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime, bands: str | list[str]
    ) -> None:
        profile_config.input.bands = bands
        profile_config.input.partition = "0.01x_train"
        profile_config.input.time_steps = 2
        profile_config.runtime.workers = 1
        channels = 13 if bands == "all" else len(bands)
        profile_runtime.load.return_value = (
            None,
            _Loader(torch.ones(4, channels, 8, 8)),
            None,
            None,
        )

        record = profile_runtime.record(profile_config)

        options = profile_runtime.load.call_args.kwargs
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
        self, profile_config: ProfileConfig, profile_runtime: _Runtime
    ) -> None:
        profile_runtime.load.return_value = (None, _Loader(torch.ones(2, 3, 8, 8)), None, None)
        with pytest.raises(RuntimeError, match="dataset returned batch size 2, requested 4"):
            _profile_runtime.run(profile_config)
        profile_runtime.build.assert_not_called()

    def test_profile_keeps_upstream_diagnostics_off_json_stdout(
        self,
        profile_config: ProfileConfig,
        profile_runtime: _Runtime,
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

        profile_runtime.load.side_effect = load
        profile_runtime.build.side_effect = build

        _profile_runtime.run(profile_config)

        captured = capsys.readouterr()
        assert json.loads(captured.out)["input_shape"] == [4, 3, 8, 8]
        assert captured.err == "upstream dataset\nupstream model\nupstream forward\n"

    @pytest.mark.parametrize(
        ("device", "available", "message"),
        [("cuda:0", False, "CUDA is unavailable"), ("cuda:2", True, "CUDA index 2")],
    )
    def test_profile_rejects_invalid_cuda_before_loading_data(
        self,
        profile_config: ProfileConfig,
        profile_runtime: _Runtime,
        device: str,
        *,
        available: bool,
        message: str,
    ) -> None:
        profile_config.runtime.device = device
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=available),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            pytest.raises(SystemExit, match=rf"error: .*{message}"),
        ):
            _profile_runtime.run(profile_config)
        profile_runtime.load.assert_not_called()
        profile_runtime.build.assert_not_called()

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [("auto", "cpu"), ("auto", "cuda:1"), ("cuda", "cuda:1"), ("cuda:0", "cuda:0")],
    )
    def test_profile_records_resolved_device(
        self,
        profile_config: ProfileConfig,
        monkeypatch: pytest.MonkeyPatch,
        profile_runtime: _Runtime,
        requested: str,
        expected: str,
    ) -> None:
        device = torch.device(expected)
        cuda = device.type == "cuda"
        batch = mock.Mock(
            spec=torch.Tensor, device=device, dtype=torch.float32, shape=(4, 3, 8, 8), ndim=4
        )
        batch.to.return_value = batch
        batch.detach.return_value = torch.ones(4, 3, 8, 8)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Mock GPU")
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * 1024**3)
        profile_runtime.load.return_value = (None, _Loader(batch), None, None)
        profile_config.runtime.device = requested

        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            record = profile_runtime.record(profile_config)

        assert record["device"] == record["profile"]["device"] == expected
        assert record["device_index"] == device.index
        assert record["profile"]["throughput_samples_per_sec"] > 0
        assert record["profile"]["peak_gpu_mem_gb"] == (1 if cuda else None)
        batch.to.assert_called_once_with(device)
        assert synchronize.call_args_list == [mock.call(device)] * (2 if cuda else 0)

    def test_profile_propagates_model_failures(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime
    ) -> None:
        profile_runtime.build.side_effect = RuntimeError("unexpected model failure")
        with pytest.raises(RuntimeError, match="unexpected model failure"):
            _profile_runtime.run(profile_config)

    def test_profile_dispatches_typed_config_through_lazy_runtime(
        self, profile_args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[ProfileConfig] = []
        monkeypatch.setattr(commands, "_profile_runtime", SimpleNamespace(run=calls.append))

        profile(profile_args)

        assert len(calls) == 1
        assert isinstance(calls[0], ProfileConfig)
        assert calls[0].model.name == "rcf"
        assert calls[0].input.normalization == "dataset"

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
        self,
        profile_args: argparse.Namespace,
        profile_runtime: _Runtime,
        capsys: pytest.CaptureFixture[str],
        invalid_request: tuple[str, object, str],
    ) -> None:
        field, value, message = invalid_request
        setattr(profile_args, field, value)

        with pytest.raises(SystemExit) as error:
            profile(profile_args)

        assert error.value.code == 2
        assert message in capsys.readouterr().err
        profile_runtime.load.assert_not_called()

    @pytest.mark.parametrize("entrypoint", ["standalone", "image-run"])
    def test_profiling_preserves_dates_for_timing_flops_and_cpu_batches(
        self, profile_config: ProfileConfig, profile_runtime: _Runtime, entrypoint: str
    ) -> None:
        from geo_eval_harness.config.presets import resolve_run_config
        from geo_eval_harness.config.run import RunConfig
        from geo_eval_harness.datasets import get_bench_dataset_class
        from geo_eval_harness.main import dataset_metadata, evaluate_profile
        from tests.support.models import DatedBenchModel

        images = torch.zeros(4, 1, 8, 8)
        dates = torch.tensor([20, 70, 150, 240])
        loader = [{"image": images, "doy": dates}]
        model = DatedBenchModel()
        observed: list[torch.Tensor] = []
        model.spatial_features.register_forward_hook(
            lambda _module, _inputs, output: observed.append(output[:, 0, 0, 0].clone())
        )
        if entrypoint == "standalone":
            profile_config.warmup = 1
            profile_config.measurements = 1
            profile_config.count_flops = True
            profile_runtime.load.return_value = (None, loader, None, None)
            profile_runtime.build.return_value = model
            record = profile_runtime.record(profile_config)
            assert record["profile"]["flops"]["status"] == "measured"
            assert record["doy_sha256"] == hashlib.sha256(dates.numpy().tobytes()).hexdigest()
        else:
            config = RunConfig.model_validate(
                {
                    "model": {"name": "rcf"},
                    "datasets": ["m-eurosat"],
                    "runtime": {"device": "cpu"},
                    "profile": {
                        "n_warmup": 1,
                        "n_measure": 1,
                        "cpu_throughput": {
                            "enabled": True,
                            "batch_size": 2,
                            "n_warmup": 0,
                            "n_measure": 1,
                        },
                    },
                }
            )
            config, preset = resolve_run_config(config, "m-eurosat")
            meta = dataset_metadata(
                config, "m-eurosat", get_bench_dataset_class("m-eurosat"), preset, "test"
            )
            meta.update(feature_dim=1, n_train=4, n_val=4, n_test=4)
            rows = evaluate_profile(model, loader, config, meta)
            assert any(row["metric_name"] == "throughput_samples_per_sec_cpu" for row in rows)
        assert [len(values) for values in observed] == (
            [4, 4, 1] if entrypoint == "standalone" else [4, 4, 1, 2]
        )
        for values in observed:
            torch.testing.assert_close(values, dates[: len(values)].float())
