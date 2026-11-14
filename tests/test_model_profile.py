"""Unit tests for model profiling helpers."""

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch
from torch import nn

from geo_eval_harness import model_profile
from geo_eval_harness.model_profile import (
    ProfileTiming,
    _count_gflops,
    _count_params,
    measure_cpu_throughput,
    measure_profile,
    profile_inference,
)
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


def test_count_params_correct() -> None:
    model = nn.Linear(4, 8)
    assert _count_params(model) == pytest.approx(40 / 1e6)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_count_gflops_matches_conv_and_linear_ops(requires_grad) -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    sample = torch.rand(1, 3, 16, 16, requires_grad=requires_grad)
    gflops = _count_gflops(model, sample)
    expected = 2 * 14 * 14 * 4 * 3 * 3 * 3 + 2 * 4 * 2
    assert gflops == pytest.approx(expected / 1e9)
    assert sample.grad is None
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("use_profile", [False, True])
@pytest.mark.parametrize("outer_inference", [False, True])
def test_count_gflops_handles_parameter_views_without_mutating_model(
    *, use_profile: bool, outer_inference: bool
) -> None:
    calls: list[tuple[bool, bool, bool, bool]] = []

    class LearnedQuery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Parameter(torch.ones(1, 4))
            self.project = nn.Linear(4, 2)
            self.project.weight.requires_grad_(False)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            calls.append(
                (
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                    self.query.requires_grad,
                    images.requires_grad,
                )
            )
            return self.project(self.query.expand(images.shape[0], -1))

    model = LearnedQuery()
    model.project.eval()
    original_parameters = dict(model.named_parameters())
    original_flags = {
        name: parameter.requires_grad for name, parameter in original_parameters.items()
    }
    for parameter in original_parameters.values():
        parameter.grad = torch.ones_like(parameter)
    original_grads = {name: parameter.grad for name, parameter in original_parameters.items()}
    original_modes = {module: module.training for module in model.modules()}
    sample = torch.randn(2, 3, 8, 8, requires_grad=True)

    with torch.inference_mode(outer_inference):
        if use_profile:
            result = profile_inference(
                model,
                sample,
                device=torch.device("cpu"),
                n_warmup=1,
                n_measure=2,
                count_flops=True,
            )
            gflops = result.flops.gflops
            assert result.flops.status == "measured"
        else:
            gflops = _count_gflops(model, sample)
        assert torch.is_inference_mode_enabled() is outer_inference

    assert gflops == pytest.approx(16 / 1e9)
    assert len(calls) == (4 if use_profile else 1)
    assert calls[-1] == (False, False, False, False)
    assert sample.requires_grad
    assert sample.grad is None
    assert {module: module.training for module in model.modules()} == original_modes
    for name, parameter in model.named_parameters():
        assert parameter is original_parameters[name]
        assert parameter.requires_grad is original_flags[name]
        assert parameter.grad is original_grads[name]


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("forward not implemented"),
        AssertionError("Expected gradient function to be set"),
        AttributeError("next_functions"),
        RuntimeError("model failed"),
    ],
)
def test_count_gflops_propagates_errors_without_retry(error) -> None:
    calls = []

    class BrokenForward(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            raise error

    with pytest.raises(type(error)) as exc:
        _count_gflops(BrokenForward(), torch.rand(1, 3, 16, 16))
    assert exc.value is error
    assert len(calls) == 1


def test_measure_profile_cpu_returns_dict() -> None:
    model = nn.Linear(4, 2)
    sample_batch = torch.rand(4, 4)
    with (
        mock.patch.object(torch.cuda, "is_available") as available,
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
    ):
        result = measure_profile(
            model,
            sample_batch=sample_batch,
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=2,
        )
    for query in (available, current, count):
        query.assert_not_called()

    assert isinstance(result, dict)
    assert "params_m" in result
    assert "throughput_samples_per_sec" in result
    assert result["params_m"] is not None
    assert result["params_m"] > 0
    assert result["gflops"] == pytest.approx(16 / 1e9)


def test_measure_profile_does_not_hide_counter_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported_counter(
        model: nn.Module,
        sample: torch.Tensor,
        *,
        device: torch.device | None = None,
        precision: str = "float32",
    ) -> float:
        raise NotImplementedError("unsupported counter operation")

    monkeypatch.setattr(model_profile, "_count_gflops", unsupported_counter)
    with pytest.raises(NotImplementedError, match="unsupported counter operation"):
        measure_profile(nn.Linear(4, 2), torch.rand(2, 4), torch.device("cpu"), 0, 1)


def test_profile_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device("cuda"))


def test_profile_validates_tensor_devices_and_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="sample_batch is on"):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device("cuda:0"))
    mixed = nn.Linear(2, 2)
    mixed.register_buffer("other", torch.ones(1, device="meta"))
    with pytest.raises(ValueError, match="one device"):
        profile_inference(mixed, torch.ones(1, 2), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="model tensors are on"):
        profile_inference(
            nn.Linear(2, 2, device="meta"), torch.ones(1, 2), device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="float32"):
        profile_inference(
            nn.Linear(2, 2).double(), torch.ones(1, 2).double(), device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="precision"):
        model_profile._precision_context(torch.device("cpu"), "invalid")


def test_mocked_cuda_timing_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Identity(nn.Module):
        def forward(self, sample: Any) -> Any:
            events.append("forward")
            return sample

    def synchronize(device: torch.device) -> None:
        assert device == torch.device("cuda:1")
        events.append("sync")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 2 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 3 * 1024**3)
    sample = cast(
        torch.Tensor,
        SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.float32, shape=(2, 3), ndim=2),
    )
    result = profile_inference(
        Identity(), sample, device=torch.device("cuda"), n_warmup=1, n_measure=1
    )
    assert events == ["forward", "sync", "reset", "forward", "sync"]
    assert result.peak_gpu_mem_gb == 2
    assert result.reserved_gpu_mem_gb == 3
    assert result.device == "cuda:1"


def test_profile_rejects_out_of_range_cuda_before_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="CUDA index 2"):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device("cuda:2"))


@pytest.mark.parametrize("device", ["meta", "mps"])
def test_profile_still_restricts_measurement_devices(device: str) -> None:
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device(device))


def test_cpu_budget_and_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([0.0, 0.2, 0.2, 0.7, 0.7, 0.7])
    monkeypatch.setattr(model_profile.time, "perf_counter", lambda: next(timestamps))
    result = measure_cpu_throughput(
        nn.Identity(),
        torch.ones(2, 2),
        ProfileTiming(batch_size=2, n_warmup=0, n_measure=10),
        time_budget_s=0.5,
    )
    assert result["throughput_samples_per_sec_cpu"] == pytest.approx(4)
    assert result["latency_ms_per_batch_p50_cpu"] == pytest.approx(500)


def test_cpu_budget_can_expire_during_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(model_profile.time, "perf_counter", lambda: next(timestamps))
    model = nn.Identity()
    result = measure_cpu_throughput(
        model,
        torch.ones(2, 2),
        ProfileTiming(batch_size=2, n_warmup=1, n_measure=1),
        time_budget_s=1,
    )
    assert all(value is None for value in result.values())
    assert model.training


def test_cpu_zero_budget_disables_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(2, 2)

    def unexpected_forward(sample: torch.Tensor) -> torch.Tensor:
        pytest.fail("A zero profiling budget must not execute the model")

    monkeypatch.setattr(model, "forward", unexpected_forward)
    result = measure_cpu_throughput(
        model,
        torch.ones(4, 2),
        ProfileTiming(batch_size=2, n_warmup=1, n_measure=1),
        time_budget_s=0.0,
    )
    assert result == {
        "throughput_samples_per_sec_cpu": None,
        "latency_ms_per_batch_p50_cpu": None,
    }
    assert model.training


def test_cpu_invalid_settings() -> None:
    with pytest.raises(ValueError, match="non-empty batch"):
        profile_inference(nn.Identity(), torch.empty(0, 2), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="sample batch"):
        measure_cpu_throughput(
            nn.Identity(),
            torch.ones(1, 2),
            ProfileTiming(batch_size=2, n_warmup=0, n_measure=1),
            time_budget_s=1,
        )
    with pytest.raises(ValueError, match="positive"):
        measure_cpu_throughput(
            nn.Identity(),
            torch.ones(1, 2),
            ProfileTiming(batch_size=1, n_warmup=0, n_measure=0),
            time_budget_s=1,
        )


@pytest.mark.parametrize("outer_inference", [False, True])
def test_count_gflops_uses_one_frozen_no_grad_forward(*, outer_inference: bool) -> None:
    calls: list[int] = []

    class InferenceAttrErrorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(x.shape[0])
            if torch.is_inference_mode_enabled():
                raise AttributeError("next_functions")
            assert not torch.is_grad_enabled()
            assert not x.requires_grad
            assert not any(parameter.requires_grad for parameter in self.parameters())
            return self.conv(x)

    with torch.inference_mode(outer_inference):
        gflops = _count_gflops(InferenceAttrErrorModel(), torch.rand(2, 3, 8, 8))
        assert torch.is_inference_mode_enabled() is outer_inference
    assert calls == [1]
    assert gflops == pytest.approx(2 * 4 * 3 * 8 * 8 / 1e9)


def test_profile_records_fixed_batch_and_precision() -> None:
    result = profile_inference(
        nn.Linear(4, 2),
        torch.rand(3, 4),
        device=torch.device("cpu"),
        precision="float32",
        n_warmup=1,
        n_measure=2,
        count_flops=False,
    )
    assert result.batch_size == 3
    assert result.precision == "float32"
    assert result.input_shape == (3, 4)
    assert result.flops.status == "disabled"
    assert result.flops.gflops is None
    assert result.throughput_samples_per_sec > 0
    assert result.latency_ms_per_batch_p50 > 0
    assert result.peak_gpu_mem_gb is None
    assert result.reserved_gpu_mem_gb is None
    values = result.as_dict()
    assert values["gflops_status"] == "disabled"
    assert values["gflops_convention"] == "one multiply-add is two operations"
    assert values["gflops_coverage"] == "registered operators only; total coverage unverified"


def test_profile_propagates_unsupported_model_flops() -> None:
    class Unsupported(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls > 1:
                raise NotImplementedError("operator is unsupported")
            return x

    model = Unsupported()
    with pytest.raises(NotImplementedError, match="unsupported"):
        profile_inference(
            model,
            torch.rand(1, 2),
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=1,
            count_flops=True,
        )
    assert model.calls == 2
    assert model.training


def test_profile_propagates_real_flop_execution_failure() -> None:
    class Broken(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("real model failure")

    with pytest.raises(RuntimeError, match="real model failure"):
        profile_inference(
            Broken(),
            torch.rand(1, 2),
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=1,
            count_flops=True,
        )


def test_profile_does_not_mutate_autograd_hooks() -> None:
    import torch.autograd.graph as autograd_graph
    import torch.utils.module_tracker as module_tracker

    graph_hook = autograd_graph.register_multi_grad_hook
    tracker_hook = module_tracker.register_multi_grad_hook
    profile_inference(
        nn.Linear(4, 2),
        torch.rand(1, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=1,
        count_flops=False,
    )
    assert autograd_graph.register_multi_grad_hook is graph_hook
    assert module_tracker.register_multi_grad_hook is tracker_hook


@pytest.mark.parametrize("count_flops", [False, True])
def test_profile_restores_mixed_training_modes(*, count_flops: bool) -> None:
    model = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))
    model.train()
    model[1].eval()
    profile_inference(
        model,
        torch.rand(2, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=1,
        count_flops=count_flops,
    )
    assert model.training is True
    assert model[1].training is False


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_profile_accepts_autocast_precisions(precision: str) -> None:
    model = nn.Linear(4, 2)
    dtypes: list[torch.dtype] = []
    with model.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype)):
        result = profile_inference(
            model,
            torch.rand(1, 4),
            device=torch.device("cpu"),
            precision=precision,
            n_warmup=0,
            n_measure=1,
            count_flops=True,
        )
    assert result.precision == precision
    assert dtypes == [getattr(torch, precision)] * 2
    assert result.flops.gflops == pytest.approx(16 / 1e9)


def test_profile_rejects_invalid_settings() -> None:
    model = nn.Linear(2, 2)
    sample = torch.rand(1, 2)
    with pytest.raises(ValueError, match="n_warmup"):
        profile_inference(model, sample, device=torch.device("cpu"), n_warmup=-1)
    with pytest.raises(ValueError, match="n_measure"):
        profile_inference(model, sample, device=torch.device("cpu"), n_measure=0)
    with pytest.raises(ValueError, match="precision"):
        profile_inference(model, sample, device=torch.device("cpu"), precision="int8")


def test_cpu_profile_restores_model_and_reports_metrics() -> None:
    model = nn.Linear(2, 2)
    metrics = measure_cpu_throughput(
        model,
        torch.rand(2, 2),
        ProfileTiming(batch_size=2, n_warmup=0, n_measure=1),
        time_budget_s=1,
    )
    assert metrics["throughput_samples_per_sec_cpu"] is not None
    assert metrics["latency_ms_per_batch_p50_cpu"] is not None
    assert next(model.parameters()).device == torch.device("cpu")
    assert model.training
