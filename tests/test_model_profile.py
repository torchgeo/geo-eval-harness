"""Unit tests for model profiling helpers."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch
from torch import nn

from geo_eval_harness import model_profile
from geo_eval_harness.model_profile import (
    ProfileResult,
    ProfileTiming,
    _count_gflops,
    _count_params,
    measure_cpu_throughput,
    measure_profile,
    profile_inference,
)
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


def _profile_cpu(model: nn.Module, sample: torch.Tensor, **options: Any) -> ProfileResult:
    """Profile on CPU with one measured pass and no warmup unless ``options`` says otherwise."""
    settings = {"n_warmup": 0, "n_measure": 1} | options
    return profile_inference(model, sample, device=torch.device("cpu"), **settings)


def _mixed_device_model() -> nn.Module:
    model = nn.Linear(2, 2)
    model.register_buffer("other", torch.ones(1, device="meta"))
    return model


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


def test_count_gflops_counts_cpu_scaled_dot_product_attention() -> None:
    class Attention(nn.Module):
        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return nn.functional.scaled_dot_product_attention(tokens, tokens, tokens)

    heads, length, head_dim = 2, 5, 4
    gflops = _count_gflops(Attention(), torch.rand(1, heads, length, head_dim))
    assert gflops == pytest.approx(2 * 2 * heads * length * length * head_dim / 1e9)


def test_count_gflops_counts_transformer_layer_in_evaluation_mode() -> None:
    length, width, hidden = 5, 8, 16
    layer = nn.TransformerEncoderLayer(width, 2, hidden, dropout=0.0, batch_first=True).eval()
    gflops = _count_gflops(layer, torch.rand(1, length, width))
    projections = 2 * length * width * (3 * width + width + 2 * hidden)
    attention = 2 * 2 * length * length * width
    assert gflops == pytest.approx((projections + attention) / 1e9)
    assert torch.backends.mha.get_fastpath_enabled()


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
            result = _profile_cpu(model, sample, n_warmup=1, n_measure=2, count_flops=True)
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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_measure_profile_without_cuda(device: str) -> None:
    model = nn.Linear(4, 2).to(device)
    sample_batch = torch.rand(4, 4, device=device)
    with (
        mock.patch.object(torch.cuda, "is_available") as available,
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
    ):
        result = measure_profile(
            model,
            sample_batch=sample_batch,
            device=torch.device(device),
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


@pytest.mark.parametrize(
    ("device", "cuda_available", "message"),
    [
        ("cuda", False, "CUDA is unavailable"),
        ("cuda:2", True, "CUDA index 2"),
        ("cuda:0", True, "sample_batch is on"),
        ("meta", False, "device must be 'cpu', 'cuda', or 'mps'"),
        ("mps", False, "MPS is unavailable"),
    ],
)
def test_profile_rejects_invalid_devices(
    monkeypatch: pytest.MonkeyPatch, device: str, *, cuda_available: bool, message: str
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=message):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device(device))


@pytest.mark.parametrize(
    ("model", "sample", "options", "message"),
    [
        (nn.Identity, torch.empty(0, 2), {}, "non-empty batch"),
        (_mixed_device_model, torch.ones(1, 2), {}, "one device"),
        (lambda: nn.Linear(2, 2, device="meta"), torch.ones(1, 2), {}, "model tensors are on"),
        (lambda: nn.Linear(2, 2).double(), torch.ones(1, 2).double(), {}, "float32"),
        (nn.Identity, torch.ones(1, 2), {"n_warmup": -1}, "n_warmup"),
        (nn.Identity, torch.ones(1, 2), {"n_measure": 0}, "n_measure"),
        (nn.Identity, torch.ones(1, 2), {"precision": "int8"}, "precision"),
    ],
)
def test_profile_rejects_invalid_settings(
    model: Callable[[], nn.Module], sample: torch.Tensor, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _profile_cpu(model(), sample, **options)


def test_precision_context_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        model_profile._precision_context(torch.device("cpu"), "invalid")


@pytest.mark.parametrize(
    ("requested", "resolved", "has_stats"),
    [("cuda", "cuda:1", True), ("mps", "mps:0", True), ("mps", "mps:0", False)],
)
def test_accelerator_timing_boundaries(
    monkeypatch: pytest.MonkeyPatch, requested: str, resolved: str, *, has_stats: bool
) -> None:
    events: list[str] = []
    device = torch.device(resolved)

    class Identity(nn.Module):
        def forward(self, sample: Any) -> Any:
            events.append("forward")
            return sample

    if requested == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

        def synchronize(indexed_device: torch.device) -> None:
            assert indexed_device == device
            events.append("sync")

        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
        allocator = torch.cuda
    else:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("sync"))
        monkeypatch.setattr(torch, "__version__", "2.13" if has_stats else "2.12")
        allocator = torch.accelerator
    if has_stats:
        monkeypatch.setattr(
            allocator,
            "reset_peak_memory_stats",
            lambda device: events.append("reset"),
            raising=False,
        )
        monkeypatch.setattr(
            allocator, "max_memory_allocated", lambda device: 2 * 1024**3, raising=False
        )
        monkeypatch.setattr(allocator, "memory_reserved", lambda device: 3 * 1024**3, raising=False)
    sample = cast(
        torch.Tensor, SimpleNamespace(device=device, dtype=torch.float32, shape=(2, 3), ndim=2)
    )
    result = profile_inference(
        Identity(), sample, device=torch.device(requested), n_warmup=1, n_measure=2
    )
    assert events == [
        "forward",
        "sync",
        *(["reset"] if has_stats else []),
        "forward",
        "sync",
        "forward",
        "sync",
    ]
    assert result.peak_gpu_mem_gb == (2 if has_stats else None)
    assert result.reserved_gpu_mem_gb == (3 if has_stats else None)
    assert result.device == resolved


@pytest.mark.parametrize(
    ("timestamps", "timing", "budget", "forwards", "expected"),
    [
        ([0.0, 0.2, 0.2, 0.7, 0.7, 0.7], (0, 10), 0.5, 1, (4, 500)),
        ([0.0, 2.0], (1, 1), 1, 1, (None, None)),
        ([], (1, 1), 0.0, 0, (None, None)),
    ],
)
def test_cpu_budget_and_denominator(
    timestamps: list[float],
    timing: tuple[int, int],
    budget: float,
    forwards: int,
    expected: tuple[float | None, float | None],
) -> None:
    model = nn.Linear(2, 2)
    calls: list[torch.Tensor] = []
    model.register_forward_hook(lambda _module, _inputs, output: calls.append(output))
    n_warmup, n_measure = timing
    with mock.patch.object(model_profile.time, "perf_counter", side_effect=timestamps) as clock:
        result = measure_cpu_throughput(
            model,
            torch.ones(4, 2),
            ProfileTiming(batch_size=2, n_warmup=n_warmup, n_measure=n_measure),
            time_budget_s=budget,
        )
    assert (
        result["throughput_samples_per_sec_cpu"],
        result["latency_ms_per_batch_p50_cpu"],
    ) == pytest.approx(expected)
    assert clock.call_count == len(timestamps)
    assert len(calls) == forwards
    assert next(model.parameters()).device == torch.device("cpu")
    assert model.training


@pytest.mark.parametrize(
    ("timing", "message"),
    [
        (ProfileTiming(batch_size=2, n_warmup=0, n_measure=1), "sample batch"),
        (ProfileTiming(batch_size=1, n_warmup=0, n_measure=0), "positive"),
    ],
)
def test_cpu_invalid_settings(timing: ProfileTiming, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        measure_cpu_throughput(nn.Identity(), torch.ones(1, 2), timing, time_budget_s=1)


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
    result = _profile_cpu(
        nn.Linear(4, 2),
        torch.rand(3, 4),
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
    assert result.flops.convention == "one multiply-add is two operations"
    assert result.flops.coverage == "registered operators only; total coverage unverified"


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
        _profile_cpu(model, torch.rand(1, 2), count_flops=True)
    assert model.calls == 2
    assert model.training


def test_profile_propagates_real_flop_execution_failure() -> None:
    class Broken(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("real model failure")

    with pytest.raises(RuntimeError, match="real model failure"):
        _profile_cpu(Broken(), torch.rand(1, 2), count_flops=True)


def test_profile_does_not_mutate_autograd_hooks() -> None:
    import torch.autograd.graph as autograd_graph
    import torch.utils.module_tracker as module_tracker

    graph_hook = autograd_graph.register_multi_grad_hook
    tracker_hook = module_tracker.register_multi_grad_hook
    _profile_cpu(nn.Linear(4, 2), torch.rand(1, 4))
    assert autograd_graph.register_multi_grad_hook is graph_hook
    assert module_tracker.register_multi_grad_hook is tracker_hook


@pytest.mark.parametrize("count_flops", [False, True])
def test_profile_restores_mixed_training_modes(*, count_flops: bool) -> None:
    model = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))
    model.train()
    model[1].eval()
    _profile_cpu(model, torch.rand(2, 4), count_flops=count_flops, doy=torch.tensor([[1], [2]]))
    assert model.training is True
    assert model[1].training is False


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_profile_accepts_autocast_precisions(precision: str) -> None:
    model = nn.Linear(4, 2)
    dtypes: list[torch.dtype] = []
    with model.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype)):
        result = _profile_cpu(model, torch.rand(1, 4), precision=precision, count_flops=True)
    assert result.precision == precision
    assert dtypes == [getattr(torch, precision)] * 2
    assert result.flops.gflops == pytest.approx(16 / 1e9)
