"""Measure the inference cost of a benchmark model.

The profiler deliberately measures one fixed batch. Data loading, host
transfers, and automatic batch-size searches are outside its scope.
"""

import contextlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from statistics import median

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from geo_eval_harness.devices import resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlopMeasurement:
    """Result of an optional FLOP measurement.

    Counts cover registered operators only; unregistered operators may make
    the reported value a lower bound.
    """

    gflops: float | None
    status: str
    reason: str | None = None
    convention: str = "one multiply-add is two operations"
    counter: str = "torch.utils.flop_counter.FlopCounterMode"
    coverage: str = "registered operators only; total coverage unverified"


@dataclass(frozen=True)
class ProfileResult:
    """Measurements for one explicit model and input configuration."""

    params_m: float
    throughput_samples_per_sec: float
    latency_ms_per_batch_p50: float
    peak_gpu_mem_gb: float | None
    reserved_gpu_mem_gb: float | None
    flops: FlopMeasurement
    batch_size: int
    device: str
    precision: str
    warmup: int
    measurements: int
    input_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Return a flat representation suitable for result storage."""
        return {
            "params_m": self.params_m,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "latency_ms_per_batch_p50": self.latency_ms_per_batch_p50,
            "peak_gpu_mem_gb": self.peak_gpu_mem_gb,
            "reserved_gpu_mem_gb": self.reserved_gpu_mem_gb,
            "gflops": self.flops.gflops,
            "gflops_status": self.flops.status,
            "gflops_reason": self.flops.reason,
            "gflops_counter": self.flops.counter,
            "gflops_convention": self.flops.convention,
            "gflops_coverage": self.flops.coverage,
            "batch_size": self.batch_size,
            "device": self.device,
            "precision": self.precision,
            "warmup": self.warmup,
            "measurements": self.measurements,
            "input_shape": str(self.input_shape),
        }


@dataclass(frozen=True, kw_only=True)
class ProfileTiming:
    """Batch size and iteration counts for a timing measurement."""

    batch_size: int
    n_warmup: int
    n_measure: int


def _count_params(model: nn.Module) -> float:
    """Return the number of model parameters in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def _count_gflops(
    model: nn.Module,
    sample: torch.Tensor | list[torch.Tensor],
    *,
    device: torch.device | None = None,
    precision: str = "float32",
) -> float:
    """Count one frozen forward pass without changing the model's parameters."""
    inputs = (
        sample[:1].detach()
        if isinstance(sample, torch.Tensor)
        else [feature[:1].detach() for feature in sample]
    )
    input_device = sample.device if isinstance(sample, torch.Tensor) else sample[0].device
    actual_device = input_device if device is None else device
    parameters = {name: parameter.detach() for name, parameter in model.named_parameters()}
    with (
        torch.inference_mode(False),
        torch.no_grad(),
        _precision_context(actual_device, precision),
        FlopCounterMode(display=False) as counter,
    ):
        torch.func.functional_call(model, parameters, (inputs,))
    return float(counter.get_total_flops()) / 1e9


def _precision_context(
    device: torch.device, precision: str
) -> contextlib.AbstractContextManager[object]:
    if precision == "float32":
        return contextlib.nullcontext()
    if precision == "float16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("precision must be 'float32', 'float16', or 'bfloat16'")


@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for module, training in states.items():
            module.train(training)


def _validate_profile_inputs(
    model: nn.Module, sample_batch: torch.Tensor, device: torch.device, precision: str
) -> torch.device:
    """Validate float32 inputs before autocast and resolve the CUDA device index."""
    if sample_batch.ndim == 0 or sample_batch.shape[0] <= 0:
        raise ValueError("sample_batch must have a non-empty batch dimension")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    device = resolve_device(device)
    if sample_batch.device != device:
        raise ValueError(f"sample_batch is on {sample_batch.device}, expected {device}")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be 'float32', 'float16', or 'bfloat16'")
    model_tensors = (*model.parameters(), *model.buffers())
    if len({tensor.device for tensor in model_tensors}) > 1:
        raise ValueError("model parameters and buffers must use one device")
    if model_tensors and model_tensors[0].device != device:
        raise ValueError(f"model tensors are on {model_tensors[0].device}, expected {device}")
    floating_dtypes = {tensor.dtype for tensor in model_tensors if tensor.is_floating_point()}
    if floating_dtypes - {torch.float32} or sample_batch.dtype != torch.float32:
        raise ValueError("profiling requires float32 inputs and model tensors before autocast")
    return device


def profile_inference(  # noqa: PLR0913 - public profiling options
    model: nn.Module,
    sample_batch: torch.Tensor,
    *,
    device: torch.device,
    precision: str = "float32",
    n_warmup: int = 3,
    n_measure: int = 20,
    count_flops: bool = False,
) -> ProfileResult:
    """Measure one fixed inference configuration.

    Args:
        model: Model to evaluate.
        sample_batch: Input batch already placed on *device*.
        device: Device used for inference.
        precision: Inference precision policy.
        n_warmup: Number of untimed warmup passes.
        n_measure: Number of timed passes.
        count_flops: Whether to run the optional FLOP counter.

    Returns:
        Parameters, timing, memory, and FLOP measurements.

    Raises:
        ValueError: If iteration counts, batch shape, device, or precision is
            invalid.
    """
    if n_warmup < 0 or n_measure <= 0:
        raise ValueError("n_warmup must be non-negative and n_measure must be positive")
    device = _validate_profile_inputs(model, sample_batch, device, precision)
    batch_size = sample_batch.shape[0]
    is_cuda = device.type == "cuda"
    with _evaluation_mode(model), torch.inference_mode(), _precision_context(device, precision):
        for _ in range(n_warmup):
            model(sample_batch)
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        started = time.perf_counter()
        for _ in range(n_measure):
            pass_started = time.perf_counter()
            model(sample_batch)
            if is_cuda:
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - pass_started) * 1000)
        elapsed = time.perf_counter() - started
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if is_cuda else None
        reserved_gpu_mem_gb = torch.cuda.memory_reserved(device) / 1024**3 if is_cuda else None
        if count_flops:
            flops = FlopMeasurement(
                _count_gflops(model, sample_batch, device=device, precision=precision), "measured"
            )
        else:
            flops = FlopMeasurement(None, "disabled", "FLOP counting was not requested")
    return ProfileResult(
        params_m=_count_params(model),
        throughput_samples_per_sec=batch_size * n_measure / elapsed,
        latency_ms_per_batch_p50=float(median(timings)),
        peak_gpu_mem_gb=peak_gpu_mem_gb,
        reserved_gpu_mem_gb=reserved_gpu_mem_gb,
        flops=flops,
        batch_size=batch_size,
        device=str(device),
        precision=precision,
        warmup=n_warmup,
        measurements=n_measure,
        input_shape=tuple(sample_batch.shape),
    )


def measure_profile(
    model: nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
    n_warmup: int = 3,
    n_measure: int = 20,
) -> dict[str, float | None]:
    """Return historical metrics, including the always-enabled FLOP count."""
    result = profile_inference(
        model, sample_batch, device=device, n_warmup=n_warmup, n_measure=n_measure, count_flops=True
    )
    return {
        "throughput_samples_per_sec": result.throughput_samples_per_sec,
        "latency_ms_per_batch_p50": result.latency_ms_per_batch_p50,
        "peak_gpu_mem_gb": result.peak_gpu_mem_gb,
        "reserved_gpu_mem_gb": result.reserved_gpu_mem_gb,
        "params_m": result.params_m,
        "gflops": result.flops.gflops,
    }


def measure_cpu_throughput(
    model: nn.Module,
    sample: torch.Tensor,
    timing: ProfileTiming,
    time_budget_s: float,
) -> dict[str, float | None]:
    """Return ``*_cpu`` throughput and latency within a wall-time budget.

    Move the model to CPU for measurement, then restore its original device.
    Return ``None`` metrics with a warning if warmup exceeds ``time_budget_s``.
    Otherwise, stop after the first completed batch that exceeds the budget.
    """
    if timing.batch_size <= 0 or timing.batch_size > sample.shape[0]:
        raise ValueError("batch_size must be within the sample batch")
    if timing.n_warmup < 0 or timing.n_measure <= 0 or time_budget_s < 0:
        raise ValueError("CPU profiling settings must be positive")
    none_result: dict[str, float | None] = {
        "throughput_samples_per_sec_cpu": None,
        "latency_ms_per_batch_p50_cpu": None,
    }
    if time_budget_s == 0:
        return none_result
    cpu_dev = torch.device("cpu")
    # Parameter-free baselines use the sample's device as their restoration target.
    first_param = next(iter(model.parameters()), None)
    orig_dev = first_param.device if first_param is not None else sample.device
    cpu_sample = sample[: timing.batch_size].detach().to(cpu_dev)
    model.to(cpu_dev)
    try:
        t0 = time.perf_counter()
        with _evaluation_mode(model), torch.inference_mode():
            for _ in range(timing.n_warmup):
                model(cpu_sample)
                if time.perf_counter() - t0 >= time_budget_s:
                    logger.warning(
                        "[profile] CPU warmup exceeded %ss budget on %s; skipping CPU throughput.",
                        time_budget_s,
                        type(model).__name__,
                    )
                    return none_result
            per_batch_ms: list[float] = []
            t_loop = time.perf_counter()
            for _ in range(timing.n_measure):
                tb = time.perf_counter()
                model(cpu_sample)
                per_batch_ms.append((time.perf_counter() - tb) * 1000.0)
                if time.perf_counter() - t0 >= time_budget_s:
                    break
            elapsed = time.perf_counter() - t_loop
        if not per_batch_ms:
            return none_result
        return {
            "throughput_samples_per_sec_cpu": (timing.batch_size * len(per_batch_ms)) / elapsed,
            "latency_ms_per_batch_p50_cpu": median(per_batch_ms),
        }
    finally:
        model.to(orig_dev)
