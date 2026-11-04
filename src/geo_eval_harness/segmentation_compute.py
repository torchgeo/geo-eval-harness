"""Per-sample FLOPs and parameters of segmentation probes, split by component.

A *cell* is a partial run configuration plus one dataset. It is resolved exactly as
``geo-eval-harness run`` resolves it (model preset, dataset overrides, explicit settings), so
the measured network is the one a run with those settings trains and evaluates. Inputs are
synthetic tensors with the dataset's resolved band count; no data is loaded.

Counting convention: :class:`torch.utils.flop_counter.FlopCounterMode` formulas, one multiply-add
is two FLOPs, matrix products, convolutions and attention only. Elementwise operations,
normalization, pooling, interpolation and the optimizer update are not counted. Every operator
the counter has no formula for is recorded; matrix-like ones mark the row as a lower bound.

See ``docs/user/segmentation-compute.md`` for definitions and the training-budget formula.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils import flop_counter
from torch.utils.flop_counter import FlopCounterMode

from geo_eval_harness.config.presets import NORMALIZATIONS, ModelPreset, resolve_run_config
from geo_eval_harness.config.run import RunConfig

logger = logging.getLogger(__name__)

aten = torch.ops.aten

# Bump when a definition below changes the meaning of a compute_key or a measured column.
COMPUTE_SCHEMA = 1

ADAPTATIONS = ("frozen", "lora", "full")

# Name tokens of unregistered operators that suggest matrix work; one makes the count a lower bound.
_HEAVY_TOKENS = frozenset(
    {
        "matmul",
        "einsum",
        "linear",
        "gemm",
        "attention",
        "attn",
        "transformer",
        "deform",
        "fft",
        "cdist",
        "mv",
        "addmv",
        "dot",
        "flash",
        "triton",
        "xformers",
    }
)
_COUNTED_NAMESPACES = {"aten", "prims", "prim"}
_ATTENTION_OP = re.compile(r"attention|attn")
_CHECKPOINT_SUFFIXES = (".pth", ".pt", ".ckpt", ".safetensors", ".bin")


# Formulas for operators that torch's registry lacks but benchmark models execute.


def _sdpa_cpu_flop(query_shape, key_shape, value_shape, *_args, **_kwargs) -> int:
    """CPU flash attention has the same cost as the registered CUDA kernels."""
    return flop_counter.sdpa_flop_count(query_shape, key_shape, value_shape)


def _sdpa_cpu_backward_flop(
    grad_out_shape, query_shape, key_shape, value_shape, *_args, **_kwargs
) -> int:
    """Backward of CPU flash attention, as the registered CUDA kernels."""
    return flop_counter.sdpa_backward_flop_count(
        grad_out_shape, query_shape, key_shape, value_shape
    )


def _addmm_activation_flop(_bias_shape, a_shape, b_shape, *_args, **_kwargs) -> int:
    """Fused ``addmm`` + activation (used by fused linear-GELU paths)."""
    return flop_counter.mm_flop(a_shape, b_shape)


def _projection_flop(tokens: int, in_features: int, out_features: int) -> int:
    return 2 * tokens * in_features * out_features


def _attention_block_flop(
    batch: int, q_len: int, k_len: int, embed_dim: int, num_heads: int
) -> int:
    """Q/K/V projections, scaled dot-product attention and the output projection."""
    head_dim = embed_dim // num_heads
    projections = _projection_flop(batch * q_len, embed_dim, embed_dim) * 2 + 2 * (
        _projection_flop(batch * k_len, embed_dim, embed_dim)
    )
    attention = flop_counter.sdpa_flop_count(
        (batch, num_heads, q_len, head_dim),
        (batch, num_heads, k_len, head_dim),
        (batch, num_heads, k_len, head_dim),
    )
    return projections + attention


def _native_mha_flop(query_shape, key_shape, *args, **_kwargs) -> int:
    """``nn.MultiheadAttention``'s inference fast path.

    Arguments follow ``aten._native_multi_head_attention(query, key, value, embed_dim,
    num_head, qkv_weight, ...)``.
    """
    embed_dim, num_heads = args[1], args[2]
    return _attention_block_flop(query_shape[0], query_shape[1], key_shape[1], embed_dim, num_heads)


def _transformer_layer_flop(src_shape, embed_dim, num_heads, *args, **_kwargs) -> int:
    """``nn.TransformerEncoderLayer``'s inference fast path: attention block plus feed-forward.

    ``args[11]`` is the first feed-forward weight in ``aten._transformer_encoder_layer_fwd``.
    """
    batch, length, _ = src_shape
    hidden = args[11][0]
    attention = _attention_block_flop(batch, length, length, embed_dim, num_heads)
    return attention + 2 * _projection_flop(batch * length, embed_dim, hidden)


EXTRA_FLOP_FORMULAS: dict[Any, Any] = {
    aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flop,
    aten._scaled_dot_product_flash_attention_for_cpu_backward: _sdpa_cpu_backward_flop,
    aten._addmm_activation: _addmm_activation_flop,
    aten._native_multi_head_attention: _native_mha_flop,
    aten._transformer_encoder_layer_fwd: _transformer_layer_flop,
}


class ComponentFlopCounter(FlopCounterMode):
    """Count FLOPs per phase and per probe component, and record operators without a formula.

    Components are attributed by forward hooks: an operator counts towards the innermost
    registered module that is running. Backward operators run outside module forwards and count
    towards the phase only.
    """

    def __init__(self, components: Mapping[str, Sequence[nn.Module]]) -> None:
        super().__init__(display=False, custom_mapping=EXTRA_FLOP_FORMULAS)
        self.phase = "forward"
        self.by_component: dict[tuple[str, str], int] = defaultdict(int)
        self.by_op: dict[str, int] = defaultdict(int)
        self.unregistered: Counter[str] = Counter()
        self._stack: list[str] = []
        self._handles: list[Any] = []
        for name, modules in components.items():
            for module in modules:
                self._handles.append(module.register_forward_pre_hook(self._push(name)))
                self._handles.append(module.register_forward_hook(self._pop, always_call=True))

    def _push(self, name: str):
        def hook(_module, _args) -> None:
            self._stack.append(name)

        return hook

    def _pop(self, _module, _args, _output) -> None:
        if self._stack:
            self._stack.pop()

    def remove_hooks(self) -> None:
        """Detach the attribution hooks from the measured modules."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flops = int(self.flop_registry[func_packet](*args, **kwargs, out_val=out))
            component = self._stack[-1] if self._stack and self.phase == "forward" else "other"
            self.by_component[(self.phase, component)] += flops
            self.by_op[str(func_packet)] += flops
        else:
            self.unregistered[str(func_packet)] += 1
        return super()._count_flops(func_packet, out, args, kwargs)

    def phase_total(self, phase: str) -> int:
        """Return all FLOPs counted in one phase."""
        return sum(v for (p, _), v in self.by_component.items() if p == phase)

    def component(self, phase: str, name: str) -> int:
        """Return the FLOPs of one component in one phase."""
        return self.by_component.get((phase, name), 0)

    def heavy_unregistered(self) -> list[str]:
        """Operators without a formula that look like matrix work, or come from custom kernels."""
        return sorted(name for name in self.unregistered if is_heavy_op(name))

    def attention_flops(self) -> int:
        """FLOPs of attention kernels (SDPA family and fused attention)."""
        return sum(v for k, v in self.by_op.items() if _ATTENTION_OP.search(k))


def is_heavy_op(name: str) -> bool:
    """Whether an operator without a FLOP formula plausibly does matrix work.

    Custom-namespace kernels always qualify. ``upsample_bilinear2d`` does not: interpolation is
    outside the convention.
    """
    namespace, _, op = name.partition(".")
    if namespace not in _COUNTED_NAMESPACES:
        return True
    tokens = [t for t in re.split(r"[._]", op) if t]
    return any(t in _HEAVY_TOKENS or t.endswith("mm") or t.startswith("conv") for t in tokens)


# Cell resolution


@dataclass(frozen=True)
class ResolvedCell:
    """One cell after preset and dataset resolution."""

    dataset: str
    config: RunConfig
    preset: ModelPreset
    band_names: tuple[str, ...]
    num_classes: int
    label: str = ""

    @property
    def image_size(self) -> int:
        """Square input size; native-size presets are not supported."""
        size = self.config.input.image_size
        if size is None:
            raise ValueError(
                "Segmentation compute needs an explicit image_size (native size unsupported)"
            )
        return size

    @property
    def adaptation(self) -> str:
        """``frozen``, ``lora`` or ``full``."""
        seg = self.config.segmentation
        if seg.full_finetune:
            return "full"
        return "lora" if seg.lora_rank is not None else "frozen"

    def inference_identity(self) -> dict[str, Any]:
        """Settings that determine the inference network, independent of how it was trained.

        LoRA adapters are merged for evaluation, so adaptation settings are excluded.
        Normalization, dropout, learning rates and seeds do not change counted FLOPs.
        """
        seg = self.config.segmentation
        return {
            "compute_schema": COMPUTE_SCHEMA,
            "target": self.preset.target,
            "name": self.preset.name,
            "model_kwargs": _key_kwargs(self.preset.kwargs),
            "dataset": self.dataset,
            "bands": list(self.band_names),
            "image_size": self.image_size,
            "time_steps": self.config.input.time_steps,
            "num_classes": self.num_classes,
            "decoder": seg.head,
            "neck": seg.neck,
            "neck_norm": seg.neck_norm if seg.neck != "none" else None,
            "upsampler": seg.upsampler,
            "layers": list(seg.layers),
            "temporal_pool": seg.temporal_pool if self.config.input.time_steps else None,
            "sweep_heads": len(seg.sweep_grid),
            "feature_standardize": seg.feature_standardize,
        }

    def train_identity(self) -> dict[str, Any]:
        """Inference identity plus the adaptation settings that change training FLOPs."""
        seg = self.config.segmentation
        lora = self.adaptation == "lora"
        return {
            **self.inference_identity(),
            "adaptation": self.adaptation,
            "lora_rank": seg.lora_rank if lora else None,
            "lora_target_modules": seg.lora_target_modules if lora else None,
            "lora_last_blocks": seg.lora_last_blocks if lora else None,
        }

    def compute_key(self) -> str:
        """Stable identifier of the inference network."""
        return _digest(self.inference_identity())

    def train_key(self) -> str:
        """Stable identifier of the training-step network."""
        return _digest(self.train_identity())


def _key_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Key checkpoint paths by file name: the same weights load from different hub checkouts."""
    out: dict[str, Any] = {}
    for key, value in sorted(kwargs.items()):
        if isinstance(value, str) and value.endswith(_CHECKPOINT_SUFFIXES):
            value = os.path.basename(value)
        out[key] = value
    return out


def _digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def resolve_cell(dataset: str, run: Mapping[str, Any], *, label: str = "") -> ResolvedCell:
    """Resolve a partial run configuration for one dataset as ``geo-eval-harness run`` does."""
    from geo_eval_harness.datasets import get_bench_dataset_class

    values = {k: v for k, v in run.items() if k != "datasets"}
    # Feature caching changes how often the backbone runs, not the network; the training budget
    # takes it from each run. Default it off so image-guided cells (AnyUp) validate.
    segmentation = dict(values.get("segmentation") or {})
    segmentation.setdefault("cache_features", False)
    values["segmentation"] = segmentation
    config = RunConfig.model_validate({**values, "datasets": [dataset]}, strict=True)
    effective, preset = resolve_run_config(config, dataset)
    bench_cls = get_bench_dataset_class(dataset)
    if bench_cls.task != "segmentation":
        raise ValueError(f"{dataset!r} is not a segmentation dataset")
    specs = bench_cls.resolve_band_specs(effective.input.bands)
    if not effective.segmentation.layers:
        raise ValueError(
            f"{preset.name!r} defines no segmentation layers for {dataset!r}; set segmentation.layers"
        )
    return ResolvedCell(
        dataset=dataset,
        config=effective,
        preset=preset,
        band_names=tuple(spec.name for spec in specs),
        num_classes=int(bench_cls.num_classes),
        label=label,
    )


# Measurement


@dataclass
class ComponentCounts:
    """Per-sample GFLOPs of one forward pass, by component."""

    backbone: float
    neck: float
    decoder: float
    upsampler: float
    other: float
    attention: float
    total: float
    uncounted_ops: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> str:
        """``ok``, or ``lower_bound`` if an operator that does matrix work has no formula."""
        return "ok" if not self.uncounted_ops else "lower_bound"


def _components(probe: nn.Module) -> dict[str, list[nn.Module]]:
    """Map probe parts to their modules; inner registrations win over the decoder root."""
    from geo_eval_harness.models.segmentation_heads import NeckedHead, StandardizedHead

    parts: dict[str, list[nn.Module]] = {
        "backbone": [probe.backbone],
        "decoder": [probe.head],
    }
    head = probe.head
    if isinstance(head, StandardizedHead):
        head = head.head
    if isinstance(head, NeckedHead):
        parts["neck"] = [head.neck]
    anyup = getattr(probe, "_anyup", None)
    if anyup is not None:
        parts["upsampler"] = [anyup.upsampler]
        # The wrapped classifier runs inside the upsampler wrapper but belongs to the decoder.
        parts["decoder"].append(anyup.classifier)
    return parts


def component_params(probe: nn.Module) -> dict[str, float]:
    """Parameters in millions; the decoder excludes the neck and a frozen upsampler it holds."""

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    parts = _components(probe)
    backbone = count(probe.backbone)
    neck = sum(count(m) for m in parts.get("neck", []))
    upsampler = sum(count(m) for m in parts.get("upsampler", []))
    decoder = count(probe.head) - neck - upsampler
    return {
        "params_backbone_m": backbone / 1e6,
        "params_neck_m": neck / 1e6,
        "params_decoder_m": decoder / 1e6,
        "params_upsampler_m": upsampler / 1e6,
        "params_total_m": (backbone + neck + decoder + upsampler) / 1e6,
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()


def synthetic_input(cell: ResolvedCell, batch: int, device: torch.device) -> torch.Tensor:
    """Random images with the cell's channel count, size and optional time axis."""
    size = cell.image_size
    shape = (batch, len(cell.band_names), size, size)
    if cell.config.input.time_steps:
        shape = (batch, cell.config.input.time_steps, *shape[1:])
    return torch.randn(shape, device=device)


def _to_gflops(value: int, batch: int) -> float:
    return value / batch / 1e9


def count_inference(probe: nn.Module, x: torch.Tensor) -> ComponentCounts:
    """Count one evaluation forward (eval mode, no gradients, AMP on CUDA) per sample."""
    probe.eval()
    batch = x.shape[0]
    counter = ComponentFlopCounter(_components(probe))
    try:
        with torch.no_grad(), _autocast(x.device), counter:
            probe(x)
    finally:
        counter.remove_hooks()
    total = counter.get_total_flops()
    if counter.phase_total("forward") != total:
        raise AssertionError(
            f"component FLOPs {counter.phase_total('forward')} != counter total {total}"
        )

    def part(name: str) -> float:
        return _to_gflops(counter.component("forward", name), batch)

    return ComponentCounts(
        backbone=part("backbone"),
        neck=part("neck"),
        decoder=part("decoder"),
        upsampler=part("upsampler"),
        other=part("other"),
        attention=_to_gflops(counter.attention_flops(), batch),
        total=_to_gflops(total, batch),
        uncounted_ops=counter.heavy_unregistered(),
    )


@dataclass
class TrainStepCounts:
    """Per-sample GFLOPs of one training step (forward + backward; optimizer not counted)."""

    forward: float
    forward_backbone: float
    backward: float
    uncounted_ops: list[str]

    @property
    def step(self) -> float:
        """Forward plus backward GFLOPs."""
        return self.forward + self.backward

    @property
    def coverage(self) -> str:
        """``ok``, or ``lower_bound`` if an operator that does matrix work has no formula."""
        return "ok" if not self.uncounted_ops else "lower_bound"


def count_train_step(
    probe: nn.Module, solver: Any, x: torch.Tensor, masks: torch.Tensor
) -> TrainStepCounts:
    """Count the solver's training step: train mode, AMP on CUDA, loss, backward."""
    solver._set_train_mode()
    batch = x.shape[0]
    counter = ComponentFlopCounter(_components(probe))
    try:
        with counter:
            with _autocast(x.device):
                logits = probe(x)
                loss = solver.criterion(logits, masks)
            counter.phase = "backward"
            loss.backward()
    finally:
        counter.remove_hooks()
        probe.zero_grad(set_to_none=True)
    forward = counter.phase_total("forward")
    backward = counter.phase_total("backward")
    if forward + backward != counter.get_total_flops():
        raise AssertionError("phase FLOPs do not add up to the counter total")
    return TrainStepCounts(
        forward=_to_gflops(forward, batch),
        forward_backbone=_to_gflops(counter.component("forward", "backbone"), batch),
        backward=_to_gflops(backward, batch),
        uncounted_ops=counter.heavy_unregistered(),
    )


def build_model(cell: ResolvedCell, device: torch.device) -> nn.Module:
    """Construct the backbone exactly as ``instantiate_dataset_model`` does, without data."""
    from geo_eval_harness.config.presets import build_model as build_preset
    from geo_eval_harness.datasets import get_bench_dataset_class

    if cell.preset.kwargs.get("mode") == "empirical":
        raise ValueError("Empirical RCF needs dataset samples; it has no synthetic compute cell")
    specs = get_bench_dataset_class(cell.dataset).resolve_band_specs(cell.config.input.bands)
    model = build_preset(
        cell.preset,
        bands=specs,
        normalization=NORMALIZATIONS[cell.config.input.normalization],
    )
    return model.to(device).eval()


def _frozen_settings(cell: ResolvedCell):
    """The cell's segmentation settings with adaptation removed (the merged inference network)."""
    return cell.config.segmentation.model_copy(
        update={"lora_rank": None, "full_finetune": False, "backbone_learning_rate": None}
    )


def _tap_metadata(probe: Any) -> dict[str, Any]:
    """Tap shapes from the probe's last forward (the measured input, not the 224 px dry run)."""
    features = [probe._process_feature(probe._features[name]) for name in probe.layer_names]
    hw = [(int(f.shape[-2]), int(f.shape[-1])) for f in features]
    return {
        "tap_channels": json.dumps([int(f.shape[1]) for f in features]),
        "tap_hw": json.dumps([list(pair) for pair in hw]),
        "n_tokens": hw[0][0] * hw[0][1],
    }


def _release(probe: Any) -> None:
    for hook in getattr(probe, "hooks", []):
        hook.remove()
    probe._features.clear()


def measure_inference(cell: ResolvedCell, device: torch.device) -> dict[str, Any]:
    """Count the evaluation forward of the cell's probe; adaptation does not change it."""
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    model = build_model(cell, device)
    probe, _ = build_seg_probe_and_solver(model, cell.num_classes, _frozen_settings(cell), device)
    probe.to(device)
    try:
        counts = count_inference(probe, synthetic_input(cell, 1, device))
        return {
            **_tap_metadata(probe),
            "gflops_backbone": counts.backbone,
            "gflops_neck": counts.neck,
            "gflops_decoder": counts.decoder,
            "gflops_upsampler": counts.upsampler,
            "gflops_other": counts.other,
            "gflops_total": counts.total,
            "gflops_sdpa": counts.attention,
            **component_params(probe),
            "flop_coverage": counts.coverage,
            "uncounted_ops": ";".join(counts.uncounted_ops),
        }
    finally:
        _release(probe)


def measure_train_step(
    cell: ResolvedCell, device: torch.device, batch_size: int = 2
) -> dict[str, Any]:
    """Count one training step with the cell's adaptation (frozen, LoRA, or full)."""
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    model = build_model(cell, device)
    probe, solver = build_seg_probe_and_solver(
        model, cell.num_classes, cell.config.segmentation, device
    )
    probe.to(device)
    try:
        batch = max(batch_size, int(getattr(probe.head, "min_train_batch_size", 1)))
        x = synthetic_input(cell, batch, device)
        size = cell.image_size
        masks = torch.randint(0, cell.num_classes, (batch, size, size), device=device)
        counts = count_train_step(probe, solver, x, masks)
        trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
        return {
            "gflops_train_forward": counts.forward,
            "gflops_train_forward_backbone": counts.forward_backbone,
            "gflops_train_backward": counts.backward,
            "gflops_train_step": counts.step,
            "params_trainable_m": trainable / 1e6,
            "train_batch_size": batch,
            "flop_coverage": counts.coverage,
            "uncounted_ops": ";".join(counts.uncounted_ops),
        }
    finally:
        _release(probe)


# Training budget


def training_budget(  # noqa: PLR0913 - explicit split sizes and settings
    inference: Mapping[str, Any],
    train: Mapping[str, Any],
    *,
    adaptation: str,
    n_train: int,
    n_val: int,
    n_test: int,
    epochs: int,
    cache_features: bool,
    feature_standardize: bool = False,
) -> dict[str, float]:
    """Total GFLOPs to produce one result, following ``evaluate_segmentation``.

    ``epochs`` is the number of epochs actually run (early stopping included). Every epoch
    trains on ``n_train`` samples and validates on ``n_val``; the selected head is tested once.
    A frozen backbone with ``cache_features`` encodes every split once and then trains the head
    on the cache. Optimizer updates, data loading and skipped undersized batches are not counted.
    """
    if adaptation not in ADAPTATIONS:
        raise ValueError(f"adaptation must be one of {ADAPTATIONS}, got {adaptation!r}")
    backbone = float(inference["gflops_backbone"])
    total = float(inference["gflops_total"])
    step = float(train["gflops_train_step"])
    step_backbone = float(train["gflops_train_forward_backbone"])
    evaluated = n_val * epochs + n_test
    if adaptation == "frozen" and cache_features:
        encode = backbone * (n_train + n_val + n_test)
        fit = (step - step_backbone) * n_train * epochs
        evaluate = (total - backbone) * evaluated
    else:
        encode = backbone * n_train if adaptation == "frozen" and feature_standardize else 0.0
        fit = step * n_train * epochs
        evaluate = total * evaluated
    return {
        "train_gflops_encode": encode,
        "train_gflops_fit": fit,
        "train_gflops_evaluate": evaluate,
        "train_gflops_total": encode + fit + evaluate,
    }


# Rows


COLUMNS = (
    "kind",
    "compute_key",
    "train_key",
    "arm",
    "band_config",
    "name",
    "model",
    "model_kwargs",
    "dataset",
    "bands",
    "n_channels",
    "image_size",
    "time_steps",
    "num_classes",
    "decoder",
    "neck",
    "neck_norm",
    "upsampler",
    "layers",
    "sweep_heads",
    "feature_standardize",
    "adaptation",
    "lora_rank",
    "lora_target_modules",
    "lora_last_blocks",
    "tap_channels",
    "tap_hw",
    "n_tokens",
    "gflops_backbone",
    "gflops_neck",
    "gflops_decoder",
    "gflops_upsampler",
    "gflops_other",
    "gflops_total",
    "gflops_sdpa",
    "params_backbone_m",
    "params_neck_m",
    "params_decoder_m",
    "params_upsampler_m",
    "params_total_m",
    "gflops_train_forward",
    "gflops_train_forward_backbone",
    "gflops_train_backward",
    "gflops_train_step",
    "params_trainable_m",
    "train_batch_size",
    "flop_coverage",
    "uncounted_ops",
    # Reserved for a separate timing run on reference hardware; FLOPs do not depend on it.
    "throughput_samples_per_sec",
    "latency_ms_per_batch_p50",
    "peak_gpu_mem_gb",
    "timing_batch_size",
    "timing_device",
    "compute_schema",
    "source_commit",
    "torch_version",
    "device",
    "measured_at",
)


def identity_columns(cell: ResolvedCell) -> dict[str, Any]:
    """Readable identity columns shared by every row kind."""
    ident = cell.train_identity()
    return {
        "name": ident["name"],
        "model": ident["target"],
        "model_kwargs": json.dumps(ident["model_kwargs"], sort_keys=True)
        if ident["model_kwargs"]
        else "",
        "dataset": cell.dataset,
        "bands": ",".join(cell.band_names),
        "n_channels": len(cell.band_names),
        "image_size": ident["image_size"],
        "time_steps": ident["time_steps"],
        "num_classes": ident["num_classes"],
        "decoder": ident["decoder"],
        "neck": ident["neck"],
        "neck_norm": ident["neck_norm"],
        "upsampler": ident["upsampler"],
        "layers": json.dumps(ident["layers"]),
        "sweep_heads": ident["sweep_heads"],
        "feature_standardize": ident["feature_standardize"],
        "compute_key": cell.compute_key(),
        "compute_schema": COMPUTE_SCHEMA,
    }


def provenance(device: torch.device) -> dict[str, Any]:
    """Code, library, device and time of a measurement."""
    name = str(device)
    if device.type == "cuda":
        name = f"{device} ({torch.cuda.get_device_name(device)})"
    return {
        "source_commit": source_commit(),
        "torch_version": torch.__version__,
        "device": name,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def source_commit() -> str:
    """Git commit of the imported package (``+dirty`` if modified), or ``$GEH_SOURCE_COMMIT``."""
    env = os.environ.get("GEH_SOURCE_COMMIT")
    if env:
        return env
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # allow-except: provenance is best effort
        return ""
    return f"{sha}+dirty" if dirty else sha


def iter_rows(  # noqa: PLR0913 - explicit cell options
    cell: ResolvedCell,
    device: torch.device,
    *,
    kinds: Sequence[str],
    done: set[tuple[str, str]],
    arm: str = "",
    band_config: str = "",
    train_batch_size: int = 2,
) -> Iterator[dict[str, Any]]:
    """Yield the missing inference, train-unit and canonical rows for one cell."""
    base = identity_columns(cell)
    ident = cell.train_identity()
    inference_key = ("inference", cell.compute_key())
    canonical_key = ("canonical", f"{cell.preset.name}|{band_config}|{arm}")
    want_inference = "inference" in kinds and inference_key not in done
    want_canonical = "canonical" in kinds and canonical_key not in done
    if want_inference or want_canonical:
        measured = {**base, **measure_inference(cell, device), **provenance(device)}
        if want_inference:
            yield {**measured, "kind": "inference"}
            done.add(inference_key)
        if want_canonical:
            yield {**measured, "kind": "canonical", "arm": arm, "band_config": band_config}
            done.add(canonical_key)
        _free(device)
    train_key = ("train_unit", cell.train_key())
    if "train_unit" in kinds and train_key not in done:
        measured = measure_train_step(cell, device, batch_size=train_batch_size)
        yield {
            **base,
            "kind": "train_unit",
            "train_key": cell.train_key(),
            "adaptation": ident["adaptation"],
            "lora_rank": ident["lora_rank"],
            "lora_target_modules": ident["lora_target_modules"],
            "lora_last_blocks": ident["lora_last_blocks"],
            **measured,
            **provenance(device),
        }
        done.add(train_key)
        _free(device)


def row_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """Resume key of an output row."""
    kind = row.get("kind")
    if kind == "inference":
        return ("inference", str(row["compute_key"]))
    if kind == "train_unit":
        return ("train_unit", str(row["train_key"]))
    if kind == "canonical":
        return ("canonical", f"{row['name']}|{row['band_config']}|{row['arm']}")
    return None


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


# Command


def load_manifest(path: str) -> list[Any]:
    """Read cells from JSON Lines (``.jsonl``) or a YAML list."""
    from geo_eval_harness.config.schema import load_yaml
    from geo_eval_harness.config.segmentation_compute import SegFlopsCell

    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
    else:
        items = load_yaml(path)
        if isinstance(items, dict):
            items = items.get("cells", [])
    if not isinstance(items, list):
        raise ValueError(f"{path}: a manifest is a list of cells")  # noqa: TRY004 - CLI input error
    return [SegFlopsCell.model_validate(item) for item in items]


def _load_done(path: str) -> set[tuple[str, str]]:
    import pandas as pd
    from filelock import FileLock

    if not os.path.exists(path):
        return set()
    with FileLock(f"{path}.lock"):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {key for row in frame.to_dict("records") if (key := row_key(row)) is not None}


def _canonical_cells(config: Any) -> Iterator[tuple[Any, str, str]]:
    from geo_eval_harness.config.segmentation_compute import ARMS, BAND_SELECTIONS, SegFlopsCell

    canonical = config.canonical
    for model in canonical.models:
        for band_config in canonical.band_configs:
            for arm in canonical.arms:
                run = {
                    "model": {"name": model},
                    "input": {
                        "bands": BAND_SELECTIONS[band_config],
                        "image_size": canonical.image_size,
                    },
                    "segmentation": dict(ARMS[arm]),
                }
                yield SegFlopsCell(dataset=canonical.band_source, run=run), arm, band_config


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    from geo_eval_harness.results import append_rows_atomic

    append_rows_atomic(path, [{column: row.get(column) for column in COLUMNS} for row in rows])


_CELL_ERRORS = (ValueError, RuntimeError)


@dataclass(frozen=True)
class _Work:
    """One cell to measure: manifest cells carry no arm, canonical cells do."""

    cell: Any
    kinds: list[str]
    arm: str = ""
    band_config: str = ""


def _work_items(config: Any, cells: list[Any]) -> list[_Work]:
    kinds = [k for k in config.measure.kinds if k != "canonical"]
    work = [_Work(cell, kinds) for cell in cells] if kinds else []
    if "canonical" in config.measure.kinds:
        work += [_Work(cell, ["canonical"], arm, bc) for cell, arm, bc in _canonical_cells(config)]
    return work


def _measure_work(  # noqa: PLR0913 - one call site, explicit state
    item: _Work, where: str, config: Any, path: str, device: torch.device, done: set
) -> tuple[int, bool]:
    """Measure one cell; return the rows written and whether it failed.

    A canonical arm that the model cannot build (``ValueError``) is skipped, not a failure.
    """
    cell = item.cell
    written = 0
    try:
        resolved = resolve_cell(cell.dataset, cell.run, label=cell.label)
        torch.manual_seed(config.runtime.seed)
        for row in iter_rows(
            resolved,
            device,
            kinds=item.kinds,
            done=done,
            arm=item.arm,
            band_config=item.band_config,
            train_batch_size=config.measure.train_batch_size,
        ):
            _write(path, [row])
            written += 1
            logger.info(
                "%s %s %s: total=%s GF train_step=%s GF coverage=%s",
                where,
                row["kind"],
                resolved.preset.name,
                _fmt(row.get("gflops_total")),
                _fmt(row.get("gflops_train_step")),
                row.get("flop_coverage"),
            )
    except _CELL_ERRORS as error:  # allow-except: one failed cell must not stop the sweep
        _free(device)
        if item.arm and isinstance(error, ValueError):
            logger.info("%s skipped: %s", where, error)
            return written, False
        logger.exception("%s failed", where)
        if config.runtime.fail_fast:
            raise
        return written, True
    return written, False


def main(config: Any, *, keys_only: bool = False) -> int:
    """Measure every requested row that the output does not have yet; return the failure count."""
    from geo_eval_harness.config.schema import resolve_output_path
    from geo_eval_harness.devices import resolve_device

    cells = list(config.cells)
    if config.manifest is not None:
        cells += load_manifest(config.manifest)
    if keys_only:
        return _print_keys(cells)

    device = resolve_device(config.runtime.device)
    path = resolve_output_path(
        config.output.directory, config.output.file, "compute_cost_segmentation.csv"
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    done = _load_done(path) if config.output.resume else set()
    work = _work_items(config, cells)
    failures = written = 0
    for index, item in enumerate(work):
        where = (
            f"[{index + 1}/{len(work)}] {item.cell.label} {item.cell.dataset} "
            f"{item.arm} {item.band_config}"
        ).rstrip()
        rows, failed = _measure_work(item, where, config, path, device, done)
        written += rows
        failures += failed
    logger.info("Wrote %d rows (%d failures) -> %s", written, failures, path)
    return failures


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _print_keys(cells: Sequence[Any]) -> int:
    """Print each cell's keys as JSON Lines without building models; return the failure count."""
    failures = 0
    for index, cell in enumerate(cells):
        record: dict[str, Any] = {"index": index, "label": cell.label, "dataset": cell.dataset}
        try:
            resolved = resolve_cell(cell.dataset, cell.run, label=cell.label)
        except ValueError as error:  # allow-except: report unresolvable cells and continue
            failures += 1
            record["error"] = str(error)
        else:
            record.update(
                compute_key=resolved.compute_key(),
                train_key=resolved.train_key(),
                adaptation=resolved.adaptation,
            )
        print(json.dumps(record, sort_keys=True))  # noqa: T201 - command output
    return failures
