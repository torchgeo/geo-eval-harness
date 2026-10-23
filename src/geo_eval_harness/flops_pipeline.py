"""Per-sample compute cost (GFLOPs) for backbones and classification/segmentation heads.

Measure RGB and Sentinel-2 inputs plus supported segmentation heads for one model config.
Inputs are synthetic. Band metadata defaults to the ``cloudsen12`` dataset class.
No data is loaded or downloaded.

Use the benchmark's probe builders and profiling helpers so measurements match evaluation.
"""

import logging
import os
import warnings
from collections.abc import Iterator
from datetime import UTC
from typing import Any

import pandas as pd
import torch
from filelock import FileLock
from torch import nn

from geo_eval_harness.bands import BandCompatibilityError
from geo_eval_harness.config.flops import FlopsConfig, FlopsSegmentationConfig
from geo_eval_harness.config.presets import NORMALIZATIONS, ModelPreset, build_model
from geo_eval_harness.config.schema import SegmentationConfig, resolve_output_path
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.devices import resolve_device
from geo_eval_harness.model_profile import (
    ProfileTiming,
    _count_gflops,
    _count_params,
    measure_profile,
)
from geo_eval_harness.results import append_rows_atomic
from geo_eval_harness.segmentation_probe import SegmentationProbe

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)

# TerraMind's modality selects its tokenizer, band table, and normalization.
# Group RGB/S2L2A configs by model name; band_config distinguishes their measurements.
_MODALITY_FOR_BAND_CONFIG = {"rgb": "RGB", "s2": "S2L2A"}
_TERRAMIND_MERGED_NAME = {
    "tt_terramind_v1_base": "tt_terramind_v1_base",
    "tt_terramind_v1_base_rgb": "tt_terramind_v1_base",
    "tt_terramind_v1_large": "tt_terramind_v1_large",
    "tt_terramind_v1_large_rgb": "tt_terramind_v1_large",
}


def _load_completed(path: str) -> frozenset[tuple]:
    """Return the ``(name, band_config, task, head_type)`` keys already in *path*."""
    if not os.path.exists(path):
        return frozenset()
    with FileLock(f"{path}.lock"):
        df = pd.read_csv(path)
    return frozenset(
        zip(df["name"], df["band_config"], df["task"], df["head_type"].fillna(""), strict=True)
    )


def _is_terramind(preset: ModelPreset) -> bool:
    return "TerraMind" in preset.target


def _build_model(
    preset: ModelPreset,
    band_specs: list[BandSpec],
    normalization: str,
    band_config: str,
) -> nn.Module | None:
    """Instantiate the model, skipping only explicitly incompatible band selections."""
    if _is_terramind(preset):
        # A modality/channel mismatch can silently select the wrong band table.
        declared = str(preset.kwargs.get("modality", "S2L2A"))
        expected = _MODALITY_FOR_BAND_CONFIG[band_config]
        if declared != expected:
            raise ValueError(
                f"TerraMind modality {declared!r} does not match band config "
                f"{band_config!r} ({len(band_specs)} channels, expects {expected!r}). "
                f"Measuring this pair would map through the wrong band table."
            )
    try:
        return build_model(preset, bands=band_specs, normalization=normalization)
    except BandCompatibilityError as exc:  # allow-except: Skip explicitly unsupported bands.
        logger.warning(
            "Skipping %s/%s: model is incompatible with this band config: %s",
            preset.name,
            band_config,
            exc,
        )
        return None


def _measure_backbone(
    model: nn.Module,
    n_channels: int,
    image_size: int,
    device: torch.device,
    timing: ProfileTiming,
) -> tuple[dict[str, float | None], int]:
    """Profile a synthetic batch, halving its size after CUDA runs out of memory.

    Returns the metrics and actual batch size.
    GFLOPs is per sample; throughput and memory depend on batch size.
    """
    batch_size = timing.batch_size
    while True:
        try:
            x = torch.randn(batch_size, n_channels, image_size, image_size, device=device)
            return measure_profile(
                model, x, device, n_warmup=timing.n_warmup, n_measure=timing.n_measure
            ), batch_size
        except torch.cuda.OutOfMemoryError:  # allow-except: Retry with a smaller timing batch.
            if batch_size <= 1:
                raise
            batch_size //= 2
            torch.cuda.empty_cache()
            logger.warning("CUDA OOM — retrying at timing_batch_size=%d", batch_size)


def _probe_gflops(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: torch.device,
    head: str,
    n_classes: int,
) -> tuple[float, float, int]:
    """Measure a linear or MLP probe using the backbone's output width.

    Infer ``feature_dim`` from an actual forward pass, as ``linear.py`` does from ``X.shape[1]``.
    """
    with torch.inference_mode():
        feats = model(torch.randn(1, *input_shape, device=device))
    feature_dim = int(feats.shape[1])

    if head == "mlp":
        probe: nn.Module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim, n_classes, bias=True),
        )
    elif head == "linear":
        probe = nn.Linear(feature_dim, n_classes, bias=True)
    else:
        raise ValueError(f"Unknown probe head {head!r}; expected 'linear' or 'mlp'.")
    probe.to(device).eval()

    gflops = _count_gflops(probe, torch.randn(2, feature_dim, device=device))
    return gflops, _count_params(probe), feature_dim


def _n_tokens(model: nn.Module, image_size: int) -> int | None:
    """Return the patch-token count, or ``None`` for CNN backbones.

    ``forward_patch_features`` returns pooled ``(B, D)`` embeddings, not token counts.
    Use the patch grid, ``n_tokens ~ (image_size / patch)^2``, excluding CLS and register tokens.
    """
    patch: tuple[int, int] | None = None
    for module in model.modules():
        p = getattr(module, "patch_size", None)
        if isinstance(p, (tuple, list)) and len(p) == 2:
            patch = (int(p[0]), int(p[1]))
            break
        if isinstance(p, int):
            patch = (p, p)
            break

    if patch is None or patch[0] <= 0 or patch[1] <= 0:
        return None
    # Use measured image_size, not grid_size: non-native resolutions change the attention cost.
    return (image_size // patch[0]) * (image_size // patch[1])


def _seg_head_gflops(
    probe: SegmentationProbe,
    n_channels: int,
    image_size: int,
    device: torch.device,
) -> float:
    """Count only the segmentation head on features from the real backbone.

    Run the backbone once to capture its feature shapes.
    Count ``head(features, H, W)`` separately to exclude backbone operations.
    """
    x = torch.randn(1, n_channels, image_size, image_size, device=device)
    probe._features.clear()
    with torch.inference_mode():
        _ = probe.backbone(x)
    features = [probe._process_feature(probe._features[n]) for n in probe.layer_names]

    class _HeadOnly(nn.Module):
        def __init__(self, head: nn.Module, size: tuple[int, int]) -> None:
            super().__init__()
            self.head = head
            self.size = size

        def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
            return self.head(feats, *self.size)

    wrapper = _HeadOnly(probe.head, (image_size, image_size)).to(device).eval()
    return _count_gflops(wrapper, features)


def _flops_row(
    base_meta: dict,
    model: nn.Module,
    image_size: int,
    *,
    task: str,
    head_type: str = "",
    **values: object,
) -> dict:
    """Build a compute_cost.csv row, leaving unmeasured metrics as None."""
    row = {
        **base_meta,
        "task": task,
        "head_type": head_type,
        "gflops_backbone": None,
        "gflops_head": None,
        "gflops_probe": None,
        "gflops_total": None,
        "params_backbone_m": None,
        "params_head_m": None,
        "params_probe_m": None,
        "feature_dim": None,
        "pool": getattr(model, "pool", None),
        "n_tokens": _n_tokens(model, image_size),
        "throughput_samples_per_sec": None,
        "latency_ms_per_batch_p50": None,
        "peak_gpu_mem_gb": None,
        "reserved_gpu_mem_gb": None,
        "timing_batch_size": None,
        "lenient_grad_hooks": False,
        "measured_at": _now(),
    }
    row.update(values)
    return row


def classification_row(
    cfg: FlopsConfig, model: nn.Module, base_meta: dict[str, Any], device: torch.device
) -> dict | None:
    """Measure classification cost, skipping incompatible input bands."""
    model_name = base_meta["name"]
    band_config = base_meta["band_config"]
    n_channels = base_meta["n_channels"]
    image_size = base_meta["image_size"]
    try:
        metrics, used_batch = _measure_backbone(
            model,
            n_channels,
            image_size,
            device,
            ProfileTiming(
                batch_size=cfg.timing.batch_size,
                n_warmup=cfg.timing.n_warmup,
                n_measure=cfg.timing.n_measure,
            ),
        )
        gflops_backbone = metrics["gflops"]
        gflops_probe, params_probe_m, feature_dim = _probe_gflops(
            model,
            (n_channels, image_size, image_size),
            device,
            cfg.classification.head,
            cfg.classification.num_classes,
        )
    except BandCompatibilityError as exc:  # allow-except: Skip explicitly unsupported bands.
        logger.warning(
            "Skipping %s/%s classification: model is incompatible with this "
            "band config at forward time: %s",
            model_name,
            band_config,
            exc,
        )
        return None
    row = _flops_row(
        base_meta,
        model,
        image_size,
        task="classification",
        gflops_backbone=gflops_backbone,
        gflops_probe=gflops_probe,
        gflops_total=(None if gflops_backbone is None else gflops_backbone + gflops_probe),
        params_backbone_m=metrics["params_m"],
        params_probe_m=params_probe_m,
        feature_dim=feature_dim,
        throughput_samples_per_sec=metrics["throughput_samples_per_sec"],
        latency_ms_per_batch_p50=metrics["latency_ms_per_batch_p50"],
        peak_gpu_mem_gb=metrics["peak_gpu_mem_gb"],
        reserved_gpu_mem_gb=metrics["reserved_gpu_mem_gb"],
        timing_batch_size=used_batch,
    )
    logger.info(
        "%s/%s classification: backbone=%s GF probe=%.6f GF (D=%d)",
        model_name,
        band_config,
        f"{gflops_backbone:.4f}" if gflops_backbone is not None else "None",
        gflops_probe,
        feature_dim,
    )

    return row


def _build_seg_probe(
    model: nn.Module, num_classes: int, settings: SegmentationConfig
) -> SegmentationProbe:
    """Use the benchmark probe without allocating an unused training solver."""
    return SegmentationProbe(
        backbone=model,
        layer_names=list(settings.layers),
        num_classes=num_classes,
        head_type=settings.head,
        freeze_backbone=True,
        temporal_pool=settings.temporal_pool,
    )


def segmentation_rows(
    options: FlopsSegmentationConfig,
    model: nn.Module,
    base_meta: dict,
    device: torch.device,
    completed: frozenset[tuple],
) -> Iterator[dict]:
    """Yield each completed segmentation measurement before starting the next head."""
    model_name = base_meta["name"]
    band_config = base_meta["band_config"]
    if band_config not in options.band_configs or not options.probe.layers:
        if band_config in options.band_configs and not options.probe.layers:
            logger.info(
                "No segmentation.probe.layers for %s — skipping segmentation cells", model_name
            )
        return

    for head_type in options.heads:
        seg_key = (model_name, band_config, "segmentation", head_type)
        if seg_key in completed:
            logger.info("Skip (%s, %s, %s) — already done", model_name, band_config, head_type)
            continue
        settings = options.probe.model_copy(update={"head": head_type})
        yield _segmentation_row(
            model, settings, {**base_meta, "num_classes": options.num_classes}, device
        )


def _segmentation_row(
    model: nn.Module, settings: SegmentationConfig, base_meta: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Measure one head, releasing captured features and hooks before the next cell."""
    probe = _build_seg_probe(model, base_meta["num_classes"], settings)
    probe.to(device).eval()
    n_channels, image_size = base_meta["n_channels"], base_meta["image_size"]
    try:
        gflops_head = _seg_head_gflops(probe, n_channels, image_size, device)
        row = _flops_row(
            base_meta,
            model,
            image_size,
            task="segmentation",
            head_type=settings.head,
            gflops_head=gflops_head,
            params_backbone_m=_count_params(model),
            params_head_m=_count_params(probe.head),
            feature_dim=sum(probe.channels_list),
        )
        logger.info(
            "%s/%s segmentation head=%s: head=%.4f GF (taps=%s)",
            base_meta["name"],
            base_meta["band_config"],
            settings.head,
            gflops_head,
            probe.channels_list,
        )
        return row
    finally:
        for hook in probe.hooks:
            hook.remove()
        probe._features.clear()
        del probe
        _free(device)


def main(config: FlopsConfig) -> None:
    """Measure per-sample compute cost for one model config."""
    cfg, preset = config.resolve()
    device = resolve_device(cfg.runtime.device)
    output_path = resolve_output_path(cfg.output.directory, cfg.output.file, "compute_cost.csv")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.manual_seed(cfg.runtime.seed)
    image_size = cfg.input.image_size
    normalization = NORMALIZATIONS[cfg.input.normalization]
    model_target = preset.target
    model_name = _TERRAMIND_MERGED_NAME.get(preset.name, preset.name)

    bench = get_bench_dataset_class(cfg.input.band_source)()
    band_specs_for = {
        "rgb": bench.select_band_specs(bench.rgb_bands),
        "s2": bench.select_band_specs(None),
    }

    completed = _load_completed(output_path) if cfg.output.resume else frozenset()

    n_written = 0
    n_skipped = 0

    for band_config in cfg.input.band_configs:
        if _is_terramind(preset):
            declared = str(preset.kwargs.get("modality", "S2L2A"))
            if declared != _MODALITY_FOR_BAND_CONFIG[band_config]:
                continue

        band_specs = band_specs_for[band_config]
        n_channels = len(band_specs)

        model = _build_model(preset, band_specs, normalization, band_config)
        if model is None:
            n_skipped += 1
            continue
        model.to(device).eval()

        base_meta = {
            "model": model_target,
            "name": model_name,
            "band_config": band_config,
            "n_channels": n_channels,
            "image_size": image_size,
            "num_classes": cfg.classification.num_classes,
        }

        cls_key = (model_name, band_config, "classification", "")
        if cls_key in completed:
            logger.info("Skip (%s, %s, classification) — already done", model_name, band_config)
        else:
            row = classification_row(cfg, model, base_meta, device)
            if row is None:
                n_skipped += 1
                del model
                _free(device)
                continue
            append_rows_atomic(output_path, [row])
            n_written += 1

        for row in segmentation_rows(cfg.segmentation, model, base_meta, device, completed):
            append_rows_atomic(output_path, [row])
            n_written += 1

        del model
        _free(device)

    logger.info(
        "Wrote %d rows for %s (%d cells skipped) → %s",
        n_written,
        model_name,
        n_skipped,
        output_path,
    )


def _now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
