r"""Compare data-loader batch sizes and worker counts for one model and dataset.

Report samples per second, peak GPU memory, and the fastest successful setting.

Usage::

    python scripts/tune_dataloader.py \\
        --model terratorch/prithvi_eo_v2_300 \\
        --dataset m-bigearthnet \\
        --bands all \\
        --root data/classification_v1.0_wds \\
        --batch-sizes 64,128,256,512 \\
        --num-workers 4,8,16,32

Uses GeoBench V1 tar shards, whose reader lets each worker open its own files.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from geo_eval_harness.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import ModelConfig
from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets._v1_webdataset import GeoBenchv1Sharded
from geo_eval_harness.datasets.base import BandSpec
from geo_eval_harness.devices import resolve_device
from geo_eval_harness.models.interface import BenchModel

logger = logging.getLogger(__name__)


def _build_dataset(name: str, bands: str, root: Path) -> GeoBenchv1Sharded:
    cls = get_bench_dataset_class(name)
    bench = cls()
    sel = (
        tuple(bench.rgb_bands)
        if bands == "rgb"
        else None
        if bands == "all"
        else tuple(bands.split(","))
    )
    band_names = [b.source_name for b in bench.select_band_specs(sel)]
    return GeoBenchv1Sharded(
        root=str(root),
        dataset_name=name,
        split="train",
        partition="default",
        bands=tuple(band_names),
    )


def _build_model(
    model_name: str,
    bands_list: list[BandSpec],
    dataset_name: str = "m-bigearthnet",
    train_dataset: Dataset | None = None,
) -> BenchModel:
    cfg, preset = resolve_run_config(
        RunConfig(model=ModelConfig(name=model_name), datasets=[dataset_name]), dataset_name
    )
    runtime_options = {}
    if preset.kwargs.get("mode") == "empirical":
        runtime_options["dataset"] = train_dataset
    return build_model(
        preset,
        bands=bands_list,
        normalization=NORMALIZATIONS[cfg.input.normalization],
        **runtime_options,
    )


def _bench(
    model: BenchModel, dl: DataLoader, device: torch.device, max_batches: int
) -> tuple[float, float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            x = batch["image"].to(device, non_blocking=True)
            _ = model.forward_patch_features(x)
            n += x.shape[0]
            if i + 1 >= max_batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return n / dt, peak_gb, dt


def main() -> None:
    """Sweep batch sizes and worker counts, reporting the fastest successful setting."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="e.g. terratorch/prithvi_eo_v2_300")
    p.add_argument("--dataset", default="m-bigearthnet")
    p.add_argument("--bands", default="all")
    p.add_argument("--root", default="data/classification_v1.0_wds")
    p.add_argument("--batch-sizes", default="64,128,256,512")
    p.add_argument("--num-workers", default="4,8,16,32")
    p.add_argument("--max-batches", type=int, default=20, help="cap per cell to stay quick")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    bs_list = [int(x) for x in args.batch_sizes.split(",")]
    nw_list = [int(x) for x in args.num_workers.split(",")]

    dataset = _build_dataset(args.dataset, args.bands, Path(args.root))
    bench_cls = get_bench_dataset_class(args.dataset)()
    sel = (
        tuple(bench_cls.rgb_bands)
        if args.bands == "rgb"
        else None
        if args.bands == "all"
        else tuple(args.bands.split(","))
    )
    bands_list = bench_cls.select_band_specs(sel)
    model = _build_model(args.model, bands_list, args.dataset, dataset).to(device).eval()

    logger.info("Tuning %s on %s/%s (%s)", args.model, args.dataset, args.bands, device)
    logger.info("%6s %6s %12s %10s %10s", "bs", "nw", "samples/sec", "peak GB", "wall (s)")

    results = []
    for bs in bs_list:
        for nw in nw_list:
            loader = DataLoader(
                dataset,
                batch_size=bs,
                num_workers=nw,
                shuffle=False,
                pin_memory=device.type == "cuda",
                persistent_workers=nw > 0,
            )
            try:
                sps, peak, dt = _bench(model, loader, device, args.max_batches)
                results.append((sps, bs, nw, peak, dt))
                logger.info("%6d %6d %12.1f %10.2f %10.2f", bs, nw, sps, peak, dt)
            except (
                torch.cuda.OutOfMemoryError
            ) as exc:  # allow-except: oversized batches are expected during tuning
                torch.cuda.empty_cache()
                logger.warning("batch_size=%d num_workers=%d: %s", bs, nw, exc)

    if not results:
        raise RuntimeError("No dataloader configuration completed successfully.")
    best = max(results, key=lambda r: r[0])
    logger.info(
        "Best: batch_size=%d num_workers=%d, %.1f samples/sec (%.2f GB peak)",
        best[1],
        best[2],
        best[0],
        best[3],
    )


if __name__ == "__main__":
    main()
