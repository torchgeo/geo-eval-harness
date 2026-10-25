"""Memory and speed footprint of one segmentation setup: model, dataset, regime, data path.

Builds the run exactly as ``geo-eval-harness run`` does (preset, bands, normalization, 224 px,
UPerNet probe, solver) and measures instead of training to convergence:

* **step**: one training batch kept on the GPU; the solver's own train step (AMP, GradScaler,
  AdamW) timed with CUDA events and split into forward / backward / optimizer, plus FLOPs;
* **epoch slice**: ``solver.fit`` for ``--passes`` one-epoch passes over the first
  ``--epoch-batches`` training batches and ``--val-batches`` validation batches, timing how long
  the training loop waits for each batch (data wait) versus total wall time;
* **memory**: peak GPU allocated/reserved, process-tree host RSS, and the bytes held by the sample
  cache (train + val + test, as a real run keeps them).

Regimes: ``frozen`` (backbone frozen, no feature cache), ``frozen-cached`` (frozen with cached
backbone features: extraction plus cached-epoch timing), ``lora`` (all preset targets),
``lora-last<k>`` (``segmentation.lora_last_blocks=k``), ``full`` (full fine-tuning).
Data paths: ``loader`` (standard DataLoader, ``--workers``), ``disk`` (memory-mapped sample cache),
``host`` (sample cache in RAM), ``gpu`` (sample cache on the device). ``host`` and ``gpu`` are
loaded from the disk cache so a cache is decoded once per (dataset, split, bands, size). ``disk``
first reads the training split once (``disk_prewarm_s``) so the timed slice sees the steady state
of epochs after the first; ``--no-prewarm`` times cold memory-map reads instead.
``--cache-only`` builds or opens the data path and stops (used to prebuild caches serially).

An out-of-memory error is recorded as ``status=infeasible_oom`` with the phase it occurred in.
Writes one JSON record per run.

    cd <dir with data/> && PYTHONPATH=<src>:<root> python -P -m experiments.seg_bench.footprint \
        --preset timm/vit/vit_base_patch16_224 --dataset burn_scars --bands rgb --neck vit \
        --regime lora-last6 --data disk --out footprint.jsonl
"""

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

from experiments.seg_bench.common import (
    append_jsonl,
    gpu_processes,
    machine_context,
    set_threads,
    tree_rss_mib,
)

SCHEMA = "seg-footprint-v1"
REGIMES = re.compile(r"frozen|frozen-cached|lora|lora-last\d+|full")


def build_config(args: argparse.Namespace) -> dict:
    """Return the run configuration mapping for one benchmark setup."""
    seg: dict = {
        "head": "upernet",
        "neck": args.neck,
        "epochs": args.passes,
        "learning_rate": 1e-3,
        "batch_size": args.batch_size,
        "cache_features": args.regime == "frozen-cached",
    }
    if args.regime.startswith("lora"):
        seg.update(lora_rank=8, lora_alpha=16.0, backbone_learning_rate=args.backbone_lr)
        if args.regime != "lora":
            seg["lora_last_blocks"] = int(args.regime.removeprefix("lora-last"))
    elif args.regime == "full":
        seg.update(full_finetune=True, backbone_learning_rate=args.backbone_lr)
    bands = args.bands if args.bands in ("rgb", "all") else args.bands.split(",")
    return {
        "model": {"name": args.preset},
        "datasets": [args.dataset],
        "input": {"image_size": args.image_size, "bands": bands},
        "segmentation": seg,
        "runtime": {
            "device": "cuda:0",
            "batch_size": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "sample_cache": "none" if args.data == "loader" else "disk",
            "sample_cache_dir": str(args.cache_dir),
        },
    }


class TimedLoader:
    """Wrap a loader: stop after ``limit`` batches and time each wait for the next batch."""

    def __init__(self, loader, limit: int, rss_every: int = 10) -> None:
        self.loader = loader
        self.limit = min(limit, len(loader))
        self.batch_size = loader.batch_size
        self.dataset = loader.dataset
        self.rss_every = rss_every
        self.waits: list[float] = []
        self.wall = 0.0
        self.peak_rss_mib = 0.0

    def __len__(self) -> int:
        return self.limit

    def __iter__(self):
        self.waits = []
        start = time.perf_counter()
        iterator = iter(self.loader)
        for i in range(self.limit):
            t0 = time.perf_counter()
            batch = next(iterator)
            self.waits.append(time.perf_counter() - t0)
            if i % self.rss_every == 0:
                self.peak_rss_mib = max(self.peak_rss_mib, tree_rss_mib())
            yield batch
        del iterator
        self.wall = time.perf_counter() - start


def to_resident(loaders, mode: str, device):
    """Turn disk-cache loaders into host- or GPU-resident ones; return (loaders, cache bytes)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from geo_eval_harness.datasets.sample_cache import CachedSamples, DeviceBatchLoader
    from geo_eval_harness.main import LoaderSplits

    out, total = [], 0
    for loader in (loaders.train, loaders.val, loaders.test):
        disk = loader.dataset
        images = torch.from_numpy(np.ascontiguousarray(disk.images))
        masks = torch.from_numpy(np.ascontiguousarray(disk.masks))
        total += images.nbytes + masks.nbytes
        shuffle = isinstance(loader.sampler, RandomSampler)
        if mode == "gpu":
            out.append(
                DeviceBatchLoader(
                    images.to(device),
                    masks.to(device),
                    batch_size=loader.batch_size,
                    shuffle=shuffle,
                )
            )
        else:
            out.append(
                DataLoader(
                    CachedSamples(images, masks),
                    batch_size=loader.batch_size,
                    shuffle=shuffle,
                    num_workers=0,
                    pin_memory=True,
                )
            )
    return LoaderSplits(*out), total


def time_step(solver, images, masks, args, record: dict) -> None:
    """Time the solver's train step on one resident batch and count FLOPs."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    def step(events=None):
        if events:
            events[0].record()
        solver.optimizer.zero_grad()
        with torch.autocast(device_type=solver.device_type, enabled=solver.use_amp):
            loss = solver.criterion(solver.model(images), masks)
        if events:
            events[1].record()
        solver.scaler.scale(loss).backward()
        if events:
            events[2].record()
        solver.scaler.step(solver.optimizer)
        solver.scaler.update()
        if events:
            events[3].record()

    solver._set_train_mode()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    events = []
    for _ in range(args.steps):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        step(ev)
        events.append(ev)
    torch.cuda.synchronize()

    def median(i: int, j: int) -> float:
        values = sorted(e[i].elapsed_time(e[j]) for e in events)
        return values[len(values) // 2]

    totals = [e[0].elapsed_time(e[3]) for e in events]
    mean = sum(totals) / len(totals)
    std = (sum((t - mean) ** 2 for t in totals) / len(totals)) ** 0.5
    record.update(
        step_ms_median=median(0, 3),
        step_ms_cv=std / mean,
        forward_ms_median=median(0, 1),
        backward_ms_median=median(1, 2),
        optimizer_ms_median=median(2, 3),
        step_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    if args.skip_flops:
        return
    solver.optimizer.zero_grad()
    with FlopCounterMode(display=False) as forward:  # noqa: SIM117 - separate counters
        with torch.autocast(device_type=solver.device_type, enabled=solver.use_amp):
            loss = solver.criterion(solver.model(images), masks)
    with FlopCounterMode(display=False) as backward:
        loss.backward()
    solver.optimizer.zero_grad()
    record.update(
        forward_gflops=forward.get_total_flops() / 1e9,
        backward_gflops=backward.get_total_flops() / 1e9,
    )


def time_epochs(solver, loaders, args, record: dict) -> None:
    """Run ``solver.fit`` passes over an epoch slice and record wall and data-wait times."""
    if args.data == "disk" and not args.no_prewarm:
        # A real run touches every mapped page in epoch 1; later epochs read resident pages.
        start = time.perf_counter()
        for _ in loaders.train:
            pass
        record["disk_prewarm_s"] = time.perf_counter() - start
    train = TimedLoader(loaders.train, args.epoch_batches)
    val = TimedLoader(loaders.val, args.val_batches)
    passes = []
    for _ in range(args.passes):
        start = time.perf_counter()
        solver.fit(train, val, epochs=1, verbose=False)
        wall = time.perf_counter() - start
        passes.append(
            {
                "train_wall_s": train.wall,
                "train_batches": train.limit,
                "first_wait_s": train.waits[0],
                "data_wait_s": sum(train.waits),
                "data_wait_after_first_s": sum(train.waits[1:]),
                "val_wall_s": val.wall,
                "val_batches": val.limit,
                "val_data_wait_s": sum(val.waits),
                "fit_wall_s": wall,
            }
        )
    steady = passes[-1]
    per_batch = (steady["train_wall_s"] - steady["first_wait_s"]) / max(
        steady["train_batches"] - 1, 1
    )
    val_per_batch = steady["val_wall_s"] / max(steady["val_batches"], 1)
    record.update(
        pass_timings=passes,
        train_batch_s=per_batch,
        data_wait_fraction=steady["data_wait_after_first_s"]
        / max(steady["train_wall_s"] - steady["first_wait_s"], 1e-9),
        val_batch_s=val_per_batch,
        est_epoch_s=per_batch * len(loaders.train) + val_per_batch * len(loaders.val),
        epoch_peak_rss_mib=max(train.peak_rss_mib, val.peak_rss_mib),
    )


def time_cached_features(probe, solver, loaders, args, record: dict) -> None:
    """Time feature extraction and cached-feature epochs on the same epoch slice."""
    import torch

    train = TimedLoader(loaders.train, args.epoch_batches)
    val = TimedLoader(loaders.val, args.val_batches)
    t0 = time.perf_counter()
    train_cache = probe.extract_segmentation_features(train)
    t1 = time.perf_counter()
    val_cache = probe.extract_segmentation_features(val)
    t2 = time.perf_counter()
    n = len(train_cache)
    feature_bytes = sum(t.nbytes for t in train_cache.layer_tensors) + train_cache.masks.nbytes
    epochs: list[float] = []
    solver.fit_cached(
        train_cache,
        val_cache,
        batch_size=args.batch_size,
        epochs=args.passes,
        verbose=False,
        on_epoch=lambda r: epochs.append(r.epoch_seconds),
    )
    n_all = sum(len(x.dataset) for x in (loaders.train, loaders.val, loaders.test))
    record.update(
        extract_train_s=t1 - t0,
        extract_val_s=t2 - t1,
        extract_data_wait_s=sum(train.waits),
        extract_s_per_sample=(t1 - t0) / n,
        feature_bytes_per_sample=feature_bytes / n,
        est_feature_cache_gib=feature_bytes / n * n_all / 2**30,
        est_extract_all_splits_s=(t1 - t0) / n * n_all,
        cached_epoch_s=epochs,
        cached_epoch_slice_samples=n + len(val_cache),
        est_cached_epoch_s=epochs[-1]
        / (n + len(val_cache))
        * (len(loaders.train.dataset) + len(loaders.val.dataset)),
        epoch_peak_rss_mib=max(train.peak_rss_mib, val.peak_rss_mib, tree_rss_mib()),
        step_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )


def count(params) -> int:
    """Return the number of elements across parameters."""
    return sum(p.numel() for p in params)


def load_data(cfg, args: argparse.Namespace, device, record: dict):
    """Build the loaders for the requested data path; return (train dataset, loaders)."""
    import torch

    from geo_eval_harness.datasets import get_datasets
    from geo_eval_harness.main import LoaderSplits, _cache_segmentation_loaders

    t0 = time.perf_counter()
    train_ds, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=args.dataset,
        partition_name=cfg.input.partition,
        batch_size=cfg.runtime.batch_size,
        num_workers=cfg.runtime.workers,
        return_val=True,
        image_size=cfg.input.image_size,
        interpolation=cfg.input.interpolation,
        bands=cfg.input.bands,
        time_steps=cfg.input.time_steps,
    )
    loaders = LoaderSplits(train_loader, val_loader, test_loader)
    record.update(
        n_train=len(train_ds),
        n_val=len(val_loader.dataset),
        n_test=len(test_loader.dataset),
        train_batches=len(train_loader),
        val_batches_total=len(val_loader),
    )
    if args.data != "loader":
        loaders = _cache_segmentation_loaders(cfg, args.dataset, loaders)
        record["disk_cache_bytes"] = sum(
            x.dataset.nbytes for x in (loaders.train, loaders.val, loaders.test)
        )
        if args.data in ("host", "gpu"):
            before = torch.cuda.memory_allocated(device)
            loaders, record["resident_cache_bytes"] = to_resident(loaders, args.data, device)
            record["gpu_cache_allocated_mib"] = (
                torch.cuda.memory_allocated(device) - before
            ) / 2**20
    record["data_ready_s"] = time.perf_counter() - t0
    record["rss_after_data_mib"] = tree_rss_mib()

    return train_ds, loaders


def run(args: argparse.Namespace, record: dict) -> None:
    """Build the setup and fill ``record``; raises on failure."""
    import torch

    from geo_eval_harness.config.presets import resolve_run_config
    from geo_eval_harness.config.run import validate_run_config
    from geo_eval_harness.datasets import get_bench_dataset_class
    from geo_eval_harness.main import (
        instantiate_dataset_model,
    )
    from geo_eval_harness.segmentation_task import build_seg_probe_and_solver

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    record["phase"] = "config"
    cfg, model_cfg = resolve_run_config(validate_run_config(build_config(args)), args.dataset)
    record["method"] = cfg.segmentation.method
    ds_cls = get_bench_dataset_class(args.dataset)

    record["phase"] = "data"
    train_ds, loaders = load_data(cfg, args, device, record)

    if args.cache_only:
        record["status"] = "ok"
        return
    record["phase"] = "model"
    model = instantiate_dataset_model(cfg, model_cfg, ds_cls(), train_ds, device)
    record["phase"] = "probe"
    probe, solver = build_seg_probe_and_solver(model, ds_cls.num_classes, cfg.segmentation, device)
    backbone, head = probe.backbone, probe.head
    record.update(
        backbone_params=count(backbone.parameters()),
        backbone_trainable_params=count(p for p in backbone.parameters() if p.requires_grad),
        head_params=count(head.parameters()),
        head_trainable_params=count(p for p in head.parameters() if p.requires_grad),
        optimizer_params=count(p for g in solver.optimizer.param_groups for p in g["params"]),
        amp=solver.use_amp,
        model_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
        rss_after_model_mib=tree_rss_mib(),
    )

    if args.regime == "frozen-cached":
        record["phase"] = "cached_features"
        torch.cuda.reset_peak_memory_stats(device)
        time_cached_features(probe, solver, loaders, args, record)
    else:
        record["phase"] = "step"
        batch = next(iter(loaders.train))
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        record["input_shape"] = list(images.shape)
        torch.cuda.reset_peak_memory_stats(device)
        time_step(solver, images, masks, args, record)
        del images, masks, batch
        record["phase"] = "epoch"
        time_epochs(solver, loaders, args, record)
    record.update(
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
        status="ok",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bands", required=True, help="rgb, all, or a comma list")
    parser.add_argument(
        "--regime", required=True, help="frozen | frozen-cached | lora | lora-last<k> | full"
    )
    parser.add_argument("--data", choices=("loader", "disk", "host", "gpu"), required=True)
    parser.add_argument("--neck", choices=("none", "vit"), default="none")
    parser.add_argument("--backbone-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, help="Torch intra-op threads (default: CPU quota)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--epoch-batches", type=int, default=60)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument(
        "--no-prewarm", action="store_true", help="time disk reads without a full warm-up pass"
    )
    parser.add_argument(
        "--cache-only", action="store_true", help="build or open the data path, then stop"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/geo_eval_harness/samples"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not REGIMES.fullmatch(args.regime):
        parser.error(f"unknown regime {args.regime!r}")

    set_threads(args.threads)
    record: dict = {
        "schema": SCHEMA,
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "out"},
        **machine_context(),
        "gpu_processes_start": gpu_processes(),
    }
    started = time.perf_counter()
    import torch

    try:
        run(args, record)
    except torch.cuda.OutOfMemoryError as error:
        record.update(status="infeasible_oom", error=str(error).splitlines()[0])
    except Exception as error:  # noqa: BLE001 - record every failure in the results file
        record.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    if torch.cuda.is_available():
        record.setdefault("peak_allocated_mib", torch.cuda.max_memory_allocated() / 2**20)
        record.setdefault("peak_reserved_mib", torch.cuda.max_memory_reserved() / 2**20)
    record["gpu_processes_end"] = gpu_processes()
    own = os.getpid()
    record["contended"] = any(
        p["pid"] != own for p in record["gpu_processes_start"] + record["gpu_processes_end"]
    )
    record["loadavg_end"] = os.getloadavg()
    record["elapsed_s"] = time.perf_counter() - started
    append_jsonl(args.out, record)
    keys = (
        "preset",
        "dataset",
        "regime",
        "data",
        "status",
        "phase",
        "step_ms_median",
        "est_epoch_s",
        "data_wait_fraction",
        "peak_allocated_mib",
        "contended",
    )
    sys.stdout.write(" ".join(f"{k}={record.get(k)}" for k in keys) + "\n")
    sys.stdout.flush()
    # Skip interpreter teardown: loader workers and CUDA contexts can hang it.
    os._exit(0 if record["status"] in ("ok", "infeasible_oom") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
