"""Profile segmentation data loading on CPU: per-sample cost split and worker scaling.

For each dataset and band setting:

* main-process split over a fixed sample of training indices: read+decode (``dataset[i]`` without
  resizing), resize to the target size, and collating a batch;
* throughput of the standard training ``DataLoader`` (shuffled, pinned memory) for each worker
  count: time to the first batch (worker start-up) and steady-state samples/s.

Run on an otherwise idle machine; the record includes CPU count and load average. File reads may
be served from the OS page cache, which this tool cannot drop (no root), so numbers are a warm-cache
lower bound on I/O cost.

    PYTHONPATH=src:. python -P -m experiments.seg_bench.loader_profile \
        --datasets burn_scars spacenet2 --bands rgb all --workers 2 4 8 --out loader_profile.jsonl

``-P`` keeps a checkout's own ``experiments`` package in the working directory from shadowing this
one; run from the directory that holds ``data/``.
"""

import argparse
import random
import sys
import time
from pathlib import Path

from experiments.seg_bench.common import append_jsonl, machine_context


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _bands_arg(value: str) -> str | list[str]:
    return value if value in ("rgb", "all") else value.split(",")


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def profile_split(dataset_name: str, bands: str | list[str], args: argparse.Namespace) -> dict:
    """Return the main-process read/resize/collate split for one dataset."""
    from torch.utils.data import default_collate

    from geo_eval_harness.datasets.loading import _ResizeTransform, get_datasets

    raw_train, *_ = get_datasets(
        dataset_name,
        batch_size=args.batch_size,
        num_workers=0,
        return_val=True,
        image_size=None,
        bands=bands,
    )
    rng = random.Random(0)
    indices = rng.sample(range(len(raw_train)), min(args.samples, len(raw_train)))
    resize = _ResizeTransform(args.image_size, "bilinear")
    raw_train[indices[0]]  # open files / build indexes outside the timed loop
    read, resized, samples = [], [], []
    for index in indices:
        t0 = time.perf_counter()
        sample = raw_train[index]
        t1 = time.perf_counter()
        sample = resize(sample)
        t2 = time.perf_counter()
        read.append(t1 - t0)
        resized.append(t2 - t1)
        samples.append(sample)
    raw_shape = list(raw_train[indices[0]]["image"].shape)
    collate = []
    for start in range(0, len(samples) - args.batch_size + 1, args.batch_size):
        t0 = time.perf_counter()
        default_collate(samples[start : start + args.batch_size])
        collate.append(time.perf_counter() - t0)
    median = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {
        "raw_image_shape": raw_shape,
        "image_shape": list(samples[0]["image"].shape),
        "split_samples": len(indices),
        "read_ms_median": _ms(median(read)),
        "read_ms_mean": _ms(sum(read) / len(read)),
        "resize_ms_median": _ms(median(resized)),
        "collate_ms_per_batch": _ms(median(collate)) if collate else None,
    }


def profile_workers(
    dataset_name: str, bands: str | list[str], workers: int, args: argparse.Namespace
) -> dict:
    """Return start-up and steady-state throughput of the standard training loader."""
    from geo_eval_harness.datasets.loading import get_datasets

    train_ds, train_loader, val_loader, test_loader = get_datasets(
        dataset_name,
        batch_size=args.batch_size,
        num_workers=workers,
        return_val=True,
        image_size=args.image_size,
        bands=bands,
    )
    t0 = time.perf_counter()
    iterator = iter(train_loader)
    next(iterator)
    first = time.perf_counter() - t0
    batches = min(args.batches, len(train_loader) - 1)
    t1 = time.perf_counter()
    n = 0
    for _ in range(batches):
        batch = next(iterator)
        n += len(batch["image"])
    steady = time.perf_counter() - t1
    del iterator
    rate = n / steady if steady else None
    return {
        "workers": workers,
        "n_train": len(train_ds),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
        "first_batch_s": round(first, 3),
        "timed_batches": batches,
        "samples_per_s": round(rate, 2) if rate else None,
        "est_train_epoch_s": round(len(train_ds) / rate, 1) if rate else None,
        "est_all_splits_pass_s": round(
            (len(train_ds) + len(val_loader.dataset) + len(test_loader.dataset)) / rate, 1
        )
        if rate
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--bands", nargs="+", default=["rgb", "all"], type=_bands_arg)
    parser.add_argument("--workers", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--samples", type=int, default=48, help="samples for the main-process split"
    )
    parser.add_argument(
        "--batches", type=int, default=40, help="steady-state batches per worker count"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for dataset_name in args.datasets:
        for bands in args.bands:
            base = {
                "schema": "seg-loader-profile-v1",
                "dataset": dataset_name,
                "bands": bands if isinstance(bands, str) else ",".join(bands),
            }
            try:
                split = profile_split(dataset_name, bands, args)
            except Exception as error:  # noqa: BLE001 - record and continue with the next setting
                append_jsonl(
                    args.out,
                    {
                        **base,
                        **machine_context(),
                        "kind": "split",
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                _say(f"{dataset_name} {base['bands']}: split error {error}")
                continue
            append_jsonl(
                args.out, {**base, **machine_context(), "kind": "split", "status": "ok", **split}
            )
            _say(f"{dataset_name} {base['bands']}: {split}")
            for workers in args.workers:
                ctx = machine_context()
                try:
                    rec = profile_workers(dataset_name, bands, workers, args)
                    status = "ok"
                except Exception as error:  # noqa: BLE001
                    rec, status = (
                        {"workers": workers, "error": f"{type(error).__name__}: {error}"},
                        "error",
                    )
                append_jsonl(args.out, {**base, **ctx, "kind": "workers", "status": status, **rec})
                _say(f"  workers={workers}: {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
