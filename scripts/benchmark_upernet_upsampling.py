"""Benchmark UPerNet pyramid-pooling upsampling: reference ``F.interpolate`` vs separable matmuls.

Times a full forward + backward training step of :class:`UPerNetHead` (single feature map,
AMP float16 as in training) with the reference ``transformers`` PSP module and with the
matmul upsampling, and reports peak memory and gradient agreement.

Example::

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_upernet_upsampling.py --output bench.json
"""

import argparse
import copy
import json
import logging
import platform
import time

import torch
import torch.nn as nn
from transformers.models.upernet.modeling_upernet import UperNetPyramidPoolingModule

from geo_eval_harness.models.segmentation_heads import UPerNetHead

logger = logging.getLogger(__name__)

# (name, channels, feature height, feature width, mask size)
CASES = [
    ("small 64x64", 768, 64, 64, 256),
    ("fotw-like 154x88", 64, 154, 88, 256),
    ("kuro_siwo-like 130x130", 64, 130, 130, 224),
    ("dynamic_earthnet-like 154x154", 64, 154, 154, 512),
    ("spacenet7-like 210x211", 128, 210, 211, 512),
]


def _reference(head: UPerNetHead) -> UPerNetHead:
    stock = copy.deepcopy(head)
    stock.decoder.psp_modules.__class__ = UperNetPyramidPoolingModule
    return stock


def _step(head: nn.Module, x: torch.Tensor, y: torch.Tensor, *, amp: bool) -> torch.Tensor:
    head.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=amp):
        loss = nn.functional.cross_entropy(head([x], *y.shape[-2:]), y)
    loss.backward()
    return torch.cat([p.grad.flatten().float() for p in head.parameters() if p.grad is not None])


def _time(head: nn.Module, x: torch.Tensor, y: torch.Tensor, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        _step(head, x, y, amp=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        _step(head, x, y, amp=True)
    torch.cuda.synchronize()
    return {
        "s_per_step": (time.perf_counter() - start) / iters,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def main() -> None:
    """Run every case and log (optionally save) timings and gradient agreement."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    rows = []
    for name, channels, height, width, mask in CASES:
        torch.manual_seed(0)
        head = UPerNetHead([channels], 2, hidden_dim=256).cuda().train()
        stock = _reference(head)
        x = torch.randn(args.batch_size, channels, height, width, device="cuda")
        y = torch.randint(0, 2, (args.batch_size, mask, mask), device="cuda")
        # Small-batch float32 and AMP gradient agreement, against a float32 reference step.
        xs, ys = x[:4], y[:4]
        ref32 = _step(stock, xs, ys, amp=False)
        row = {
            "case": name,
            "input": [args.batch_size, channels, height, width],
            "mask": mask,
            "grad_rel_l2_fp32_vs_reference_fp32": _rel(_step(head, xs, ys, amp=False), ref32),
            "grad_rel_l2_amp_reference_vs_fp32": _rel(_step(stock, xs, ys, amp=True), ref32),
            "grad_rel_l2_amp_separable_vs_fp32": _rel(_step(head, xs, ys, amp=True), ref32),
        }
        for label, module in (("reference", stock), ("separable", head)):
            row[label] = _time(module, x.half(), y, args.warmup, args.iters)
            torch.cuda.empty_cache()
        row["speedup"] = row["reference"]["s_per_step"] / row["separable"]["s_per_step"]
        rows.append(row)
        logger.info(
            "%-32s reference %7.3f s (%5.1f GiB)  separable %7.3f s (%5.1f GiB)  x%5.1f  "
            "fp32 grad rel %.1e  AMP grad rel ref/sep %.4f/%.4f",
            name,
            row["reference"]["s_per_step"],
            row["reference"]["peak_gib"],
            row["separable"]["s_per_step"],
            row["separable"]["peak_gib"],
            row["speedup"],
            row["grad_rel_l2_fp32_vs_reference_fp32"],
            row["grad_rel_l2_amp_reference_vs_fp32"],
            row["grad_rel_l2_amp_separable_vs_fp32"],
        )
        del head, stock, x, y
        torch.cuda.empty_cache()

    meta = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "batch_size": args.batch_size,
        "iters": args.iters,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=2)
    logger.info(json.dumps(meta))


if __name__ == "__main__":
    main()
