"""Write the shareable Markdown report; every number is computed from the snapshot's tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.seg_cost import analysis, resources
from experiments.seg_cost.build import Bundle
from experiments.seg_cost.figures import FIGURES, value_table


def md_table(df: pd.DataFrame, digits: int = 2) -> str:
    """Render a DataFrame as a GitHub Markdown table."""

    def cell(v: object) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def dataset_fit_summary(bundle: Bundle) -> pd.DataFrame:
    """Widest band set per dataset with which host and GPU budgets its exact cache fits."""
    c = bundle.caches.sort_values("channels").groupby("dataset").tail(1)
    out = c[["dataset", "bands", "channels", "fp32_all_gib", "compact16_all_gib"]].copy()
    for b in resources.HOST_BUDGETS:
        out[f"host {b.name} ({b.gib:.0f} GiB)"] = np.where(
            out["fp32_all_gib"] <= b.gib, "yes", "no"
        )
    for b in resources.GPU_BUDGETS[:3]:
        out[f"GPU {b.name}, data only"] = np.where(
            out["fp32_all_gib"] <= b.total_gib - b.headroom_gib, "yes", "no"
        )
    return out.sort_values("fp32_all_gib")


def hypothesis_table(bundle: Bundle) -> pd.DataFrame:
    """Per GPU budget: cells where LoRA + data fit, full + data fit, and LoRA-only cells."""
    f = bundle.fit[bundle.fit["kind"] == "gpu"]
    keys = ["model_key", "dataset", "bands", "decoder", "budget"]
    lora = f[f["strategy"] == "lora"].set_index(keys)
    full = f[f["strategy"] == "full"].set_index(keys)
    both = (
        lora[["fits_fp32", "fits_model_only"]]
        .join(
            full[["fits_fp32", "fits_model_only", "source"]],
            lsuffix="_lora",
            rsuffix="_full",
            how="inner",
        )
        .reset_index()
    )
    rows = []
    for b in resources.GPU_BUDGETS:
        g = both[both["budget"] == b.name]
        rows.append(
            {
                "GPU": b.name,
                "cells": len(g),
                "LoRA + data fit": int(g["fits_fp32_lora"].sum()),
                "full + data fit": int(g["fits_fp32_full"].sum()),
                "LoRA + data fit, full + data not": int(
                    (g["fits_fp32_lora"] & ~g["fits_fp32_full"]).sum()
                ),
                "full training does not fit at all": int((~g["fits_model_only_full"]).sum()),
                "full peak measured": int(
                    g["source"].str.startswith(("measured", "derived")).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_table(bundle: Bundle) -> pd.DataFrame:
    """Footprint step time and training peak per model x strategy (bs16, GPU data, 8 jobs)."""
    m = bundle.memory[
        (bundle.memory["data"] == "gpu")
        & (bundle.memory["batch_size"] == 16)
        & bundle.memory["concurrent"]
    ]
    g = m.groupby(["model_key", "regime"])[["step_ms_median", "train_peak_gib"]].median()
    step = g["step_ms_median"].unstack()
    peak = g["train_peak_gib"].unstack()
    out = pd.DataFrame(
        {
            "frozen ms": step["frozen"],
            "LoRA-last6 ms": step["lora-last6"],
            "LoRA ms": step["lora"],
            "full ms": step["full"],
            "LoRA / full time": step["lora"] / step["full"],
            "LoRA-last6 / full time": step["lora-last6"] / step["full"],
            "LoRA GiB": peak["lora"],
            "full GiB": peak["full"],
            "LoRA / full memory": peak["lora"] / peak["full"],
        }
    )
    return out.reset_index()


def residency_table(bundle: Bundle) -> pd.DataFrame:
    """Median epoch-time multiplier of each data path relative to GPU-resident data."""
    s = bundle.speedups[bundle.speedups["batch_size"] == 16]
    t = s.groupby(["concurrent", "data"])["slowdown_vs_gpu"].median().unstack()
    t.index = t.index.map({False: "one job", True: "8 concurrent jobs"})
    return (
        t[[c for c in ("loader", "disk", "host", "gpu") if c in t]]
        .reset_index()
        .rename(columns={"concurrent": "load"})
    )


def _strategy_text(bundle: Bundle) -> str:
    s = analysis.summarize_pairs(bundle.strategy_pairs, ["strategy"])
    lines = []
    for _, r in s.iterrows():
        lines.append(
            f"- **{r['strategy']} vs frozen** ({int(r['pairs'])} pairs): median "
            f"{r['gain_median_pp']:+.2f} pp (IQR {r['gain_q25_pp']:+.2f} to "
            f"{r['gain_q75_pp']:+.2f}); {int(r['clear_gains'])} gains and "
            f"{int(r['clear_losses'])} losses exceed the combined bootstrap CI; "
            f"epoch time \u00d7{r['time_ratio_median']:.2f}, peak memory "
            f"\u00d7{r['mem_ratio_median']:.2f}."
        )
    if "full" not in set(bundle.strategy_pairs["strategy"]):
        lines.append(
            "- **full fine-tuning vs frozen:** no completed 20-epoch accuracy runs in "
            "this snapshot; its accuracy is unknown, not zero."
        )
    return "\n".join(lines)


def _key_numbers(bundle: Bundle) -> dict:
    fit = dataset_fit_summary(bundle)
    hub = resources.HOST_BUDGETS[0]
    hyp = hypothesis_table(bundle).set_index("GPU")
    comp = compute_table(bundle)
    res = residency_table(bundle).set_index("load")
    conc = res.loc["8 concurrent jobs"] if "8 concurrent jobs" in res.index else None
    return {
        "n_datasets": len(fit),
        "fit_hub": int((fit["fp32_all_gib"] <= hub.gib).sum()),
        "fit_16": int((fit["fp32_all_gib"] <= 16 - 1.5).sum()),
        "largest": fit.iloc[-1],
        "hyp32": hyp.loc["32 GB (V100)"],
        "hyp16": hyp.loc["16 GB"],
        "hyp24": hyp.loc["24 GB"],
        "lora_full_time": comp["LoRA / full time"].median(),
        "last6_full_time": comp["LoRA-last6 / full time"].median(),
        "lora_full_mem": comp["LoRA / full memory"].median(),
        "loader_x": float(conc["loader"]) if conc is not None else np.nan,
        "disk_x": float(conc["disk"]) if conc is not None else np.nan,
        "host_x": float(conc["host"]) if conc is not None else np.nan,
    }


def _hypothesis_verdict(h: pd.Series) -> str:
    share = h["LoRA + data fit, full + data not"] / max(h["cells"], 1)
    if share < 0.05:
        return "**not supported** at this GPU size"
    if share < 0.25:
        return "**supported only for a minority of cells** at this GPU size"
    return "**supported** at this GPU size"


def _answers(bundle: Bundle) -> str:
    k = _key_numbers(bundle)
    big = k["largest"]
    h16, h24, h32 = k["hyp16"], k["hyp24"], k["hyp32"]
    saved = (1 - 1 / k["disk_x"]) * 100 if k["disk_x"] else np.nan
    return f"""
1. **Do the datasets fit in memory?** With the widest band set used anywhere,
   {k["fit_hub"]}/{k["n_datasets"]} datasets' exact float32 caches (train+val+test, 224 px) fit a
   per-job share of hub RAM, and {k["fit_16"]}/{k["n_datasets"]} fit a 16 GB GPU before any
   training state. The largest is {big["dataset"]} ({big["fp32_all_gib"]:.1f} GiB,
   {int(big["channels"])} channels). The disk cache is memory-mapped, so jobs on one hub share
   page-cache copies; the per-job share is the bound for private in-process copies. (Figure 2)
2. **Does LoRA fit alongside the data while full fine-tuning does not?** On 32 GB V100s:
   {_hypothesis_verdict(h32)}. {int(h32["LoRA + data fit"])}/{int(h32["cells"])} cells fit with LoRA
   and {int(h32["full + data fit"])}/{int(h32["cells"])} with full fine-tuning;
   {int(h32["LoRA + data fit, full + data not"])} fit with LoRA only. At 24 GB:
   {int(h24["LoRA + data fit, full + data not"])} LoRA-only cells ({_hypothesis_verdict(h24)}); at
   16 GB: {int(h16["LoRA + data fit, full + data not"])} ({_hypothesis_verdict(h16)}). LoRA's
   training peak is a median {k["lora_full_mem"]:.0%} of full fine-tuning's: it saves gradients and
   AdamW state but keeps most activation memory, because gradients still flow through the
   backbone. Full peaks are measured for {int(h32["full peak measured"])} cells and projected for
   the rest. (Figures 3, 4)
3. **Is LoRA faster?** All-block LoRA's training step takes a median {k["lora_full_time"]:.2f}\u00d7
   the full fine-tuning step: its backward pass still traverses the whole backbone. LoRA on the
   last 6 blocks takes {k["last6_full_time"]:.2f}\u00d7 full. (Figure 5)
4. **Where does the speed-up come from?** Under fleet load (8 jobs per hub), relative to
   GPU-resident data: original loader \u00d7{k["loader_x"]:.1f}, disk (mmap) cache \u00d7{k["disk_x"]:.2f},
   host-RAM cache \u00d7{k["host_x"]:.2f}. Moving the existing exact cache into RAM or onto the GPU would
   save about {saved:.0f}% of epoch time; the large win (the exact cache itself) is already in
   use. (Figures 6, 11)
5. **Is fine-tuning worth it?**
{_strategy_text(bundle)}
"""


CAVEATS = """
- **Seeds:** single seed except DynamicEarthNet and SpaceNet7, where `overfit-s1` repeats the
  base recipe with seeds 1-2. Figure 7 shades ±2 SD of a seed-to-seed difference there.
- **CI approximation:** paired-delta CIs combine both runs' bootstrap intervals as if independent
  (conservative: the runs share the test set).
- **Observational timing:** fleet epoch times come from shared hubs (8 jobs per hub, mixed
  workloads, disk cache); ratios within a pair cancel most of that. The footprint benchmark gives
  controlled step/memory measurements but on 5 models and 3 datasets.
- **Projections are labelled:** full fine-tuning peaks outside the footprint grid use the median
  measured full/LoRA ratio; 16-bit caches are projected (not bit-exact after resizing);
  data-path multipliers for LoRA/full reuse those measured with frozen and LoRA-last6.
- **Exclusions:** superseded Kuro Siwo \u00d7 OlmoEarth results, owner-stopped pilot cells, unfinished
  or queued jobs. One-epoch gates contribute memory/time only.
- **Fit criterion:** GPU-resident data uses the current implementation (all splits on the GPU)
  plus 1.5 GiB headroom; keeping only train+val resident would lower the need slightly.
"""


def write_report(bundle: Bundle, out: Path) -> Path:
    """Write ``REPORT.md`` next to ``figures/`` and ``tables/``."""
    meta = bundle.meta
    figs = "\n".join(f"![{n}](figures/{n}.png)" for n in FIGURES)
    decoders = analysis.summarize_pairs(bundle.decoder_pairs, ["reference", "decoder", "strategy"])
    text = f"""# Segmentation fine-tuning: speed, memory and accuracy (cost-benefit report)

Snapshot {meta["snapshot_utc"]} · analysis code `{meta["analysis_commit"]}` · runs per campaign
{meta["campaigns"]} · {meta["excluded"]} rows excluded (`tables/exclusions.csv`).
Interactive version: `explorer.html`. Every figure is in `figures/` as PNG, SVG and PDF.

## Questions and answers
{_answers(bundle)}

## Evidence tables

### Dataset caches against memory budgets
{md_table(dataset_fit_summary(bundle))}

### The residency hypothesis per GPU size (data + training state on one GPU)
{md_table(hypothesis_table(bundle), 0)}

### Controlled compute and memory (footprint benchmark, batch 16, 8 concurrent jobs)
{md_table(compute_table(bundle))}

### Data path: epoch-time multiplier vs GPU-resident data
{md_table(residency_table(bundle))}

### Accuracy vs frozen, by dataset
{md_table(analysis.summarize_pairs(bundle.strategy_pairs, ["dataset", "strategy"]))}

### Decoders vs the WS3 reference
{md_table(decoders)}

### Value for cost
{md_table(value_table(bundle))}

### WS3 programme budget (training GPU-hours, 20 epochs, disk cache)
{md_table(bundle.scenarios, 1)}

## Figures
{figs}

## Caveats
{CAVEATS}
## Reproduce
```bash
cd ~/geh-seg
~/geo-eval-harness/.venv/bin/python -m experiments.seg_cost build \\
  --root ~/geo-eval-harness/outputs --out <dir>
```
Merge fleet campaigns first (`fleet merge <campaign dir> --no-hooks`); the tool only reads them.
"""
    path = out / "REPORT.md"
    path.write_text(text)
    return path
