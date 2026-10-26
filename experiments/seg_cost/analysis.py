"""Paired accuracy comparisons, seed noise, GPU-hour cost, Pareto frontiers and budget scenarios."""

import numpy as np
import pandas as pd

from experiments.seg_cost.ingest import STRATEGY_ORDER

CELL = ["dataset", "model_key", "bands", "decoder", "seed"]
Z90 = 1.645


def accuracy_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Completed full-length runs with the base recipe (no weight decay / dropout variants)."""
    acc = runs[(runs["role"] == "accuracy") & (runs["config"] == "base")].copy()
    acc["gpu_h"] = acc["wall_s"] / 3600
    acc["train_gpu_h"] = acc["train_s_total"] / 3600
    # Bootstrap 90% half-width of the test mIoU (from the result's CI), in mIoU points.
    acc["ci_half_pp"] = (acc["ci_upper"] - acc["ci_lower"]) * 100 / 2
    acc["test_pp"] = acc["test_miou"] * 100
    acc["boundary_pp"] = acc["boundary_iou"] * 100
    return acc.drop_duplicates([*CELL, "strategy"], keep="last")


def paired(acc: pd.DataFrame, factor: str, reference: str, keys: list[str]) -> pd.DataFrame:
    """Pair each run with the ``reference`` level of ``factor`` in the same cell.

    The delta CI combines both runs' bootstrap half-widths as if independent: conservative, since
    both share the test set.
    """
    ref = acc[acc[factor] == reference]
    other = acc[acc[factor] != reference]
    cols = [
        *keys,
        "test_pp",
        "boundary_pp",
        "ci_half_pp",
        "epoch_s_median",
        "peak_gib",
        "gpu_h",
        "train_gpu_h",
    ]
    m = other.merge(ref[cols], on=keys, suffixes=("", "_ref"))
    m["delta_pp"] = m["test_pp"] - m["test_pp_ref"]
    m["delta_boundary_pp"] = m["boundary_pp"] - m["boundary_pp_ref"]
    m["delta_ci_pp"] = np.sqrt(m["ci_half_pp"] ** 2 + m["ci_half_pp_ref"] ** 2)
    m["time_ratio"] = m["epoch_s_median"] / m["epoch_s_median_ref"]
    m["mem_ratio"] = m["peak_gib"] / m["peak_gib_ref"]
    m["extra_gpu_h"] = m["train_gpu_h"] - m["train_gpu_h_ref"]
    m["pp_per_extra_gpu_h"] = m["delta_pp"] / m["extra_gpu_h"].where(m["extra_gpu_h"] > 0)
    m["pp_per_extra_gib"] = m["delta_pp"] / (m["peak_gib"] - m["peak_gib_ref"]).where(
        m["peak_gib"] > m["peak_gib_ref"]
    )
    return m


def strategy_pairs(acc: pd.DataFrame) -> pd.DataFrame:
    """Every non-frozen strategy against frozen with the same model, data, decoder and seed."""
    return paired(acc, "strategy", "frozen", CELL)


def decoder_pairs(acc: pd.DataFrame) -> pd.DataFrame:
    """Decoders against the WS3 reference at the same strategy and seed.

    The reference is the LayerNorm neck for ViTs and the native pyramid decoder for ConvNeXt/Swin.
    """
    keys = ["dataset", "model_key", "bands", "strategy", "seed"]
    vit = paired(acc[acc["family"] == "vit"], "decoder", "neck-ln", keys)
    hier = paired(acc[acc["family"] != "vit"], "decoder", "native", keys)
    return pd.concat(
        [vit.assign(reference="neck-ln"), hier.assign(reference="native")], ignore_index=True
    )


def seed_noise(acc: pd.DataFrame) -> pd.DataFrame:
    """Across-seed SD of test mIoU for cells with at least two seeds."""
    keys = ["dataset", "model_key", "bands", "decoder", "strategy"]
    g = acc.groupby(keys)["test_pp"].agg(["count", "std", "mean"]).reset_index()
    return g[g["count"] >= 2].rename(columns={"std": "seed_sd_pp", "mean": "seed_mean_pp"})


def summarize_pairs(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median and quartiles of gains and cost ratios, plus how often the gain exceeds its CI."""

    def q(p: float):
        return lambda s: s.quantile(p)

    pairs = pairs.assign(
        clear_gain=pairs["delta_pp"] > pairs["delta_ci_pp"],
        clear_loss=pairs["delta_pp"] < -pairs["delta_ci_pp"],
    )
    return (
        pairs.groupby(by)
        .agg(
            pairs=("delta_pp", "size"),
            gain_median_pp=("delta_pp", "median"),
            gain_q25_pp=("delta_pp", q(0.25)),
            gain_q75_pp=("delta_pp", q(0.75)),
            clear_gains=("clear_gain", "sum"),
            clear_losses=("clear_loss", "sum"),
            time_ratio_median=("time_ratio", "median"),
            mem_ratio_median=("mem_ratio", "median"),
            extra_gpu_h_median=("extra_gpu_h", "median"),
        )
        .reset_index()
    )


def pareto_front(points: pd.DataFrame, cost: str, value: str) -> pd.DataFrame:
    """Rows not dominated by a cheaper-or-equal row with higher-or-equal value."""
    pts = points.sort_values([cost, value], ascending=[True, False])
    best, keep = -np.inf, []
    for idx, row in pts.iterrows():
        if row[value] > best:
            keep.append(idx)
            best = row[value]
    return points.loc[keep]


def pareto_table(acc: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, flag configurations on the test-mIoU vs training-GPU-hour frontier (seed 0)."""
    s0 = acc[acc["seed"] == 0].copy()
    s0["on_frontier"] = False
    for _, group in s0.groupby("dataset"):
        front = pareto_front(group, "train_gpu_h", "test_pp")
        s0.loc[front.index, "on_frontier"] = True
    return s0


def strategy_time_ratios(acc: pd.DataFrame) -> pd.DataFrame:
    """Median epoch-time ratio of each strategy to frozen, per model family and decoder."""
    p = strategy_pairs(acc)
    return p.groupby(["family", "decoder", "strategy"], as_index=False)["time_ratio"].median()


def _cell_epoch_times(
    cell: pd.Series, acc: pd.DataFrame, fleet: pd.DataFrame, fp: pd.DataFrame
) -> dict[str, tuple[float, bool]]:
    """Per strategy: (epoch seconds, measured?) for one matrix cell.

    Unmeasured strategies use frozen x the fleet's median ratio for the family and decoder. Full
    fine-tuning, which has no fleet pairs yet, uses the cell's LoRA time x the footprint's measured
    full/LoRA epoch ratio, because LoRA and full share the backbone backward pass.
    """
    same = acc[
        (acc["dataset"] == cell["dataset"])
        & (acc["model_key"] == cell["model_key"])
        & (acc["decoder"] == cell["decoder"])
        & (acc["seed"] == 0)
    ]
    times: dict[str, tuple[float, bool]] = {}
    for strategy in STRATEGY_ORDER:
        hit = same[same["strategy"] == strategy]
        if not hit.empty:
            times[strategy] = (float(hit["epoch_s_median"].iloc[0]), True)
        elif strategy == "full" and "lora" in times:
            full_over_lora = _footprint_ratio(fp, "full") / _footprint_ratio(fp, "lora")
            times[strategy] = (times["lora"][0] * full_over_lora, False)
        else:
            ratio = _ratio(fleet, fp, cell, strategy)
            times[strategy] = (cell["frozen_epoch_s"] * ratio, False)
    return times


def budget_scenarios(
    cells: pd.DataFrame, acc: pd.DataFrame, footprint_ratios: pd.DataFrame, epochs: int = 20
) -> pd.DataFrame:
    """Projected training GPU-hours of a cell matrix per strategy (see _cell_epoch_times)."""
    fleet = strategy_time_ratios(acc)
    per_cell = [_cell_epoch_times(c, acc, fleet, footprint_ratios) for _, c in cells.iterrows()]
    return pd.DataFrame(
        [
            {
                "strategy": s,
                "train_gpu_h": sum(t[s][0] for t in per_cell) * epochs / 3600,
                "cells": len(cells),
                "measured_cells": sum(t[s][1] for t in per_cell),
            }
            for s in STRATEGY_ORDER
        ]
    )


def _footprint_ratio(fp: pd.DataFrame, strategy: str) -> float:
    hit = fp[fp["strategy"] == strategy]
    return float(hit["epoch_ratio"].median()) if not hit.empty else np.nan


def _ratio(fleet: pd.DataFrame, fp: pd.DataFrame, cell: pd.Series, strategy: str) -> float:
    if strategy == "frozen":
        return 1.0
    hit = fleet[
        (fleet["family"] == cell["family"])
        & (fleet["decoder"] == cell["decoder"])
        & (fleet["strategy"] == strategy)
    ]
    if not hit.empty:
        return float(hit["time_ratio"].iloc[0])
    return _footprint_ratio(fp, strategy)


def footprint_epoch_ratios(paths: pd.DataFrame) -> pd.DataFrame:
    """Strategy / frozen estimated epoch time per footprint cell (GPU data, bs16, concurrent)."""
    g = paths[(paths["data"] == "gpu") & (paths["batch_size"] == 16) & paths["concurrent"]]
    piv = g.pivot_table(index=["model_key", "dataset"], columns="regime", values="est_epoch_s")
    out = []
    for strategy in STRATEGY_ORDER[1:]:
        if strategy in piv:
            r = (piv[strategy] / piv["frozen"]).rename("epoch_ratio").reset_index()
            out.append(r.assign(strategy=strategy))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
