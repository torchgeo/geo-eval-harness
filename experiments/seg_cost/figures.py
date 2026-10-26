"""Static, shareable figures (PNG + SVG + PDF) for the segmentation cost-benefit report."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.seg_cost import analysis, resources
from experiments.seg_cost.build import Bundle
from experiments.seg_cost.ingest import DECODER_ORDER, STRATEGY_ORDER

STRATEGY_COLORS = {
    "frozen": "#7f7f7f",
    "lora-last6": "#9ecae1",
    "lora": "#2171b5",
    "full": "#d94801",
}
DECODER_COLORS = {"neck": "#bdbdbd", "neck-ln": "#636363", "anyup": "#e7298a", "native": "#1b9e77"}
STRATEGY_MARKERS = {"frozen": "o", "lora-last6": "v", "lora": "s", "full": "D"}
MEMORY_COLORS = {
    "weights_gib": "#4d4d4d",
    "grad_opt_gib": "#d94801",
    "activation_gib": "#6baed6",
    "gpu_cache_gib": "#74c476",
}
MEMORY_LABELS = {
    "weights_gib": "weights (measured)",
    "grad_opt_gib": "gradients + AdamW state (from param. count)",
    "activation_gib": "activations, decoder, workspace (derived)",
    "gpu_cache_gib": "exact data cache on GPU (measured)",
}
PATH_ORDER = ("loader", "disk", "host", "gpu")
PATH_LABELS = {
    "loader": "original loader",
    "disk": "disk cache (mmap)",
    "host": "host-RAM cache",
    "gpu": "GPU-resident cache",
}
GPU_LINES = (16, 24, 32)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def _footer(fig: plt.Figure, bundle: Bundle, note: str) -> None:
    """Reserve a strip at the bottom of the layout and write the provenance note into it."""
    meta = bundle.meta
    text = f"{note}\nSnapshot {meta['snapshot_utc']}, analysis code {meta['analysis_commit']}."
    strip = 0.42 / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, strip, 1, 1 - strip))
    fig.text(0.005, 0.005, text, fontsize=7, color="#555555", ha="left", va="bottom")


def save(fig: plt.Figure, out: Path, name: str) -> list[Path]:
    """Write one figure as PNG, SVG and PDF; return the paths."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def fig_decision(bundle: Bundle) -> plt.Figure:
    """Median paired gain vs frozen and epoch-time ratio per dataset x strategy."""
    s = analysis.summarize_pairs(bundle.strategy_pairs, ["dataset", "strategy"])
    datasets = sorted(s["dataset"].unique())
    cols = [c for c in STRATEGY_ORDER if c != "frozen"]
    grid = np.full((len(datasets), len(cols)), np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(datasets) + 1.6))
    for i, d in enumerate(datasets):
        for j, c in enumerate(cols):
            row = s[(s["dataset"] == d) & (s["strategy"] == c)]
            if row.empty:
                ax.text(
                    j,
                    i,
                    "no accuracy\nevidence yet",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#888888",
                )
                continue
            r = row.iloc[0]
            grid[i, j] = r["gain_median_pp"]
            ax.text(
                j,
                i,
                f"{r['gain_median_pp']:+.1f} pp\n\u00d7{r['time_ratio_median']:.2f} time, "
                f"n={int(r['pairs'])}",
                ha="center",
                va="center",
                fontsize=7,
            )
    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(datasets)), datasets)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="median test mIoU gain vs frozen (pp)", shrink=0.8)
    ax.set_title("Fine-tuning strategy vs frozen: median paired gain and epoch-time ratio")
    _footer(fig, bundle, "Pairs: same model, bands, decoder, seed. Seed 0 unless repeated.")
    return fig


def _ladder_rows(caches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in caches.groupby("dataset"):
        rgb = g[g["bands"] == "rgb"]
        widest = g.sort_values("channels").iloc[-1]
        rows.append(
            {
                "dataset": d,
                "rgb_gib": float(rgb["fp32_all_gib"].iloc[0]) if not rgb.empty else np.nan,
                "widest_gib": widest["fp32_all_gib"],
                "widest_compact_gib": widest["compact16_all_gib"],
                "widest_label": f"{int(widest['channels'])} ch",
            }
        )
    return pd.DataFrame(rows).sort_values("widest_gib")


def fig_residency_ladder(bundle: Bundle) -> plt.Figure:
    """Exact cache size per dataset (RGB and widest band set) against memory budgets."""
    rows = _ladder_rows(bundle.caches)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(rows) + 1.8))
    ax.barh(
        y + 0.2,
        rows["widest_gib"],
        height=0.38,
        color="#08519c",
        label="widest band set, exact float32",
    )
    ax.barh(
        y + 0.2,
        rows["widest_compact_gib"],
        height=0.38,
        color="#9ecae1",
        label="widest band set, 16-bit (projected)",
    )
    ax.barh(y - 0.2, rows["rgb_gib"], height=0.38, color="#fd8d3c", label="RGB, exact float32")
    for yi, (_, r) in zip(y, rows.iterrows(), strict=True):
        ax.text(
            r["widest_gib"] * 1.05,
            yi + 0.2,
            f"{r['widest_gib']:.1f} GiB ({r['widest_label']})",
            va="center",
            fontsize=7,
        )
    for b in resources.HOST_BUDGETS:
        ax.axvline(b.gib, color="#444444", ls="--", lw=0.8)
        ax.text(
            b.gib,
            len(rows) - 0.3,
            f"host: {b.name}\n{b.gib:.0f} GiB",
            fontsize=6.5,
            ha="center",
            va="bottom",
        )
    for k, g in enumerate(GPU_LINES):
        ax.axvline(g - 1.5, color="#d94801", ls=":", lw=0.8)
        ax.text(g - 1.5, -0.75 - 0.45 * k, f"{g} GB GPU", fontsize=6.5, color="#d94801", ha="left")
    ax.set_xscale("log")
    ax.set_yticks(y, rows["dataset"])
    ax.set_xlabel("train + val + test cache at 224 px (GiB, log scale)")
    ax.set_ylim(-2.2, len(rows) + 0.6)
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Do the processed segmentation datasets fit in memory?")
    _footer(
        fig,
        bundle,
        "Exact = images float32 + uint8 masks (matches all measured caches). "
        "GPU lines: capacity minus 1.5 GiB headroom, before any training state.",
    )
    return fig


def fig_memory_stack(bundle: Bundle, dataset: str = "cloudsen12") -> plt.Figure:
    """Stacked GPU memory per model x strategy (footprint, batch 16, GPU-resident data)."""
    m = bundle.memory[
        (bundle.memory["data"] == "gpu")
        & (bundle.memory["batch_size"] == 16)
        & (bundle.memory["dataset"] == dataset)
    ]
    m = m.groupby(["model_key", "regime"], as_index=False)[list(MEMORY_COLORS)].median()
    models = sorted(m["model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(2.0 * len(models) + 1, 3.8), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models, strict=True):
        g = m[m["model_key"] == model].set_index("regime").reindex(STRATEGY_ORDER).dropna(how="all")
        bottom = np.zeros(len(g))
        for col, color in MEMORY_COLORS.items():
            ax.bar(g.index, g[col], bottom=bottom, color=color, label=MEMORY_LABELS[col])
            bottom += g[col].to_numpy()
        for line in GPU_LINES:
            ax.axhline(line - 1.5, color="#d94801", ls=":", lw=0.8)
        ax.set_title(model, fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    np.atleast_1d(axes)[0].set_ylabel("GiB on one GPU")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7)
    fig.suptitle(f"Where GPU memory goes: {dataset}, batch 16, data resident on the GPU")
    _footer(fig, bundle, "Dotted lines: 16/24/32 GB GPUs minus 1.5 GiB headroom.")
    return fig


def fig_fit_matrix(bundle: Bundle) -> plt.Figure:
    """Share of models whose training + exact GPU-resident data fit, per dataset x strategy."""
    f = bundle.fit[bundle.fit["kind"] == "gpu"]
    budgets = [b.name for b in resources.GPU_BUDGETS][:3]
    datasets = sorted(f["dataset"].unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(10, 0.4 * len(datasets) + 1.8), sharey=True)
    for ax, budget in zip(axes, budgets, strict=True):
        g = f[f["budget"] == budget]
        share = g.pivot_table(
            index="dataset", columns="strategy", values="fits_fp32", aggfunc="mean"
        ).reindex(index=datasets, columns=list(STRATEGY_ORDER))
        n = g.pivot_table(
            index="dataset", columns="strategy", values="fits_fp32", aggfunc="size"
        ).reindex(index=datasets, columns=list(STRATEGY_ORDER))
        ax.imshow(share.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(datasets)):
            for j in range(len(STRATEGY_ORDER)):
                v = share.iat[i, j]
                if pd.notna(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.0%}\n({int(n.iat[i, j])})",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                    )
        ax.set_xticks(range(len(STRATEGY_ORDER)), STRATEGY_ORDER, rotation=30)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.grid(False)
        ax.set_title(budget)
    fig.suptitle(
        "Hypothesis test: do exact data + training state fit on one GPU? "
        "(share of model \u00d7 decoder cells)"
    )
    _footer(
        fig,
        bundle,
        f"Full fine-tuning: measured where available, else LoRA peak \u00d7 "
        f"{bundle.full_ratio:.2f} (median measured full/LoRA ratio). 1.5 GiB headroom.",
    )
    return fig


def fig_step_decomposition(bundle: Bundle) -> plt.Figure:
    """Forward / backward / optimizer time per training step (batch 16, GPU data, 8 jobs)."""
    m = bundle.memory[
        (bundle.memory["data"] == "gpu")
        & (bundle.memory["batch_size"] == 16)
        & bundle.memory["concurrent"]
    ]
    parts = {
        "forward_ms_median": "#6baed6",
        "backward_ms_median": "#2171b5",
        "optimizer_ms_median": "#d94801",
    }
    m = m.groupby(["model_key", "regime"], as_index=False)[list(parts)].median()
    models = sorted(m["model_key"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(2.0 * len(models) + 1, 3.6), sharey=False)
    for ax, model in zip(np.atleast_1d(axes), models, strict=True):
        g = m[m["model_key"] == model].set_index("regime").reindex(STRATEGY_ORDER).dropna(how="all")
        bottom = np.zeros(len(g))
        for col, color in parts.items():
            ax.bar(g.index, g[col], bottom=bottom, color=color, label=col.replace("_ms_median", ""))
            bottom += g[col].to_numpy()
        ax.set_title(model, fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    np.atleast_1d(axes)[0].set_ylabel("ms per training step (median)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper")
    fig.suptitle("Training-step time: LoRA-all costs about as much compute as full fine-tuning")
    _footer(
        fig,
        bundle,
        "CUDA-event timing, 8 concurrent jobs on 8 V100s, AMP; all datasets "
        "pooled per model (median).",
    )
    return fig


def fig_residency_speed(bundle: Bundle) -> plt.Figure:
    """Estimated epoch time by data path, isolated vs 8 concurrent jobs (ViT-B, footprint)."""
    p = bundle.data_paths[bundle.data_paths["regime"].isin(["frozen", "lora-last6"])]
    p = p[p["model_key"] == "vit_base_patch16_224"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, conc in zip(axes, (False, True), strict=True):
        g = p[p["concurrent"] == conc]
        cells = sorted({(d, r) for d, r in zip(g["dataset"], g["regime"], strict=True)})
        x = np.arange(len(cells))
        for k, path in enumerate(PATH_ORDER):
            vals = [
                g[(g["dataset"] == d) & (g["regime"] == r) & (g["data"] == path)][
                    "est_epoch_s"
                ].median()
                for d, r in cells
            ]
            ax.bar(
                x + (k - 1.5) * 0.2,
                vals,
                width=0.2,
                label=PATH_LABELS[path],
                color=["#cb181d", "#fd8d3c", "#9ecae1", "#08519c"][k],
            )
        ax.set_xticks(x, [f"{d}\n{r}" for d, r in cells], fontsize=7)
        ax.set_yscale("log")
        ax.set_title("8 concurrent jobs (fleet load)" if conc else "one job on the machine")
    axes[0].set_ylabel("estimated epoch time (s, log)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Data residency: the exact cache matters; where it lives barely does")
    _footer(
        fig,
        bundle,
        "ViT-B, batch 16, footprint epoch slices extrapolated to one epoch (train + val).",
    )
    return fig


def fig_accuracy_gains(bundle: Bundle) -> plt.Figure:
    """Paired gain vs frozen per strategy and dataset, against the measured seed noise."""
    p = bundle.strategy_pairs
    datasets = sorted(p["dataset"].unique())
    cols = [c for c in STRATEGY_ORDER if c != "frozen"]
    noise = bundle.seed_noise.groupby("dataset")["seed_sd_pp"].median()
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(2.6 * len(datasets) + 0.6, 3.8), sharey=True, squeeze=False
    )
    rng = np.random.default_rng(0)
    for ax, d in zip(axes[0], datasets, strict=True):
        if d in noise:
            band = 2 * np.sqrt(2) * noise[d]
            ax.axhspan(
                -band, band, color="#dddddd", zorder=0, label="±2 SD of a seed-to-seed difference"
            )
        g = p[p["dataset"] == d]
        labels = []
        for j, c in enumerate(cols):
            h = g[g["strategy"] == c]
            labels.append(f"{c}\nn={len(h)}" if len(h) else f"{c}\nnot run yet")
            if h.empty:
                continue
            x = j + rng.uniform(-0.18, 0.18, len(h))
            ax.scatter(
                x,
                h["delta_pp"],
                s=12,
                c=h["decoder"].map(DECODER_COLORS),
                alpha=0.8,
                edgecolor="none",
            )
            ax.hlines(h["delta_pp"].median(), j - 0.3, j + 0.3, color="black", lw=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(cols)), labels, fontsize=7)
        ax.set_title(d + ("" if d in noise else " (single seed)"), fontsize=9)
    axes[0][0].set_ylabel("test mIoU gain vs frozen (pp)")
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=c, label=k) for k, c in DECODER_COLORS.items()
    ]
    handles += [
        plt.Line2D([], [], color="black", lw=2, label="median"),
        mpl.patches.Patch(color="#dddddd", label="seed-noise band (where repeated)"),
    ]
    fig.legend(handles=handles, loc="outside right upper", fontsize=7)
    fig.suptitle("How much does fine-tuning buy over a frozen backbone? (paired, per run)")
    _footer(
        fig,
        bundle,
        "Each point: one model \u00d7 decoder \u00d7 seed, against frozen with the same "
        "model, bands, decoder and seed.",
    )
    return fig


def fig_pareto(bundle: Bundle) -> plt.Figure:
    """Test mIoU vs training GPU-hours per job; Pareto frontier per dataset (seed 0)."""
    p = bundle.pareto
    datasets = sorted(p["dataset"].unique())
    ncol = min(3, len(datasets))
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
    for ax, d in zip(axes.flat, datasets, strict=False):
        g = p[p["dataset"] == d]
        for (strategy, dec), h in g.groupby(["strategy", "decoder"]):
            ax.scatter(
                h["train_gpu_h"],
                h["test_pp"],
                s=16,
                marker=STRATEGY_MARKERS[strategy],
                color=DECODER_COLORS[dec],
                alpha=0.8,
                edgecolor="none",
            )
        front = g[g["on_frontier"]].sort_values("train_gpu_h")
        ax.step(front["train_gpu_h"], front["test_pp"], where="post", color="black", lw=0.8)
        for _, r in front.tail(2).iterrows():
            ax.annotate(
                f"{r['model_key'].split('_')[0]} {r['strategy']}/{r['decoder']}",
                (r["train_gpu_h"], r["test_pp"]),
                fontsize=5.5,
                xytext=(3, -8),
                textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_title(d, fontsize=9)
        ax.set_xlabel("training GPU-hours per job (log)")
        ax.set_ylabel("test mIoU (%)")
    for ax in list(axes.flat)[len(datasets) :]:
        ax.set_visible(False)
    handles = [
        plt.Line2D([], [], ls="", marker=m, color="#333333", label=s)
        for s, m in STRATEGY_MARKERS.items()
    ]
    handles += [
        plt.Line2D([], [], ls="", marker="o", color=c, label=k) for k, c in DECODER_COLORS.items()
    ]
    fig.legend(handles=handles, loc="outside right upper", fontsize=7)
    fig.suptitle("Accuracy for the compute: Pareto frontier of test mIoU vs training GPU-hours")
    _footer(fig, bundle, "GPU-hours from per-epoch wall time on shared V100 hubs (observational).")
    return fig


def fig_decoder_quadrant(bundle: Bundle) -> plt.Figure:
    """Decoder variants: epoch-time ratio vs mIoU gain relative to the WS3 reference decoder."""
    d = bundle.decoder_pairs
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, metric, label in zip(
        axes,
        ("delta_pp", "delta_boundary_pp"),
        ("test mIoU gain (pp)", "boundary IoU gain (pp)"),
        strict=True,
    ):
        for (dec, strategy), h in d.groupby(["decoder", "strategy"]):
            ax.scatter(
                h["time_ratio"],
                h[metric],
                s=14,
                color=DECODER_COLORS[dec],
                marker=STRATEGY_MARKERS[strategy],
                alpha=0.7,
                edgecolor="none",
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(1, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("epoch time \u00d7 reference decoder (log)")
        ax.set_ylabel(label)
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=DECODER_COLORS[k], label=k)
        for k in ("neck", "anyup")
    ]
    handles += [
        plt.Line2D([], [], ls="", marker=m, color="#333333", label=s)
        for s, m in STRATEGY_MARKERS.items()
        if s in set(d["strategy"])
    ]
    fig.legend(handles=handles, loc="outside right upper", fontsize=7)
    fig.suptitle("Decoders vs the WS3 reference decoder: cost against gain")
    _footer(
        fig,
        bundle,
        "Reference: LayerNorm ViT neck for ViTs, native pyramid decoder for ConvNeXt/Swin. "
        "Each point: same model, dataset, strategy and seed.",
    )
    return fig


def value_table(bundle: Bundle) -> pd.DataFrame:
    """Median gain and compute multiplier per comparison (strategies and decoders)."""
    s = bundle.strategy_pairs.assign(comparison=lambda x: x["strategy"] + " vs frozen")
    names = {
        "neck": "plain neck",
        "neck-ln": "LayerNorm neck",
        "anyup": "AnyUp",
        "native": "native decoder",
    }
    d = bundle.decoder_pairs.assign(
        comparison=lambda x: (
            x["decoder"].map(names)
            + " vs "
            + x["reference"].map(names)
            + " ("
            + x["strategy"]
            + ")"
        )
    )
    both = pd.concat([s, d], ignore_index=True)
    return (
        both.groupby("comparison")
        .agg(
            pairs=("delta_pp", "size"),
            gain_pp=("delta_pp", "median"),
            gain_q25=("delta_pp", lambda v: v.quantile(0.25)),
            gain_q75=("delta_pp", lambda v: v.quantile(0.75)),
            time_ratio=("time_ratio", "median"),
            mem_ratio=("mem_ratio", "median"),
            extra_gpu_h=("extra_gpu_h", "median"),
        )
        .reset_index()
        .sort_values("gain_pp", ascending=False)
    )


def fig_value_per_cost(bundle: Bundle) -> plt.Figure:
    """Gain (median, IQR) next to the epoch-time and memory multiplier it costs."""
    v = value_table(bundle)
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 0.34 * len(v) + 1.8), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]}
    )
    y = np.arange(len(v))
    axes[0].barh(
        y,
        v["gain_pp"],
        xerr=[v["gain_pp"] - v["gain_q25"], v["gain_q75"] - v["gain_pp"]],
        color=["#2171b5" if g > 0 else "#cb181d" for g in v["gain_pp"]],
        error_kw={"lw": 0.8, "ecolor": "#555555"},
    )
    axes[0].set_xlabel("test mIoU gain (pp), median and IQR")
    axes[0].set_yticks(
        y, [f"{c} (n={n})" for c, n in zip(v["comparison"], v["pairs"], strict=True)], fontsize=7
    )
    axes[1].barh(y - 0.18, v["time_ratio"], height=0.34, color="#6a51a3", label="epoch time \u00d7")
    axes[1].barh(y + 0.18, v["mem_ratio"], height=0.34, color="#fdae6b", label="peak memory \u00d7")
    axes[1].set_xscale("log")
    axes[1].axvline(1, color="black", lw=0.6)
    axes[1].set_xlabel("cost multiplier vs reference (median, log)")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[0].axvline(0, color="black", lw=0.6)
    axes[0].invert_yaxis()
    fig.suptitle("Value for cost: accuracy gained against the compute and memory it costs")
    _footer(
        fig,
        bundle,
        "Paired cells; multipliers from observed fleet epoch times and CUDA "
        "peaks (disk-cache data path).",
    )
    return fig


def fig_budget(bundle: Bundle) -> plt.Figure:
    """Projected training GPU-hours of the WS3 matrix per strategy and data path."""
    s = bundle.scenarios.copy()
    sp = bundle.speedups[bundle.speedups["concurrent"] & (bundle.speedups["batch_size"] == 16)]
    factors = {}
    for path in ("loader", "gpu"):
        rows = sp[sp["data"].isin([path, "disk"])].pivot_table(
            index=["model_key", "dataset", "regime"], columns="data", values="est_epoch_s"
        )
        factors[path] = float((rows[path] / rows["disk"]).median()) if path in rows else np.nan
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    widths = 0.26
    ax.bar(
        x - widths,
        s["train_gpu_h"] * factors["loader"],
        widths,
        color="#cb181d",
        label=f"original loader (\u00d7 {factors['loader']:.1f}, projected)",
    )
    ax.bar(x, s["train_gpu_h"], widths, color="#fd8d3c", label="disk cache (as run)")
    ax.bar(
        x + widths,
        s["train_gpu_h"] * factors["gpu"],
        widths,
        color="#08519c",
        label=f"GPU-resident cache (\u00d7 {factors['gpu']:.2f}, projected)",
    )
    for xi, (_, r) in zip(x, s.iterrows(), strict=True):
        ax.text(
            xi,
            r["train_gpu_h"] * 1.03,
            f"{r['train_gpu_h']:.0f} h\n{r['measured_cells']}/{r['cells']} measured",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x, s["strategy"])
    ax.set_ylabel("training GPU-hours, 20 epochs, all cells")
    ax.set_yscale("log")
    ax.set_ylim(top=float((s["train_gpu_h"] * factors["loader"]).max()) * 6)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.set_title(
        f"Programme budget: WS3 matrix ({int(s['cells'].iloc[0])} model \u00d7 dataset \u00d7 "
        "decoder cells) per strategy"
    )
    _footer(
        fig,
        bundle,
        "Unmeasured: frozen time \u00d7 fleet strategy ratio; full = LoRA time \u00d7 measured "
        "full/LoRA epoch ratio. "
        "Data-path factors: footprint, 8 concurrent jobs.",
    )
    return fig


def coverage_table(bundle: Bundle) -> pd.DataFrame:
    """Status per dataset x strategy x decoder: accuracy, resource-only, excluded or missing."""
    runs = bundle.runs
    acc = runs[(runs["role"] == "accuracy") & (runs["config"] == "base")]
    res = runs[runs["role"] == "resource"]
    keys = ["dataset", "strategy", "decoder"]
    a = acc.groupby(keys).size().rename("accuracy_runs")
    r = res.groupby(keys).size().rename("resource_runs")
    return pd.concat([a, r], axis=1).fillna(0).reset_index()


def fig_coverage(bundle: Bundle) -> plt.Figure:
    """Evidence map: what is measured for accuracy, resources only, or not at all."""
    c = coverage_table(bundle)
    datasets = sorted(set(c["dataset"]) | set(bundle.caches["dataset"]))
    decoders = [d for d in DECODER_ORDER if d in set(c["decoder"])]
    cols = [(s, d) for s in STRATEGY_ORDER for d in decoders]
    grid = np.zeros((len(datasets), len(cols)))
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 2, 0.4 * len(datasets) + 1.8))
    for _, r in c.iterrows():
        i, j = datasets.index(r["dataset"]), cols.index((r["strategy"], r["decoder"]))
        grid[i, j] = 2 if r["accuracy_runs"] else 1
        ax.text(
            j,
            i,
            f"{int(r['accuracy_runs'])}" if r["accuracy_runs"] else "gate",
            ha="center",
            va="center",
            fontsize=6.5,
        )
    cmap = mpl.colors.ListedColormap(["#f0f0f0", "#fdd49e", "#74c476"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{s}\n{d}" for s, d in cols], fontsize=6.5)
    ax.set_yticks(range(len(datasets)), datasets)
    for k in range(1, len(STRATEGY_ORDER)):
        ax.axvline(k * len(decoders) - 0.5, color="white", lw=3)
    ax.grid(False)
    ax.set_title(
        "Evidence coverage: green = completed accuracy runs (count), "
        "orange = 1-epoch gate only, grey = none yet"
    )
    _footer(
        fig,
        bundle,
        f"Excluded rows: {len(bundle.exclusions)} (tables/exclusions.csv): "
        "queued/unfinished, owner-stopped, superseded.",
    )
    return fig


FIGURES = {
    "01_decision_overview": fig_decision,
    "02_dataset_residency_ladder": fig_residency_ladder,
    "03_gpu_memory_breakdown": fig_memory_stack,
    "04_hypothesis_fit_matrix": fig_fit_matrix,
    "05_step_time_breakdown": fig_step_decomposition,
    "06_data_residency_speed": fig_residency_speed,
    "07_accuracy_gains_vs_frozen": fig_accuracy_gains,
    "08_pareto_accuracy_vs_gpu_hours": fig_pareto,
    "09_decoder_cost_vs_gain": fig_decoder_quadrant,
    "10_value_per_cost": fig_value_per_cost,
    "11_programme_budget": fig_budget,
    "12_evidence_coverage": fig_coverage,
}


def write_figures(bundle: Bundle, out: Path) -> dict[str, list[Path]]:
    """Render every figure into ``out``."""
    _style()
    return {name: save(fn(bundle), out, name) for name, fn in FIGURES.items()}
