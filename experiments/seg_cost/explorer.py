"""Offline interactive HTML explorer (Plotly, script inlined once) for the cost-benefit report."""

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from experiments.seg_cost import resources
from experiments.seg_cost.build import Bundle
from experiments.seg_cost.figures import DECODER_COLORS, STRATEGY_COLORS, value_table
from experiments.seg_cost.ingest import STRATEGY_ORDER

ORDER = {"strategy": list(STRATEGY_ORDER), "decoder": list(DECODER_COLORS)}


def _pareto(bundle: Bundle) -> go.Figure:
    p = bundle.accuracy.copy()
    fig = px.scatter(
        p,
        x="train_gpu_h",
        y="test_pp",
        color="strategy",
        symbol="decoder",
        facet_col="dataset",
        facet_col_wrap=3,
        log_x=True,
        category_orders=ORDER,
        color_discrete_map=STRATEGY_COLORS,
        height=650,
        hover_data=[
            "model_key",
            "bands",
            "seed",
            "boundary_pp",
            "peak_gib",
            "epoch_s_median",
            "hub",
            "campaign",
            "job_id",
        ],
        labels={"train_gpu_h": "training GPU-hours", "test_pp": "test mIoU (%)"},
        title="Test mIoU vs training GPU-hours (every completed accuracy run)",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def _ladders(bundle: Bundle) -> go.Figure:
    a = bundle.accuracy[bundle.accuracy["seed"] == 0].copy()
    a["line"] = a["model_key"] + " / " + a["decoder"]
    a["order"] = a["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)})
    a = a.sort_values("order")
    fig = px.line(
        a,
        x="strategy",
        y="test_pp",
        color="decoder",
        line_group="line",
        markers=True,
        facet_col="dataset",
        facet_col_wrap=3,
        error_y="ci_half_pp",
        category_orders=ORDER,
        color_discrete_map=DECODER_COLORS,
        height=650,
        hover_data=["model_key", "boundary_pp", "peak_gib", "epoch_s_median"],
        title="Paired accuracy ladders (seed 0): one line per model \u00d7 decoder",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def _fit(bundle: Bundle) -> go.Figure:
    f = bundle.fit[bundle.fit["kind"] == "gpu"].drop_duplicates(
        ["model_key", "dataset", "bands", "decoder", "strategy"]
    )
    fig = px.strip(
        f,
        x="dataset",
        y="need_gib",
        color="strategy",
        category_orders=ORDER,
        color_discrete_map=STRATEGY_COLORS,
        height=520,
        hover_data=["model_key", "decoder", "bands", "train_peak_gib", "fp32_all_gib", "source"],
        labels={"need_gib": "exact fp32 data cache + training peak (GiB)"},
        title="GPU-resident training: total memory need per cell vs GPU sizes",
    )
    for b in resources.GPU_BUDGETS:
        fig.add_hline(
            y=b.total_gib - b.headroom_gib,
            line_dash="dot",
            annotation_text=b.name,
            line_color="#d94801",
        )
    return fig


def _steps(bundle: Bundle) -> go.Figure:
    m = bundle.memory[(bundle.memory["data"] == "gpu") & (bundle.memory["batch_size"] == 16)]
    long = m.melt(
        id_vars=["model_key", "dataset", "regime", "concurrent"],
        value_vars=["forward_ms_median", "backward_ms_median", "optimizer_ms_median"],
        var_name="part",
        value_name="ms",
    )
    long["part"] = long["part"].str.replace("_ms_median", "")
    fig = px.bar(
        long[long["concurrent"]],
        x="regime",
        y="ms",
        color="part",
        facet_col="model_key",
        facet_row="dataset",
        height=750,
        category_orders={"regime": list(STRATEGY_ORDER)},
        title="Training-step decomposition (batch 16, GPU-resident data, 8 concurrent jobs)",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def _paths(bundle: Bundle) -> go.Figure:
    p = bundle.data_paths.copy()
    p["load"] = p["concurrent"].map({True: "8 concurrent jobs", False: "one job"})
    p["cell"] = p["model_key"] + " / " + p["regime"]
    fig = px.bar(
        p,
        x="cell",
        y="est_epoch_s",
        color="data",
        barmode="group",
        facet_col="load",
        facet_row="dataset",
        log_y=True,
        height=750,
        category_orders={"data": ["loader", "disk", "host", "gpu"]},
        hover_data=["data_wait_fraction", "train_compute_s", "data_wait_s", "val_s"],
        title="Estimated epoch time by data path",
    )
    return fig


def _curves(bundle: Bundle) -> go.Figure:
    e = bundle.epochs.merge(
        bundle.accuracy[
            ["campaign", "job_id", "dataset", "model_key", "decoder", "strategy", "seed"]
        ],
        on=["campaign", "job_id"],
    )
    e["run"] = (
        e["model_key"]
        + " / "
        + e["decoder"]
        + " / "
        + e["strategy"]
        + " / s"
        + e["seed"].astype(str)
    )
    fig = px.line(
        e,
        x="epoch",
        y="val_miou",
        color="strategy",
        line_group="run",
        facet_col="dataset",
        facet_col_wrap=3,
        color_discrete_map=STRATEGY_COLORS,
        hover_data=["run", "train_miou", "train_loss", "val_loss", "epoch_seconds"],
        category_orders=ORDER,
        height=650,
        title="Validation mIoU curves (all completed accuracy runs)",
    )
    fig.update_traces(opacity=0.5, line={"width": 1})
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def _table(df: pd.DataFrame, title: str) -> str:
    return f"<h3>{html.escape(title)}</h3>" + df.round(3).to_html(
        index=False, border=0, classes="tbl"
    )


SECTIONS = (
    ("Accuracy vs compute", _pareto),
    ("Accuracy ladders", _ladders),
    ("Memory fit", _fit),
    ("Step time", _steps),
    ("Data residency", _paths),
    ("Training curves", _curves),
)
CSS = """body{font-family:system-ui,sans-serif;margin:24px;max-width:1500px}
nav a{margin-right:14px}.tbl{border-collapse:collapse;font-size:12px}
.tbl td,.tbl th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}
.note{color:#555;font-size:13px}"""


def write_explorer(bundle: Bundle, out: Path) -> Path:
    """Write ``explorer.html`` with every interactive section and the summary tables."""
    parts, first = [], True
    for title, fn in SECTIONS:
        fig = fn(bundle)
        anchor = title.lower().replace(" ", "-")
        parts.append(
            f'<h2 id="{anchor}">{html.escape(title)}</h2>'
            + fig.to_html(full_html=False, include_plotlyjs="inline" if first else False)
        )
        first = False
    tables = _table(value_table(bundle), "Value for cost (paired medians)")
    tables += _table(bundle.caches, "Dataset cache sizes (GiB)")
    tables += _table(bundle.scenarios, "WS3 programme budget scenarios")
    nav = " ".join(f'<a href="#{t.lower().replace(" ", "-")}">{t}</a>' for t, _ in SECTIONS)
    meta = bundle.meta
    body = (
        f"<h1>Segmentation fine-tuning: cost-benefit explorer</h1>"
        f"<p class='note'>Snapshot {meta['snapshot_utc']}; analysis code "
        f"{meta['analysis_commit']}; runs per campaign {meta['campaigns']}. "
        "Static figures and the written report are next to this file.</p>"
        f"<nav>{nav} <a href='#tables'>Tables</a></nav>{''.join(parts)}"
        f"<h2 id='tables'>Tables</h2>{tables}"
    )
    path = out / "explorer.html"
    path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Segmentation "
        f"cost-benefit</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    return path
