"""Render the static ranking explorer from benchmark and compute measurements.

The hand-authored source lives in ``ranking_explorer.template.html``.  This
script selects one complete normalization policy per model and view, computes
the ranking and sensitivity payloads, and writes a self-contained HTML page.

Usage::

    python scripts/regen_leaderboard.py
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

from geo_eval_harness.results import (
    NUMERIC_SETUP_COLS,
    SETUP_COLS,
    format_setup_value,
    load_results,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "models"
COMPUTE_PATH = ROOT / "results" / "compute_cost.csv"
TEMPLATE_PATH = ROOT / "scripts" / "ranking_explorer.template.html"
HTML_PATH = ROOT / "docs" / "_static" / "ranking_explorer.html"

# Rows whose ``name`` contains any of these are single-dataset / experimental
# OlmoEarth ablations, not benchmark entries.
ABLATION_TAGS = ("_lsat_", "_naip_", "_sar_")

# ``<base>_landsat_as_s2`` rows feed a Landsat dataset's bands through the
# S2-pretrained model.  They are folded back onto the base model and win a
# collision with the plain reading for the Landsat dataset.
LANDSAT_SUB_SUFFIX = "_landsat_as_s2"

# Datasets excluded from the leaderboard only (all_results.csv is untouched).
DROPPED_DATASETS = frozenset({"m-pv4ger", "m-brick-kiln"})

# Datasets that only ship RGB bands.  A ``bands=all`` run on these is still an
# RGB measurement and must never enter the Multispectral views.
RGB_ONLY_DATASETS = frozenset({"aid", "hotosm_buildings", "open_cities", "resisc45", "ucmerced"})
# An explicit band list made only of these names (``red,green,blue``) is RGB too.
RGB_BAND_NAMES = frozenset({"red", "green", "blue"})

# The plain (S2L2A) TerraMind configs were additionally swept with
# ``dataset.bands=rgb``, which forces their S2L2A-modality pathway onto
# RGB-only input -- a real, off-modality measurement, not a duplicate of the
# dedicated ``_rgb`` config. For the RGB cell, only the ``_rgb`` config's row
# is authoritative; drop this off-modality artifact before alias resolution.
TERRAMIND_OFF_MODALITY = frozenset({"tt_terramind_v1_base", "tt_terramind_v1_large"})
# The mirror case: the RGB-modality ``_rgb`` configs swept with ``bands=all``. For the
# Multispectral cell, only the plain config's row is authoritative.
TERRAMIND_RGB_CONFIGS = frozenset({"tt_terramind_v1_base_rgb", "tt_terramind_v1_large_rgb"})

# Raw metric name -> canonical name registered in the evaluma metric registry.
METRIC_MAP = {"accuracy": "accuracy", "micro_mAP": "map", "mIoU": "miou"}
METRIC_BOUNDS = {"accuracy": (0.0, 1.0), "map": (0.0, 1.0), "miou": (0.0, 1.0)}
DEFAULT_RANDOM_SEED = 0
SMALL_N_MODELS = 8

# Segmentation methods are recorded as ``seg-<head>`` rows; every other method
# value is a classification probe.
SEGMENTATION_METHOD_PREFIX = "seg-"

TASK_ORDER = ["classification", "segmentation"]
PROBE_ORDER = ["linear", "knn5"]

# Canonical segmentation protocol per dataset: ``dataset -> (epochs, LoRA backbone LR)``.
# Unlisted datasets use SEG_PROTOCOL_DEFAULT. Full fine-tuning and the head LR
# are the same everywhere. Setups off this protocol are variants.
SEG_PROTOCOL: dict[str, tuple[int, float]] = {
    "burn_scars": (20, 5e-4),
    "caffe": (20, 5e-4),
    "cloudsen12": (20, 5e-4),
    "fotw": (20, 5e-4),
    "kuro_siwo": (20, 5e-4),
    "flair2": (20, 1e-4),
    "openearthmap-sar": (20, 1e-4),
    "spacenet2": (20, 1e-4),
    "spacenet7": (20, 1e-4),
    "sen1floods11-s1": (30, 5e-4),
    "sen1floods11-s2": (30, 5e-4),
    "oscd-post": (30, 5e-4),
}
SEG_PROTOCOL_DEFAULT: tuple[int, float] = (30, 1e-4)
SEG_HEAD_LR = 1e-3
SEG_FULL_BACKBONE_LR = 1e-5

# A segmentation probe is ``<adaptation>/<decoder>/<pyramid>/<protocol>``; the
# explorer exposes each part as its own selector.
SEG_PROBE_SEP = "/"
SEG_CANONICAL = "canonical"
SEG_ADAPTATION_ORDER = ["frozen", "lora", "full"]
SEG_DECODER_ORDER = ["upernet", "unet"]
SEG_PYRAMID_ORDER = ["native", "vit", "anyup"]
BANDCLASS_ORDER = ["RGB", "Multispectral"]

# ``*_rgb`` is a band-specific TerraMind configuration, not a different model.
# Apply aliases before any coverage or duplicate validation.
MODEL_ALIASES = {
    "tt_terramind_v1_base_rgb": "tt_terramind_v1_base",
    "tt_terramind_v1_large_rgb": "tt_terramind_v1_large",
}

# Base ``name`` slug -> (display name, is_baseline).  Family is deliberately
# generator-only metadata: it is not part of the rendered data contract.
MODEL_DISPLAY: dict[str, tuple[str, bool]] = {
    "rcf": ("Random conv. features", True),
    "imagestats": ("Image statistics", True),
    "olmoearth_v1_nano": ("OlmoEarth v1 nano", False),
    "olmoearth_v1_tiny": ("OlmoEarth v1 tiny", False),
    "olmoearth_v1_base": ("OlmoEarth v1 base", False),
    "olmoearth_v1_large": ("OlmoEarth v1 large", False),
    "olmoearth_v1_1_nano": ("OlmoEarth v1.1 nano", False),
    "olmoearth_v1_1_tiny": ("OlmoEarth v1.1 tiny", False),
    "olmoearth_v1_1_base": ("OlmoEarth v1.1 base", False),
    "olmoearth_v1_2_nano": ("OlmoEarth v1.2 nano", False),
    "olmoearth_v1_2_tiny": ("OlmoEarth v1.2 tiny", False),
    "olmoearth_v1_2_small": ("OlmoEarth v1.2 small", False),
    "olmoearth_v1_2_base": ("OlmoEarth v1.2 base", False),
    "tgeo_croma_base": ("CROMA base", False),
    "tgeo_croma_large": ("CROMA large", False),
    "tgeo_dofa_base": ("DOFA base", False),
    "tgeo_dofa_large": ("DOFA large", False),
    "tgeo_earthloc_s2_resnet50": ("EarthLoc ResNet-50", False),
    "tgeo_panopticon": ("Panopticon", False),
    "tgeo_resnet18_s2rgb_seco": ("SeCo ResNet-18", False),
    "tgeo_resnet50_s2rgb_seco": ("SeCo ResNet-50", False),
    "tgeo_resnet50_fmow_gassl": ("GASSL ResNet-50", False),
    "tgeo_resnet50_s2all_moco": ("MoCo ResNet-50 (MSI)", False),
    "tgeo_resnet50_s2rgb_moco": ("MoCo ResNet-50 (RGB)", False),
    "tgeo_scalemae_large_fmow": ("ScaleMAE large", False),
    "tgeo_swinv2b_s2rgb_satlas_mi": ("SatlasNet Swin-v2-B (MI)", False),
    "tgeo_swinv2b_s2rgb_satlas_si": ("SatlasNet Swin-v2-B (SI)", False),
    "tgeo_swinv2t_s2rgb_satlas_mi": ("SatlasNet Swin-v2-T (MI)", False),
    "tgeo_swinv2t_s2rgb_satlas_si": ("SatlasNet Swin-v2-T (SI)", False),
    "tt_clay_v1_5_base": ("Clay v1.5 base", False),
    "tt_prithvi_eo_v1_100": ("Prithvi-EO-v1 100M", False),
    "tt_prithvi_eo_v2_100_tl": ("Prithvi-EO-v2 100M (TL)", False),
    "tt_prithvi_eo_v2_300": ("Prithvi-EO-v2 300M", False),
    "tt_prithvi_eo_v2_300_tl": ("Prithvi-EO-v2 300M (TL)", False),
    "tt_prithvi_eo_v2_600": ("Prithvi-EO-v2 600M", False),
    "tt_terramind_v1_base": ("TerraMind v1 base", False),
    "tt_terramind_v1_large": ("TerraMind v1 large", False),
    "vit_large_patch16_dinov3": ("DINOv3 ViT-L/16", False),
    "vit_large_patch16_dinov3sat": ("DINOv3-SAT ViT-L/16", False),
}

# Dataset id -> header label shown in the table.  The GeoBench version is a
# suffix because several datasets appear in both V1 and V2 under near-identical
# names, and the raw ids alone do not say which is which.
DATASET_DISPLAY = {
    "m-eurosat": "EuroSAT · V1",
    "m-so2sat": "So2Sat · V1",
    "m-forestnet": "ForestNet · V1",
    "m-bigearthnet": "BigEarthNet · V1",
    "m-pv4ger": "PV4GER · V1",
    "m-brick-kiln": "Brick Kiln · V1",
    "benv2": "BigEarthNet · V2",
    "treesatai": "TreeSatAI · V2",
    "so2sat": "So2Sat · V2",
    "forestnet": "ForestNet · V2",
    "eurosat-spatial": "EuroSAT spatial",
    "resisc45": "RESISC45",
    "burn_scars": "Burn Scars · V2",
    "caffe": "CAFFE · V2",
    "cloudsen12": "CloudSEN12 · V2",
    "dynamic_earthnet": "Dynamic EarthNet · V2",
    "flair2": "FLAIR #2 · V2",
    "fotw": "Fields of the World · V2",
    "kuro_siwo": "Kuro Siwo · V2",
    "pastis": "PASTIS · V2",
    "hotosm_buildings": "HOTOSM Buildings",
    "open_cities": "Open Cities",
    "openearthmap-sar": "OpenEarthMap-SAR",
    "spacenet2": "SpaceNet 2",
    "spacenet7": "SpaceNet 7",
    "loveda": "LoveDA",
    "levir-cd-post": "LEVIR-CD (post)",
    "sen1floods11-s1": "Sen1Floods11 · S1",
    "sen1floods11-s2": "Sen1Floods11 · S2",
    "bright-post-sar": "BRIGHT (post, SAR)",
    "oscd-post": "OSCD (post)",
}

# Canonical metric name -> the label a reader needs to interpret the column.
METRIC_LABEL = {"accuracy": "Top-1 accuracy", "map": "Micro mAP", "miou": "mIoU"}

AGG_META = {
    "avg_rank": {"label": "Average rank", "direction": "lower-is-better", "unit": "rank"},
    "elo": {"label": "ELO", "direction": "higher-is-better", "unit": "elo"},
    "improvability": {"label": "Improvability", "direction": "lower-is-better", "unit": "%"},
}


def canonical_slug(slug: str) -> str:
    """Return the model identity used for quality coverage and compute joins."""
    return MODEL_ALIASES.get(slug, slug)


def resolve_meta(slug: str) -> tuple[str, bool]:
    """Resolve a model slug to its display name and baseline flag."""
    return MODEL_DISPLAY.get(slug, (slug, False))


def dataset_meta(harmonized: pd.DataFrame, datasets: list[str]) -> dict[str, dict[str, str]]:
    """Return the header label and metric label for each seated dataset."""
    metrics = harmonized.groupby("dataset")["metric"].first()
    return {
        dataset: {
            "label": DATASET_DISPLAY.get(dataset, dataset),
            "metric": METRIC_LABEL.get(metrics.get(dataset), str(metrics.get(dataset, ""))),
        }
        for dataset in datasets
    }


def ordered_values(values: pd.Series, preferred: list[str]) -> list[str]:
    """Return observed string values with a stable preferred order first."""
    observed = [str(value) for value in values.dropna().unique()]
    pinned = [value for value in preferred if value in observed]
    return pinned + sorted(value for value in observed if value not in set(preferred))


def _setup_text(value: object, column: str) -> str:
    """Return a setup cell as comparable text; empty and NaN cells become ``""``."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return ""
    text = str(value).strip()
    if column in NUMERIC_SETUP_COLS and text:
        try:
            number = float(text)
        except ValueError:  # allow-except: a malformed cell compares as its own text
            return text
        return format_setup_value(number) or ""
    return text


def _same_number(text: str, expected: float) -> bool:
    try:
        return math.isclose(float(text), expected, rel_tol=1e-9, abs_tol=0.0)
    except ValueError:  # allow-except: a non-numeric cell never matches the protocol
        return False


def segmentation_protocol(dataset: str) -> tuple[int, float]:
    """Return the canonical ``(epochs, LoRA backbone LR)`` of a segmentation dataset."""
    return SEG_PROTOCOL.get(dataset, SEG_PROTOCOL_DEFAULT)


def segmentation_variant(setup: dict[str, str], dataset: str) -> str:
    """Name how a segmentation setup departs from the dataset's canonical protocol.

    Returns :data:`SEG_CANONICAL`, or ``key=value`` parts joined by ``,`` for the
    epochs, backbone LR and head LR that differ (for example ``epochs=50``).
    """
    epochs, lora_lr = segmentation_protocol(dataset)
    backbone_lr = {"frozen": None, "lora": lora_lr, "full": SEG_FULL_BACKBONE_LR}.get(
        setup["adaptation"]
    )
    parts: list[str] = []
    if not _same_number(setup["epochs"], epochs):
        parts.append(f"epochs={setup['epochs'] or 'unset'}")
    if backbone_lr is None:
        if setup["backbone_lr"]:
            parts.append(f"backbone_lr={setup['backbone_lr']}")
    elif not _same_number(setup["backbone_lr"], backbone_lr):
        parts.append(f"backbone_lr={setup['backbone_lr'] or 'unset'}")
    if not _same_number(setup["head_lr"], SEG_HEAD_LR):
        parts.append(f"head_lr={setup['head_lr'] or 'unset'}")
    return ",".join(parts) or SEG_CANONICAL


def segmentation_pyramid(setup: dict[str, str]) -> str:
    """Return the feature-pyramid choice: ``anyup``, a neck name, or ``native``."""
    if setup["upsampler"] not in ("", "none"):
        return setup["upsampler"]
    if setup["neck"] not in ("", "none"):
        return setup["neck"]
    return "native"


def segmentation_probe(setup: dict[str, str], dataset: str) -> str:
    """Return the ``<adaptation>/<decoder>/<pyramid>/<protocol>`` view key of a setup."""
    return SEG_PROBE_SEP.join(
        [
            setup["adaptation"],
            setup["decoder"],
            segmentation_pyramid(setup),
            segmentation_variant(setup, dataset),
        ]
    )


def is_segmentation_variant(probe: str) -> bool:
    """Return whether a probe key is a segmentation setup off the canonical protocol."""
    parts = probe.split(SEG_PROBE_SEP)
    return len(parts) == 4 and parts[3] != SEG_CANONICAL


def _probe_sort_key(probe: str) -> tuple[object, ...]:
    """Order segmentation probes by adaptation, decoder, pyramid, then canonical first."""
    parts = probe.split(SEG_PROBE_SEP)
    if len(parts) != 4:
        return (1,)  # classification probes keep their PROBE_ORDER position (stable sort)

    def position(value: str, order: list[str]) -> tuple[int, str]:
        return (order.index(value), "") if value in order else (len(order), value)

    adaptation, decoder, pyramid, protocol = parts
    return (
        0,
        position(adaptation, SEG_ADAPTATION_ORDER),
        position(decoder, SEG_DECODER_ORDER),
        position(pyramid, SEG_PYRAMID_ORDER),
        protocol != SEG_CANONICAL,
        protocol,
    )


def _aggregate_seeds(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average the seeds of each segmentation setup into one cell (mean, sd, n).

    Seeds share every setup column, band list and source config.  Two different
    setups with the same band list that land in the same leaderboard cell are an
    error: the view axes do not distinguish them, so averaging them would mix
    experiments.  Two band lists of one band class (``vv,vh`` and ``vv``) stay
    separate cells, so, as for classification, the model is not complete in that
    view and is listed as excluded.
    """
    seg = df["task"] == "segmentation"
    other = df[~seg].assign(score_sd=float("nan"), n_seeds=1)
    if not seg.any():
        return other
    group = [*keys, "source_model", "bands", *SETUP_COLS]
    stats = (
        df[seg]
        .groupby(group, dropna=False, sort=False)["score"]
        .agg(score="mean", score_sd="std", n_seeds="count")
        .reset_index()
    )
    clash = stats.duplicated([*keys, "source_model", "bands"], keep=False)
    if clash.any():
        offenders = stats.loc[clash, [*group, "n_seeds"]].sort_values([*keys, "source_model"])
        raise ValueError(
            "Different segmentation setups share one leaderboard cell; the view axes "
            "(adaptation, decoder, pyramid, protocol) do not separate them.\n"
            f"{offenders.to_string(index=False)}"
        )
    return pd.concat([other, stats], ignore_index=True)


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw result rows into canonical quality rows.

    The output remains long-format so view selection can choose either a fully
    native model row or a fully z-scored one.  Pooling and normalization are
    retained internally only; the public explorer has neither as an axis.
    """
    df = df.copy()
    df = df[~df["dataset"].isin(DROPPED_DATASETS)]
    name = df["name"].astype(str)

    is_rgb_sweep = df["bands"].astype(str) == "rgb"
    off_modality = (name.isin(TERRAMIND_OFF_MODALITY) & is_rgb_sweep) | (
        name.isin(TERRAMIND_RGB_CONFIGS) & ~is_rgb_sweep
    )
    df = df[~off_modality].copy()
    name = df["name"].astype(str)

    ablation = pd.Series(False, index=df.index)
    for tag in ABLATION_TAGS:
        ablation |= name.str.contains(tag, regex=False)
    df = df[~ablation].copy()

    df = df[df["metric_name"].isin(METRIC_MAP)].copy()
    df["metric"] = df["metric_name"].map(METRIC_MAP)

    name = df["name"].astype(str)
    is_cls = name.str.endswith("_cls")
    df["pooling"] = is_cls.map({True: "cls-token", False: "patch-mean"})
    base = name.where(~is_cls, name.str.slice(0, -4))
    df["is_landsat_sub"] = base.str.endswith(LANDSAT_SUB_SUFFIX)
    base = base.where(~df["is_landsat_sub"], base.str.removesuffix(LANDSAT_SUB_SUFFIX))
    df["source_model"] = base
    df["base_model"] = base.map(canonical_slug)

    band_lists = df["bands"].astype(str).str.lower().str.replace(" ", "").str.split(",")
    rgb_list = band_lists.map(lambda bands: bool(bands) and set(bands) <= RGB_BAND_NAMES)
    is_rgb = (df["bands"].astype(str) == "rgb") | rgb_list | df["dataset"].isin(RGB_ONLY_DATASETS)
    df["bandclass"] = is_rgb.map({True: "RGB", False: "Multispectral"})
    df["probe"] = df["method"].astype(str)
    is_segmentation = df["probe"].str.startswith(SEGMENTATION_METHOD_PREFIX)
    df["task"] = is_segmentation.map({True: "segmentation", False: "classification"})
    for column in SETUP_COLS:
        values = df[column] if column in df else pd.Series(pd.NA, index=df.index)
        df[column] = [_setup_text(value, column) for value in values]
    # Rows without setup columns predate them (legacy seg-linear/fpn/dpt/conv_block
    # grids); they are retired from display.
    df = df[~(is_segmentation & (df["adaptation"] == ""))].copy()
    is_segmentation = df["task"] == "segmentation"
    if is_segmentation.any():
        df.loc[is_segmentation, "probe"] = [
            segmentation_probe(setup, dataset)
            for setup, dataset in zip(
                df.loc[is_segmentation, list(SETUP_COLS)].to_dict("records"),
                df.loc[is_segmentation, "dataset"],
                strict=True,
            )
        ]
    df["normalization"] = df["normalization"].astype("string").str.strip()
    missing_norm = df["normalization"].isna() | (df["normalization"] == "")
    if missing_norm.any():
        preview = df.loc[missing_norm, ["name", "dataset", "method", "bands", "metric_name"]].head(
            10
        )
        raise ValueError(
            "Ranking explorer quality rows must record `normalization`. "
            f"Found {int(missing_norm.sum())} missing rows.\n{preview.to_string(index=False)}"
        )
    df["normalization"] = df["normalization"].astype(str)
    df["score"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df = df.dropna(subset=["score"])

    keys = [
        "base_model",
        "dataset",
        "task",
        "probe",
        "bandclass",
        "pooling",
        "normalization",
        "metric",
    ]
    has_landsat = df.groupby(keys)["is_landsat_sub"].transform("any")
    df = df[~(has_landsat & ~df["is_landsat_sub"])]
    df = df.assign(bands=df["bands"].astype(str))
    df = _aggregate_seeds(df[[*keys, "source_model", "bands", *SETUP_COLS, "score"]], keys)
    return df[[*keys, "source_model", "score", "score_sd", "n_seeds"]].reset_index(drop=True)


def _describe_view(task: str, probe: str, bandclass: str) -> str:
    return f"task={task}, probe={probe}, bandclass={bandclass}, pooling=patch-mean"


def _raise_on_alias_collisions(sub: pd.DataFrame, *, task: str, probe: str, bandclass: str) -> None:
    """Reject duplicate cells created by a TerraMind identity alias.

    Ordinary duplicates on an incomplete model stay outside the seated roster,
    as they did before the refactor.  In contrast, an alias collision would
    merge two configurations into one claimed model identity, so it must fail
    before coverage can mask it.
    """
    keys = ["base_model", "normalization", "dataset", "metric"]
    source_counts = sub.groupby(keys)["source_model"].nunique()
    collisions = source_counts[source_counts > 1]
    if collisions.empty:
        return
    collision_index = collisions.index
    offender_index = pd.MultiIndex.from_frame(sub[keys]).isin(collision_index)
    offenders = sub.loc[offender_index, [*keys, "source_model", "score"]].sort_values(
        [*keys, "source_model", "score"]
    )
    raise ValueError(
        f"Duplicate leaderboard cells remain after model aliasing for "
        f"{_describe_view(task, probe, bandclass)}.\n"
        "Aliased model configurations must not occupy the same "
        f"(model, normalization, dataset, metric) cell.\n{offenders.to_string(index=False)}"
    )


def _raise_on_duplicate_selected_cells(
    selected: pd.DataFrame, *, task: str, probe: str, bandclass: str
) -> None:
    """Reject a duplicate cell that would be emitted into a selected model row."""
    keys = ["base_model", "dataset", "metric"]
    duplicate = selected.duplicated(keys, keep=False)
    if not duplicate.any():
        return
    offenders = selected.loc[duplicate, [*keys, "normalization", "score"]].sort_values(
        [*keys, "normalization", "score"]
    )
    raise ValueError(
        f"Duplicate selected leaderboard cells remain for "
        f"{_describe_view(task, probe, bandclass)}.\n"
        "Each seated (model, dataset, metric) cell must come from exactly one evaluation row.\n"
        f"{offenders.to_string(index=False)}"
    )


def _complete_rows(rows: pd.DataFrame, universe: set[tuple[str, str]]) -> bool:
    """Return whether rows provide exactly one score for every universe cell."""
    if len(rows) != len(universe):
        return False
    cells = list(rows[["dataset", "metric"]].itertuples(index=False, name=None))
    return len(set(cells)) == len(universe) and set(cells) == universe


def select_view(
    harmonized: pd.DataFrame,
    task: str,
    probe: str,
    bandclass: str,
) -> tuple[pd.DataFrame | None, list[dict[str, object]]]:
    """Select one complete normalization row per model for one quality view.

    The dataset/metric universe is established before choosing a normalization.
    A model uses ``model_native`` only when that *entire* row is complete;
    otherwise it uses its complete ``bandspec_zscore`` row.  Partial native
    measurements cannot displace or mix with a z-score row.  Returns the
    selected ``model, dataset, metric, score, score_sd, n_seeds`` cells.
    """
    sub = harmonized[
        (harmonized["task"] == task)
        & (harmonized["probe"] == probe)
        & (harmonized["bandclass"] == bandclass)
        & (harmonized["pooling"] == "patch-mean")
    ].copy()
    if sub.empty:
        return None, []

    _raise_on_alias_collisions(sub, task=task, probe=probe, bandclass=bandclass)
    universe = set(sub[["dataset", "metric"]].itertuples(index=False, name=None))
    selected: list[pd.DataFrame] = []
    excluded: list[dict[str, object]] = []

    for model in sorted(sub["base_model"].unique()):
        model_rows = sub[sub["base_model"] == model]
        native = model_rows[model_rows["normalization"] == "model_native"]
        zscore = model_rows[model_rows["normalization"] == "bandspec_zscore"]
        if _complete_rows(native, universe):
            selected.append(native)
        elif _complete_rows(zscore, universe):
            selected.append(zscore)
        else:
            excluded.append(
                {
                    "model": model,
                    "display": resolve_meta(model)[0],
                    "n_tasks": int(max(native["dataset"].nunique(), zscore["dataset"].nunique())),
                }
            )

    if not selected:
        return None, excluded
    frame = pd.concat(selected, ignore_index=True)
    _raise_on_duplicate_selected_cells(frame, task=task, probe=probe, bandclass=bandclass)
    frame = frame.rename(columns={"base_model": "model"})
    return frame[["model", "dataset", "metric", "score", "score_sd", "n_seeds"]], excluded


def _load_bench(frame: pd.DataFrame) -> object:
    import evaluma

    return evaluma.load_df(
        frame[["model", "dataset", "metric", "score"]],
        metric_type_bounds=METRIC_BOUNDS,
        drop_incomplete=True,
    )


def build_view(
    harmonized: pd.DataFrame,
    task: str,
    probe: str,
    bandclass: str,
) -> tuple[object | None, list[dict[str, object]]]:
    """Build a complete hybrid-normalization benchmark for one quality view."""
    frame, excluded = select_view(harmonized, task, probe, bandclass)
    if frame is None:
        return None, excluded
    return _load_bench(frame), excluded


def avg_rank(bench: object) -> list[dict[str, object]]:
    """Return average dataset ranks, where lower is better."""
    scores = bench.scores_
    mean_ranks = scores.rank(ascending=False, axis=0).mean(axis=1).sort_values()
    return [{"model": model, "overall": float(rank)} for model, rank in mean_ranks.items()]


def elo(
    bench: object,
    n_bootstrap: int = 1000,
    random_state: int | None = DEFAULT_RANDOM_SEED,
) -> list[dict[str, object]]:
    """Return ELO ratings and (when requested) bootstrap confidence intervals."""
    table = bench.elo_ranking(n_bootstrap=n_bootstrap, random_state=random_state).table.sort_values(
        "ELO", ascending=False
    )
    rows: list[dict[str, object]] = []
    for row in table.itertuples(index=False):
        ci_low = getattr(row, "CI_low", None)
        ci_high = getattr(row, "CI_high", None)
        has_ci = (
            ci_low is not None and ci_high is not None and not (pd.isna(ci_low) or pd.isna(ci_high))
        )
        rows.append(
            {
                "model": row.model,
                "overall": float(row.ELO),
                "overall_ci": (float(ci_low), float(ci_high)) if has_ci else None,
            }
        )
    return rows


def improvability(bench: object) -> list[dict[str, object]]:
    """Return mean percent error reduction to each dataset's best score."""
    table = bench.improvability_ranking().table.sort_values("improvability")
    return [
        {"model": row.model, "overall": float(row.improvability)}
        for row in table.itertuples(index=False)
    ]


def _ranked_rows(
    bench: object,
    aggregation: str,
    *,
    n_bootstrap: int,
    random_state: int | None,
) -> list[dict[str, object]]:
    """Apply one ranking aggregation with the requested ELO bootstrap policy."""
    if aggregation == "avg_rank":
        return avg_rank(bench)
    if aggregation == "elo":
        return elo(bench, n_bootstrap=n_bootstrap, random_state=random_state)
    if aggregation == "improvability":
        return improvability(bench)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _enrich_rows(
    bench: object,
    ranked: list[dict[str, object]],
    compute: dict[tuple[str, str], float],
    bandclass: str,
    cells: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Attach the remaining table/scatter fields to a ranked model row.

    ``per_task`` holds the score (the seed mean for segmentation) and
    ``per_task_sd`` / ``per_task_n`` its seed standard deviation (``None`` for
    one seed) and seed count, taken from the selected ``cells``.
    """
    band_config = "rgb" if bandclass == "RGB" else "s2"
    scores = bench.scores_
    seeds: dict[tuple[str, str], tuple[float | None, int]] = {}
    if cells is not None:
        for cell in cells.itertuples(index=False):
            sd = float(cell.score_sd)
            seeds[(str(cell.model), str(cell.dataset))] = (
                None if math.isnan(sd) else sd,
                int(cell.n_seeds),
            )
    n_tasks = len(bench.datasets_)
    rows: list[dict[str, object]] = []
    for row in ranked:
        model = str(row["model"])
        display, is_baseline = resolve_meta(model)
        ci = row.get("overall_ci")
        rows.append(
            {
                "model": model,
                "display": display,
                "is_baseline": is_baseline,
                "gflops_backbone": compute.get((model, band_config)),
                "overall": row["overall"],
                "overall_ci": list(ci) if ci is not None else None,
                "n_tasks": n_tasks,
                "per_task": {dataset: float(value) for dataset, value in scores.loc[model].items()},
                "per_task_sd": {
                    dataset: seeds.get((model, dataset), (None, 1))[0] for dataset in scores.columns
                },
                "per_task_n": {
                    dataset: seeds.get((model, dataset), (None, 1))[1] for dataset in scores.columns
                },
            }
        )
    return rows


def extract_compute_cost(raw: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Extract strict canonical ``(model, band_config) -> backbone GFLOPs`` joins.

    Blank measurement rows are normal in the current data and ignored.  Two
    distinct non-empty values for a model/band pair are ambiguous and fail
    loudly instead of choosing one by file order.
    """
    required = {"name", "band_config", "gflops_backbone"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"compute_cost.csv is missing required columns: {sorted(missing)}")

    values: dict[tuple[str, str], set[float]] = {}
    for row in raw[["name", "band_config", "gflops_backbone"]].itertuples(index=False):
        name, band_config, value = row
        if pd.isna(value) or str(value).strip() == "":
            continue
        try:
            gflops = float(value)
        except (TypeError, ValueError) as exc:  # allow-except: report invalid compute measurements
            raise ValueError(
                f"Invalid gflops_backbone for model={name!r}, band_config={band_config!r}: {value!r}"
            ) from exc
        if not math.isfinite(gflops):
            raise ValueError(
                f"Non-finite gflops_backbone for model={name!r}, band_config={band_config!r}: {value!r}"
            )
        key = (canonical_slug(str(name)), str(band_config))
        values.setdefault(key, set()).add(gflops)

    conflicts = {key: sorted(measured) for key, measured in values.items() if len(measured) > 1}
    if conflicts:
        details = "; ".join(f"{key}: {measured}" for key, measured in sorted(conflicts.items()))
        raise ValueError(f"Conflicting non-empty backbone GFLOPs measurements: {details}")
    return {key: next(iter(measured)) for key, measured in values.items()}


def _view_keys(harmonized: pd.DataFrame) -> list[dict[str, str]]:
    """Enumerate all patch-mean quality views in display order."""
    patch_mean = harmonized[harmonized["pooling"] == "patch-mean"]
    tasks = ordered_values(patch_mean["task"], TASK_ORDER)
    probes = sorted(ordered_values(patch_mean["probe"], PROBE_ORDER), key=_probe_sort_key)
    bands = ordered_values(patch_mean["bandclass"], BANDCLASS_ORDER)
    keys: list[dict[str, str]] = []
    for task in tasks:
        for probe in probes:
            keys.extend(
                {"task": task, "probe": probe, "bandclass": bandclass}
                for bandclass in bands
                if not patch_mean[
                    (patch_mean["task"] == task)
                    & (patch_mean["probe"] == probe)
                    & (patch_mean["bandclass"] == bandclass)
                ].empty
            )
    return keys


def _rank_vector(bench: object, aggregation: str) -> pd.Series:
    """Return a tie-aware rank vector for cross-view Kendall tau-b."""
    ranked = _ranked_rows(bench, aggregation, n_bootstrap=0, random_state=DEFAULT_RANDOM_SEED)
    values = pd.Series({str(row["model"]): float(row["overall"]) for row in ranked})
    ascending = AGG_META[aggregation]["direction"] == "lower-is-better"
    return values.rank(method="average", ascending=ascending)


def _matrix_entry(
    bench_a: object,
    bench_b: object,
    aggregation: str,
) -> dict[str, object]:
    """Compare two complete views on their shared roster and full own datasets."""
    models = sorted(set(bench_a.models_) & set(bench_b.models_))
    n_datasets_a = len(bench_a.datasets_)
    n_datasets_b = len(bench_b.datasets_)
    if not models:
        return {
            "tau": None,
            "n_models": 0,
            "n_datasets_a": n_datasets_a,
            "n_datasets_b": n_datasets_b,
            "small_n": True,
        }

    from scipy.stats import kendalltau

    ranks_a = _rank_vector(bench_a.select_models(models), aggregation)
    ranks_b = _rank_vector(bench_b.select_models(models), aggregation)
    result = kendalltau(ranks_a.loc[models], ranks_b.loc[models], variant="b")
    tau = result.statistic
    return {
        "tau": None if tau is None or math.isnan(float(tau)) else float(tau),
        "n_models": len(models),
        "n_datasets_a": n_datasets_a,
        "n_datasets_b": n_datasets_b,
        "small_n": len(models) < SMALL_N_MODELS,
    }


def sensitivity_matrix(views: list[dict[str, str]], benches: list[object]) -> dict[str, object]:
    """Build every aggregation's cross-view Kendall tau-b matrix."""
    by_aggregation: dict[str, list[list[dict[str, object]]]] = {}
    for aggregation in AGG_META:
        matrix: list[list[dict[str, object] | None]] = [[None for _ in views] for _ in views]
        for row_index, bench_a in enumerate(benches):
            for column_index in range(row_index, len(benches)):
                bench_b = benches[column_index]
                entry = _matrix_entry(bench_a, bench_b, aggregation)
                matrix[row_index][column_index] = entry
                if row_index == column_index:
                    continue
                matrix[column_index][row_index] = {
                    **entry,
                    "n_datasets_a": entry["n_datasets_b"],
                    "n_datasets_b": entry["n_datasets_a"],
                }
        by_aggregation[aggregation] = [
            [entry for entry in row if entry is not None] for row in matrix
        ]
    return {
        "views": [
            {"key": f"{view['task']}|{view['probe']}|{view['bandclass']}", **view} for view in views
        ],
        "by_aggregation": by_aggregation,
    }


def _default_slice(rankings: dict[str, object]) -> dict[str, str]:
    """Pick the conventional default, falling back to the first populated view."""
    preferred = {
        "task": "classification",
        "probe": "linear",
        "bandclass": "RGB",
        "aggregation": "avg_rank",
    }
    try:
        rankings["classification"]["linear"]["RGB"]["avg_rank"]
    except KeyError:  # allow-except: choose an available view when the preferred view is absent
        task = next(iter(rankings))
        probe = next(iter(rankings[task]))
        bandclass = next(iter(rankings[task][probe]))
        preferred.update({"task": task, "probe": probe, "bandclass": bandclass})
    return preferred


def assemble(
    harmonized: pd.DataFrame,
    n_bootstrap: int = 1000,
    compute: dict[tuple[str, str], float] | None = None,
    random_state: int | None = DEFAULT_RANDOM_SEED,
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    """Assemble the four-level ranking contract and sensitivity matrix.

    Returns ``(RANKINGS, SENSITIVITY, DEFAULT_SLICE)``.  The ranking payload
    is indexed by ``task/probe/bandclass/aggregation`` and rows contain only
    fields consumed by the reduced table and compute scatter.
    """
    compute = compute or {}
    rankings: dict[str, object] = {}
    views: list[dict[str, str]] = []
    benches: list[object] = []

    for view in _view_keys(harmonized):
        cells, excluded = select_view(harmonized, **view)
        bench = None if cells is None else _load_bench(cells)
        if bench is None or len(bench.models_) < 2:
            continue
        aggregation_blocks: dict[str, object] = {}
        for aggregation, meta in AGG_META.items():
            ranked = _ranked_rows(
                bench, aggregation, n_bootstrap=n_bootstrap, random_state=random_state
            )
            aggregation_blocks[aggregation] = {
                "rows": _enrich_rows(bench, ranked, compute, view["bandclass"], cells),
                "overall_meta": meta,
                "dataset_meta": dataset_meta(harmonized, list(bench.datasets_)),
                "excluded": excluded,
            }
        rankings.setdefault(view["task"], {}).setdefault(view["probe"], {})[view["bandclass"]] = (
            aggregation_blocks
        )
        if is_segmentation_variant(view["probe"]):
            # Variants are ranked but left out of the cross-view sensitivity matrix.
            continue
        views.append(view)
        benches.append(bench)

    if not rankings:
        raise ValueError("No ranking views contain at least two complete models.")
    return rankings, sensitivity_matrix(views, benches), _default_slice(rankings)


def inline_into_html(html_text: str, blocks: dict[str, object]) -> str:
    """Replace empty ``const NAME = …;`` anchors with compact JSON payloads."""
    text = html_text
    for name, obj in blocks.items():
        payload = json.dumps(obj, separators=(",", ":"), allow_nan=False)
        pattern = re.compile(rf"const {name} = .*?;", re.DOTALL)
        if not pattern.search(text):
            raise SystemExit(f"Could not locate `const {name} = …;` in HTML.")
        text = pattern.sub(lambda _match, n=name, p=payload: f"const {n} = {p};", text, count=1)
    return text


def main() -> None:
    """Parse CLI arguments and write a rendered explorer page."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(CSV_PATH),
        help="Path to a per-model results directory (default) or a single all_results.csv file.",
    )
    parser.add_argument("--compute", default=str(COMPUTE_PATH), help="Path to compute_cost.csv.")
    parser.add_argument(
        "--template", default=str(TEMPLATE_PATH), help="Hand-authored ranking explorer template."
    )
    parser.add_argument("--html", default=str(HTML_PATH), help="Generated leaderboard HTML path.")
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Bootstrap resamples for ELO CIs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible ELO bootstrap confidence intervals.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    raw = load_results(csv_path) if csv_path.is_dir() else pd.read_csv(csv_path, low_memory=False)
    compute_raw = pd.read_csv(args.compute, low_memory=False)
    rankings, sensitivity, default_slice = assemble(
        harmonize(raw),
        n_bootstrap=args.bootstrap,
        compute=extract_compute_cost(compute_raw),
        random_state=args.seed,
    )
    output = inline_into_html(
        Path(args.template).read_text(encoding="utf-8"),
        {"RANKINGS": rankings, "SENSITIVITY": sensitivity, "DEFAULT_SLICE": default_slice},
    )
    html_path = Path(args.html)
    html_path.write_text(output, encoding="utf-8")

    logger.info(
        "Wrote %s: %d condition views, default %s/%s/%s",
        html_path,
        len(sensitivity["views"]),
        default_slice["task"],
        default_slice["probe"],
        default_slice["bandclass"],
    )


if __name__ == "__main__":
    main()
