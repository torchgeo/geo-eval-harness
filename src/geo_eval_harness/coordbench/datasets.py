"""CoordBench dataset loaders — coordinate (lon/lat) -> label benchmark tables.

Every benchmark is read directly from the unified HuggingFace dataset repo
``taylor-geospatial/coordbench`` (one normalized parquet per config). This is the
location-encoder evaluation suite ported from the MIND codebase: point (lon, lat)
in, a downstream label out, probed with a frozen encoder.

A loader returns one or more :class:`CoordBenchmark`. ``task_type`` selects the
probe (ridge R^2 for regression, accuracy for classification). Families are
registered in :data:`FAMILY_LOADERS`; :func:`load_benchmarks` expands a selection
(``"all"``, family names, or individual benchmark names) into a flat list.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from geo_eval_harness.coordbench.catalog import (
    CDC_PLACES_MEASURES,
    DEEPMIND_EVAL_CONFIGS,
    FAMILY_BENCHMARKS,
    SUSTAINBENCH_TASKS,
    USAVARS_LABELS,
)

logger = logging.getLogger(__name__)

# https://huggingface.co/datasets/taylor-geospatial/coordbench
COORDBENCH_REPO = os.environ.get("COORDBENCH_REPO", "taylor-geospatial/coordbench")

# canonical (non-task) schema columns, excluded when scanning for task columns
_CANONICAL_EXTRA = frozenset({"timestamp", "timestamp_end", "split", "id"})

USAVARS_NODATA = -999.0  # nodata sentinel in the label CSVs
# log1p the heavy right-skewed targets (matches PDFM's own convention for pop/nightlights).
USAVARS_LOG_LABELS = frozenset({"population", "income", "nightlights", "housing"})

PDFM_NON_TASK = frozenset(
    {
        "place",
        "county",
        "state",
        "latitude",
        "longitude",
        "Count_Person",
        "imputation_split",
        "superresolution_split",
        "extrapolation_split",
    }
)


@dataclass
class CoordBenchmark:
    """A single coordinate -> label benchmark.

    Args:
        name: Unique benchmark identifier (e.g. ``"sustainbench-asset"``).
        lat: Latitudes, shape ``(N,)``.
        lon: Longitudes, shape ``(N,)``.
        tasks: Mapping of task/column name -> label array, each shape ``(N,)``.
        task_type: ``"regression"`` (R^2) or ``"classification"`` (accuracy).
        year: Optional per-point year for year-conditioned encoders.
        test_mask: Optional boolean held-out test mask (official split); when
            ``None`` the probe uses k-fold cross-validation.
    """

    name: str
    lat: np.ndarray
    lon: np.ndarray
    tasks: dict[str, np.ndarray] = field(default_factory=dict)
    task_type: str = "regression"
    year: np.ndarray | None = None
    test_mask: np.ndarray | None = None


def load_config(config: str) -> pd.DataFrame:
    """Read one CoordBench config's normalized parquet table from HuggingFace."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(COORDBENCH_REPO, f"data/{config}/data.parquet", repo_type="dataset")
    return pd.read_parquet(path)


def _lonlat_cols(df: pd.DataFrame) -> tuple[str, str]:
    lon_col = next(c for c in ("lon", "longitude", "Lon", "x") if c in df.columns)
    lat_col = next(c for c in ("lat", "latitude", "Lat", "y") if c in df.columns)
    return lon_col, lat_col


def _first_col(df: pd.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


def load_pdfm() -> list[CoordBenchmark]:
    """PDFM conus27 — 27 US ZIP-level health/socio/environment regression tasks."""
    df = load_config("pdfm_conus27")
    lon_col, lat_col = _lonlat_cols(df)
    non_task = PDFM_NON_TASK | {lon_col, lat_col} | _CANONICAL_EXTRA
    task_cols = [c for c in df.columns if c not in non_task]
    return [
        CoordBenchmark(
            name="pdfm-conus27",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={c: df[c].to_numpy(np.float64) for c in task_cols},
        )
    ]


def load_air_temp() -> list[CoordBenchmark]:
    """SatCLIP air temperature — ~3000 global stations (Hooker et al. 2018)."""
    df = load_config("air_temp")
    lon_col, lat_col = _lonlat_cols(df)
    cols = {c.lower(): c for c in df.columns}
    temp_keys = [k for k in cols if "temp" in k or k == "meant"] or [
        k for k in cols if k.startswith("mean")
    ]
    temp_col = cols[temp_keys[0]]
    return [
        CoordBenchmark(
            name="satclip-air-temp",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={"air_temperature": df[temp_col].to_numpy(np.float64)},
        )
    ]


def load_california_housing() -> list[CoordBenchmark]:
    """California housing (sklearn) — regional median house value regression."""
    df = load_config("california_housing")
    lon_col, lat_col = _lonlat_cols(df)
    return [
        CoordBenchmark(
            name="california-housing",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={"median_house_value": df["MedHouseVal"].to_numpy(np.float64)},
        )
    ]


def load_satclip() -> list[CoordBenchmark]:
    """SatCLIP's 5 released downstream datasets (Klemmer et al.).

    country/ecoregion/biome (classification) + population/elevation (regression).
    Ocean nodata (country index 0, ecoregion/biome index -1) is dropped.
    """
    out: list[CoordBenchmark] = []

    c = load_config("satclip_country")
    lon_c, lat_c = _lonlat_cols(c)
    c = c[c["country"] != 0]  # index 0 = ocean/nodata
    out.append(
        CoordBenchmark(
            name="satclip-country",
            lat=c[lat_c].to_numpy(np.float64),
            lon=c[lon_c].to_numpy(np.float64),
            tasks={"country": c["country"].to_numpy()},
            task_type="classification",
        )
    )

    e = load_config("satclip_ecoregion")
    lon_e, lat_e = _lonlat_cols(e)
    e = e[e["ecoregion_class_index"] != -1]  # -1 = ocean/nodata
    for col, name in [
        ("ecoregion_class_index", "satclip-ecoregion"),
        ("biome_class_index", "satclip-biome"),
    ]:
        out.append(
            CoordBenchmark(
                name=name,
                lat=e[lat_e].to_numpy(np.float64),
                lon=e[lon_e].to_numpy(np.float64),
                tasks={col: e[col].to_numpy()},
                task_type="classification",
            )
        )

    p = load_config("satclip_population")  # target is already log_population
    lon_p, lat_p = _lonlat_cols(p)
    out.append(
        CoordBenchmark(
            name="satclip-population",
            lat=p[lat_p].to_numpy(np.float64),
            lon=p[lon_p].to_numpy(np.float64),
            tasks={"log_population": p["log_population"].to_numpy(np.float64)},
        )
    )

    el = load_config("satclip_elevation")
    lon_el, lat_el = _lonlat_cols(el)
    out.append(
        CoordBenchmark(
            name="satclip-elevation",
            lat=el[lat_el].to_numpy(np.float64),
            lon=el[lon_el].to_numpy(np.float64),
            tasks={"elevation": el["elevation"].to_numpy(np.float64)},
        )
    )
    return out


def load_sustainbench() -> list[CoordBenchmark]:
    """SustainBench DHS SDG indices — 6 socioeconomic regressions at cluster centroids.

    Uses the dataset's official trainval/test split (``test_mask``), matching the
    SustainBench protocol rather than k-fold CV.
    """
    df = load_config("sustainbench")  # merged table; split == "trainval" | "test"
    is_test = (df["split"] == "test").to_numpy()
    out: list[CoordBenchmark] = []
    for key, col in SUSTAINBENCH_TASKS.items():
        m = df[col].notna() & df["lat"].notna() & df["lon"].notna()
        sub = df[m]
        out.append(
            CoordBenchmark(
                name=f"sustainbench-{key}",
                lat=sub["lat"].to_numpy(np.float64),
                lon=sub["lon"].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
                test_mask=is_test[m.to_numpy()],
            )
        )
    return out


def load_better_together() -> list[CoordBenchmark]:
    """Better Together (van der Plas et al.) — 6 coordinate -> label tasks.

    dw/bioclim/footprint share one 10k random-sample pool joined by ``id``; this
    restricts bioclim/human_footprint to that pool, matching the original protocol.
    """
    out: list[CoordBenchmark] = []

    ch = load_config("bt_cropharvest")
    out.append(
        CoordBenchmark(
            name="bt-cropharvest",
            lat=ch["lat"].to_numpy(np.float64),
            lon=ch["lon"].to_numpy(np.float64),
            tasks={"crop_type": ch["label_name"].to_numpy()},
            task_type="classification",
        )
    )

    bm = load_config("bt_biomass")  # biomass_mean is clean; biomass_center has -9999 sentinels
    m = bm["biomass_mean"].notna()
    out.append(
        CoordBenchmark(
            name="bt-biomass",
            lat=bm.loc[m, "lat"].to_numpy(np.float64),
            lon=bm.loc[m, "lon"].to_numpy(np.float64),
            tasks={"biomass": bm.loc[m, "biomass_mean"].to_numpy(np.float64)},
        )
    )

    dw = load_config("bt_landcover")
    dw = dw[dw["random_sample"] == 1]  # the 10k pool the paper probes
    lc_cols = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]
    out.append(
        CoordBenchmark(
            name="bt-landcover",
            lat=dw["lat"].to_numpy(np.float64),
            lon=dw["lon"].to_numpy(np.float64),
            tasks={c: dw[c].to_numpy(np.float64) for c in lc_cols},  # fractional cover
        )
    )

    pool_ids = set(dw["id"])
    bio = load_config("bt_bioclim")
    bio = bio[bio["id"].isin(pool_ids)]
    bio_cols = [f"bioclim_{i:02d}" for i in range(1, 20)]
    out.append(
        CoordBenchmark(
            name="bt-bioclim",
            lat=bio["lat"].to_numpy(np.float64),
            lon=bio["lon"].to_numpy(np.float64),
            tasks={c: bio[c].to_numpy(np.float64) for c in bio_cols},
        )
    )

    hf = load_config("bt_human_footprint")
    hf = hf[hf["id"].isin(pool_ids)]
    out.append(
        CoordBenchmark(
            name="bt-population",
            lat=hf["lat"].to_numpy(np.float64),
            lon=hf["lon"].to_numpy(np.float64),
            tasks={"pop_density": hf["pop_density"].to_numpy(np.float64)},
        )
    )
    out.append(
        CoordBenchmark(
            name="bt-distroad",
            lat=hf["lat"].to_numpy(np.float64),
            lon=hf["lon"].to_numpy(np.float64),
            tasks={"maxdist_road": hf["maxdist_road"].to_numpy(np.float64)},  # clipped at 50km
        )
    )
    return out


def load_cdc_places() -> list[CoordBenchmark]:
    """CDC PLACES 2023 ZCTA health/behavior measures — one regression per measure."""
    df = load_config("cdc_places")  # already lowercased + lon/lat parsed at build time
    out: list[CoordBenchmark] = []
    for key, pref in CDC_PLACES_MEASURES.items():
        col = f"{pref.lower()}_crudeprev"
        if col not in df.columns:
            logger.warning("[cdc_places] missing column %s; skipping %s", col, key)
            continue
        m = df[col].notna() & df["lat"].notna() & df["lon"].notna()
        sub = df[m]
        out.append(
            CoordBenchmark(
                name=f"places-{key}",
                lat=sub["lat"].to_numpy(np.float64),
                lon=sub["lon"].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
            )
        )
    return out


def load_usavars() -> list[CoordBenchmark]:
    """MOSAIKS/USAVars — one regression benchmark per label (7 total)."""
    out: list[CoordBenchmark] = []
    skip = {"id", "lat", "lon"} | _CANONICAL_EXTRA
    for label in USAVARS_LABELS:
        df = load_config(f"usavars_{label}")
        df.columns = [str(c).strip() for c in df.columns]
        cl = {c.lower(): c for c in df.columns}
        cands = [
            c
            for c in df.columns
            if c.lower() not in skip
            and not c.startswith("Unnamed")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cands or not ({"lat", "lon"} <= set(cl)):
            continue
        value_col = next((c for c in cands if label in c.lower()), None)
        value_col = value_col or next((c for c in cands if c.lower() == "price"), None) or cands[0]
        lat_a = df[cl["lat"]].to_numpy(np.float64)
        lon_a = df[cl["lon"]].to_numpy(np.float64)
        val_a = df[value_col].to_numpy(np.float64)
        keep = val_a != USAVARS_NODATA
        lat_a, lon_a, val_a = lat_a[keep], lon_a[keep], val_a[keep]
        if label in USAVARS_LOG_LABELS:
            val_a = np.log1p(val_a)
        out.append(
            CoordBenchmark(name=f"mosaiks-{label}", lat=lat_a, lon=lon_a, tasks={label: val_a})
        )
    return out


def load_country() -> list[CoordBenchmark]:
    """Global country classification — 30k random land points, Natural Earth admin-0."""
    df = load_config("country")
    return [
        CoordBenchmark(
            name="country",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"country": df["country"].to_numpy()},
            task_type="classification",
        )
    ]


def load_ecoregions() -> list[CoordBenchmark]:
    """Global biome + ecoregion classification — RESOLVE Ecoregions2017, 30k points."""
    df = load_config("ecoregions")
    return [
        CoordBenchmark(
            name="ecoregions",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"biome": df["biome"].to_numpy(), "ecoregion": df["ecoregion"].to_numpy()},
            task_type="classification",
        )
    ]


def load_worldclim() -> list[CoordBenchmark]:
    """WorldClim v2.1 bioclim — annual mean temperature (bio1) + precipitation (bio12)."""
    df = load_config("worldclim_bio")
    return [
        CoordBenchmark(
            name=f"worldclim-bio{v}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={f"bio{v}": df[f"bio{v}"].to_numpy(np.float64)},
        )
        for v in (1, 12)
        if f"bio{v}" in df.columns
    ]


def load_soilgrids() -> list[CoordBenchmark]:
    """ISRIC SoilGrids v2 — soil organic carbon (soc) + pH (phh2o) regression."""
    df = load_config("soilgrids")
    return [
        CoordBenchmark(
            name=f"soilgrids-{p}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={p: df[p].to_numpy(np.float64)},
        )
        for p in ("soc", "phh2o")
        if p in df.columns
    ]


def load_deepmind() -> list[CoordBenchmark]:
    """DeepMind/AlphaEarth geospatial eval suite — one benchmark per eval config.

    Classification vs regression is inferred (integer label + few classes ->
    classification). Carries per-point year from the valid-time-start timestamp.
    """
    out: list[CoordBenchmark] = []
    for stem in DEEPMIND_EVAL_CONFIGS:
        df = load_config(f"dm_{stem}")
        if "label" not in df.columns:
            continue
        lon_col, lat_col = _lonlat_cols(df)
        ts_col = _first_col(df, "timestamp", "valid_time_start_ms")
        label = df["label"].to_numpy()
        integral = np.all(np.isfinite(label)) and np.allclose(label, np.round(label))
        is_clf = bool(integral and np.unique(label).size <= 100)
        # Missing source timestamps produce an all-null column; use year=None, not an all-NaN array.
        year = None
        if ts_col is not None:
            yr = pd.to_datetime(df[ts_col], unit="ms").dt.year
            if yr.notna().any():
                year = yr.to_numpy()
        out.append(
            CoordBenchmark(
                name=f"dm-{stem}",
                lat=df[lat_col].to_numpy(np.float64),
                lon=df[lon_col].to_numpy(np.float64),
                tasks={stem: label.astype(np.int64) if is_clf else label.astype(np.float64)},
                task_type="classification" if is_clf else "regression",
                year=year,
            )
        )
    return out


# Family name -> loader. A family groups benchmarks that share a source/protocol.
FAMILY_LOADERS: dict[str, Callable[[], list[CoordBenchmark]]] = {
    "pdfm": load_pdfm,
    "air_temp": load_air_temp,
    "california_housing": load_california_housing,
    "satclip": load_satclip,
    "sustainbench": load_sustainbench,
    "better_together": load_better_together,
    "cdc_places": load_cdc_places,
    "usavars": load_usavars,
    "country": load_country,
    "ecoregions": load_ecoregions,
    "worldclim": load_worldclim,
    "soilgrids": load_soilgrids,
    "deepmind": load_deepmind,
}


_BENCHMARK_TO_FAMILY: dict[str, str] = {
    name: family for family, names in FAMILY_BENCHMARKS.items() for name in names
}


def list_families() -> list[str]:
    """Return the registered benchmark family names."""
    return list(FAMILY_LOADERS)


def list_benchmarks() -> list[str]:
    """Return every individual benchmark name, in family/registration order."""
    return [name for family in FAMILY_LOADERS for name in FAMILY_BENCHMARKS[family]]


def load_benchmarks(names: str | list[str] = "all") -> list[CoordBenchmark]:
    """Expand a selection into a flat list of :class:`CoordBenchmark`.

    Args:
        names: ``"all"``, a comma-separated string, or a list. Each entry is a
            family name (loads every benchmark in it) or an individual benchmark
            name (loads only its family, then keeps just that benchmark). Only
            the families needed to satisfy the selection are fetched.

    Returns:
        Flat list of benchmarks, deduplicated by name, in family order.
    """
    if isinstance(names, str):
        selection = (
            list(FAMILY_LOADERS)
            if names == "all"
            else [n.strip() for n in names.split(",") if n.strip()]
        )
    else:
        selection = list(names)

    families_to_load: list[str] = []
    name_filter: dict[str, set[str]] = {}
    for entry in selection:
        if entry in FAMILY_LOADERS:
            family = entry
            name_filter[family] = set()  # empty filter -> keep all of the family
        elif entry in _BENCHMARK_TO_FAMILY:
            family = _BENCHMARK_TO_FAMILY[entry]
            if family not in name_filter or name_filter[family]:
                name_filter.setdefault(family, set()).add(entry)
        else:
            raise KeyError(
                f"unknown coordbench selection {entry!r}; not a family "
                f"({sorted(FAMILY_LOADERS)}) or benchmark name"
            )
        if family not in families_to_load:
            families_to_load.append(family)

    out: list[CoordBenchmark] = []
    seen: set[str] = set()
    for family in families_to_load:
        wanted = name_filter.get(family, set())
        for bench in FAMILY_LOADERS[family]():
            if bench.name in seen or (wanted and bench.name not in wanted):
                continue
            seen.add(bench.name)
            out.append(bench)
    return out
