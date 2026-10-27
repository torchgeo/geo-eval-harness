"""Per-dataset geographic metadata for the benchmark datasets.

Map where each registered dataset's imagery comes from.

The extraction script writes dataset JSON and ``index.json`` to :data:`STORE_DIR`.

Commit the store so maps and figures work without raw imagery.

Track registry coverage with :func:`~geo_eval_harness.datasets.list_datasets`.

Every dataset needs a status; ``tests/test_geography.py`` detects missing records.

Three statuses:

* ``extracted`` — coordinates were read from the raw files.
* ``no_geo`` — no coordinates are available; see :attr:`GeoRecord.reason` and :data:`NO_GEO`.
* ``not_downloaded`` — registered, but no data files were found locally.

Three coordinate sources are handled:

* **V2**: read coordinates and place labels from ``.tortilla`` metadata, not imagery.
* **V1**: reproject JSON affine/CRS origins from shards or HDF5 to EPSG:4326.
* **HOTOSM buildings**: read the ``lon``/``lat``/``country`` columns that
  ``geo-eval-harness download hotosm_buildings`` writes into its split files.
* **PLANTEUR**: read the patch centres and islands in the ``metadata.csv`` that
  ``geo-eval-harness download planteur`` writes.

Unchanged raw data produces byte-identical JSON.
"""

import glob
import json
import logging
import os
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .datasets import get_bench_dataset_class, list_datasets
from .datasets._metadata import read_hdf5_metadata
from .datasets.loading import download_command

logger = logging.getLogger(__name__)

STORE_DIR = Path(__file__).resolve().parents[2] / "docs" / "_static" / "_dataset_geography"
INDEX_NAME = "index.json"

Status = Literal["extracted", "no_geo", "not_downloaded"]

NO_GEO: dict[str, str] = {
    "so2sat": "tortilla metadata has no lon/lat columns",
    "m-so2sat": "every sample has transform=None and crs=None",
    "resisc45": (
        "scene-classification JPEGs with no georeferencing: no sampled image "
        "carries any EXIF block, and the upstream release ships no coordinate table"
    ),
    "aid": (
        "scene-classification JPEGs with no georeferencing: no sampled image "
        "carries any EXIF block, and the upstream release ships no coordinate table"
    ),
    "ucmerced": (
        "land-use TIFFs with no georeferencing: all 2,100 images lack GeoTIFF tags, "
        "EXIF GPS, and XMP location fields; the release ships no coordinate table"
    ),
}

# Re-splits reuse coordinates from the dataset with the same imagery.
GEO_ALIAS: dict[str, str] = {
    "eurosat-spatial": "m-eurosat",
    "eurosat": "m-eurosat",
    **{
        f"planteur_loio_{fold}": "planteur"
        for fold in ("guadeloupe", "martinique", "guyane", "reunion", "mayotte")
    },
}

BIN_RES = 0.25  # Degrees per density-map cell.
POINT_CAP = 2000  # per-dataset cap on individually plotted samples
POINT_DP = 3  # Decimal places in coordinates (~110 m precision).

# Allow isolated corrupt samples, but fail if errors suggest a broken data layout.
MAX_READ_FAILURE_RATE = 0.01


@dataclass(frozen=True)
class GeoRecord:
    """Geographic footprint of one benchmark dataset.

    Attributes:
        name: Dataset identifier, as registered (e.g. ``"m-eurosat"``).
        status: ``"extracted"``, ``"no_geo"``, or ``"not_downloaded"``.
        version: Source family — ``"v1"``, ``"v2"``, ``"torchgeo"``, ``"hotosm"``, or
            ``"planteur"``.
            ``None`` when nothing was read.
        n: Number of geolocated samples found.
        reason: Why coordinates are unavailable; always set for ``"no_geo"``.
        alias_of: Set when this dataset borrows another's geometry (see
            :data:`GEO_ALIAS`).
        bbox: ``[min_lon, min_lat, max_lon, max_lat]``, or ``None``.
        continents: Percentage of samples per continent, descending.
        bins: ``[x, y, count]`` density cells on a :data:`BIN_RES` grid, where
            ``x = floor((lon + 180) / BIN_RES)`` and
            ``y = floor((lat + 90) / BIN_RES)``.
        points: Deterministic subsample, each ``[lon, lat]`` or
            ``[lon, lat, place]``, capped at :data:`POINT_CAP`.
    """

    name: str
    status: Status
    version: str | None = None
    n: int = 0
    reason: str | None = None
    alias_of: str | None = None
    bbox: list[float] | None = None
    continents: dict[str, float] = field(default_factory=dict)
    bins: list[list[int]] = field(default_factory=list)
    points: list[list[Any]] = field(default_factory=list)

    def to_json(self) -> dict:
        """Return the JSON-serialisable form written to the store."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "GeoRecord":
        """Rebuild a record from its stored form, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def _extract_v2(paths: list[str]) -> dict | None:
    """Read lon/lat (and any place labels) from a dataset's tortilla files."""
    import tacoreader

    lons: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    places: list[np.ndarray] = []

    for path in sorted(paths):
        df = tacoreader.load(path)
        if "lon" not in df.columns or "lat" not in df.columns:
            return None
        # TortillaDataFrame does not support standard pandas operations; work with NumPy arrays.
        lons.append(np.asarray(df["lon"], dtype=float))
        lats.append(np.asarray(df["lat"], dtype=float))
        place_col = next((c for c in ("country", "city", "rai:admin0") if c in df.columns), None)
        if place_col is not None:
            places.append(np.asarray(df[place_col], dtype=object))
        else:
            places.append(np.full(len(lons[-1]), None, dtype=object))

    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    place = np.concatenate(places)

    ok = np.isfinite(lon) & np.isfinite(lat)
    return {"lon": lon[ok], "lat": lat[ok], "place": place[ok], "version": "v2"}


def _extract_hotosm(paths: list[Path]) -> dict:
    """Read tile centres and countries from the converted ``hotosm_buildings`` splits."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables(
        pq.read_table(path, columns=["lon", "lat", "country"]) for path in paths
    )
    return {
        "lon": table["lon"].to_numpy(),
        "lat": table["lat"].to_numpy(),
        "place": np.asarray(table["country"].to_pylist(), dtype=object),
        "version": "hotosm",
    }


def _extract_planteur(path: Path) -> dict:
    """Read patch centres and islands from the converted ``planteur`` metadata."""
    import pandas as pd

    frame = pd.read_csv(path)
    return {
        "lon": frame["lon"].to_numpy(),
        "lat": frame["lat"].to_numpy(),
        "place": frame["island"].to_numpy(dtype=object),
        "version": "planteur",
    }


def _extract_converted(name: str, directory: Path) -> dict | None:
    """Read coordinates written by a converting downloader, or ``None`` if they are absent."""
    if name == "hotosm_buildings":
        splits = [directory / f"{split}.parquet" for split in ("train", "val", "test")]
        return _extract_hotosm(splits) if all(path.is_file() for path in splits) else None
    if name == "planteur" and (directory / "metadata.csv").is_file():
        return _extract_planteur(directory / "metadata.csv")
    return None


def _metadata_origin(meta: dict) -> tuple[float | None, float | None, str]:
    """Return the first usable affine origin and CRS from JSON metadata."""
    for entry in meta.values():
        if not isinstance(entry, dict):
            continue
        transform, crs = entry.get("transform"), entry.get("crs")
        if transform is None or crs is None:
            continue
        if not isinstance(crs, str):
            raise TypeError("GeoBench JSON metadata CRS must be a string.")
        if (
            not isinstance(transform, list)
            or len(transform) not in (6, 9)
            or any(not isinstance(value, (int, float)) for value in transform)
        ):
            raise ValueError("GeoBench JSON metadata transform must have 6 or 9 coefficients.")
        return float(transform[2]), float(transform[5]), crs
    return None, None, "NOGEO"


def _v1_origin(path: str) -> tuple[float | None, float | None, str]:
    """Return the raster origin and CRS for one custom V1 HDF5 sample."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            meta = read_hdf5_metadata(f.attrs)
        return _metadata_origin(meta)
    except (KeyError, TypeError, ValueError) as exc:  # allow-except: report invalid sample metadata
        return None, None, f"ERR {type(exc).__name__}: {exc}"


def _v1_shard_origins(path: str) -> list[tuple[float | None, float | None, str]]:
    """Read sample origins from JSON shard members without loading image arrays."""
    from .datasets._metadata import decode_metadata

    results: dict[str, tuple[float | None, float | None, str]] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if member.name.endswith(".bands.npz"):
                sample_id = member.name.removesuffix(".bands.npz")
                results.setdefault(
                    sample_id,
                    (
                        None,
                        None,
                        "ERR ValueError: missing .meta.json metadata; replace legacy V1 data",
                    ),
                )
            elif member.name.endswith(".meta.json"):
                sample_id = member.name.removesuffix(".meta.json")
                try:
                    stream = archive.extractfile(member)
                    if stream is None:
                        results[sample_id] = (
                            None,
                            None,
                            f"ERR ValueError: Not a metadata file: {member.name}",
                        )
                        continue
                    with stream:
                        results[sample_id] = _metadata_origin(decode_metadata(stream.read()))
                except (
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:  # allow-except: report invalid sample metadata
                    results[sample_id] = (None, None, f"ERR {type(exc).__name__}: {exc}")
    if not results:
        raise ValueError(f"No GeoBench V1 sample metadata in {path}.")
    return list(results.values())


def _extract_v1(name: str, files: list[str], workers: int) -> dict | None:
    """Extract and reproject raster origins for every sample of a V1 dataset."""
    from pyproj import Transformer

    with ProcessPoolExecutor(max_workers=workers) as pool:
        if files and files[0].endswith(".tar"):
            results = [origin for shard in pool.map(_v1_shard_origins, files) for origin in shard]
        else:
            results = list(pool.map(_v1_origin, files, chunksize=64))

    errors = [r[2] for r in results if r[0] is None and r[2].startswith("ERR")]
    if errors:
        rate = len(errors) / len(results)
        logger.warning(
            "%s: %d read errors (%.1f%%); first: %s", name, len(errors), 100 * rate, errors[0]
        )
        if rate > MAX_READ_FAILURE_RATE:
            raise RuntimeError(f"{name}: {rate:.1%} of samples failed to read; first: {errors[0]}")

    usable = [(x, y, epsg) for x, y, epsg in results if x is not None and y is not None]
    if not usable:
        return None

    # A dataset can span several coordinate systems; reproject each group separately.
    by_crs: dict[str, list[tuple[float, float]]] = {}
    for x, y, epsg in usable:
        by_crs.setdefault(epsg, []).append((x, y))

    lons: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    for epsg, coords in sorted(by_crs.items()):
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        if epsg.upper() in ("EPSG:4326", "4326"):
            lons.append(xs)
            lats.append(ys)
            continue
        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs, ys)
        lons.append(np.asarray(lon))
        lats.append(np.asarray(lat))

    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    ok = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    return {
        "lon": lon[ok],
        "lat": lat[ok],
        "place": np.full(int(ok.sum()), None, dtype=object),
        "version": "v1",
    }


def _natural_earth_countries() -> str:
    """Locate the Natural Earth admin-0 shapefile, downloading if needed."""
    import cartopy.io.shapereader as shpreader

    return shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")


def _attribute_continents(lon: np.ndarray, lat: np.ndarray) -> Counter:
    """Count samples by continent using Natural Earth country boundaries."""
    import geopandas as gpd

    shp = _natural_earth_countries()
    world = gpd.read_file(shp)[["CONTINENT", "geometry"]]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    joined = gpd.sjoin(pts, world, how="left", predicate="within")
    # A point can land in overlapping polygons; keep one row per point.
    joined = joined[~joined.index.duplicated(keep="first")]
    return Counter(joined["CONTINENT"].fillna("Ocean / unmatched").tolist())


def _dataset_dir(name: str) -> Path | None:
    """Return the on-disk directory for a dataset, or ``None`` if absent.

    V1 prefers the same sharded cache as the benchmark; other layouts use
    the dataset's fixed ``data_root()``.
    """
    from .datasets.geobench_v1 import V1_SHARDED_ROOT, _V1Dataset

    cls = get_bench_dataset_class(name)
    if issubclass(cls, _V1Dataset):
        sharded = V1_SHARDED_ROOT / name
        if any(sharded.glob("shard_*.tar")):
            return sharded
    root = cls.data_root()
    # V1/V2 roots contain dataset subdirectories; torchgeo roots point directly to a dataset.
    candidate = root / name
    if candidate.is_dir():
        return candidate
    return root if root.is_dir() else None


def _build_record(name: str, data: dict, continents: Counter) -> GeoRecord:
    """Summarize sample locations as density bins, map points, and continent shares."""
    lon, lat = data["lon"], data["lat"]

    xi = np.floor((lon + 180.0) / BIN_RES).astype(int)
    yi = np.floor((lat + 90.0) / BIN_RES).astype(int)
    counts = Counter(zip(xi.tolist(), yi.tolist(), strict=True))
    bins = [[int(x), int(y), int(c)] for (x, y), c in sorted(counts.items())]

    # Deterministic subsample so re-runs are byte-identical.
    n = len(lon)
    idx = np.linspace(0, n - 1, POINT_CAP).astype(int) if n > POINT_CAP else np.arange(n)

    place = data["place"]
    points: list[list[Any]] = []
    for j in idx.tolist():
        row: list[Any] = [round(float(lon[j]), POINT_DP), round(float(lat[j]), POINT_DP)]
        p = place[j] if j < len(place) else None
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            row.append(str(p))
        points.append(row)

    total = sum(continents.values()) or 1
    shares = {
        k: round(100.0 * v / total, 2)
        for k, v in sorted(continents.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    return GeoRecord(
        name=name,
        status="extracted",
        version=data["version"],
        n=int(n),
        continents=shares,
        bins=bins,
        points=points,
        bbox=[
            round(float(lon.min()), 3),
            round(float(lat.min()), 3),
            round(float(lon.max()), 3),
            round(float(lat.max()), 3),
        ],
    )


def extract_geography(name: str, *, workers: int | None = None) -> GeoRecord:
    """Extract the geographic footprint of one registered dataset.

    Args:
        name: Registered dataset identifier.
        workers: Processes used for the V1 metadata scan. Defaults to
            ``min(32, os.cpu_count())``.

    Returns:
        A :class:`GeoRecord` with extracted coordinates or a reason they are unavailable.

    Raises:
        FileNotFoundError: If dataset imagery is missing.
    """
    workers = workers or min(32, os.cpu_count() or 8)

    if name in NO_GEO:
        return GeoRecord(name=name, status="no_geo", reason=NO_GEO[name])

    if name in GEO_ALIAS:
        target = GEO_ALIAS[name]
        record = extract_geography(target, workers=workers)
        return replace(record, name=name, alias_of=target)

    command = (
        "geo-eval-harness download geobench_v1" if name.startswith("m-") else download_command(name)
    )
    directory = _dataset_dir(name)
    if directory is None:
        raise FileNotFoundError(f"No imagery for {name!r}. Run `{command}`.")

    tortillas = sorted(glob.glob(str(directory / "*.tortilla")))
    shards = sorted(glob.glob(str(directory / "shard_*.tar")))
    hdf5s = sorted(glob.glob(str(directory / "*.hdf5")))
    converted = _extract_converted(name, directory)

    if tortillas:
        data = _extract_v2(tortillas)
        if data is None:
            return GeoRecord(
                name=name, status="no_geo", reason="tortilla metadata has no lon/lat columns"
            )
    elif shards or hdf5s:
        data = _extract_v1(name, shards or hdf5s, workers)
        if data is None:
            return GeoRecord(
                name=name, status="no_geo", reason="no sample carries a usable transform/crs"
            )
    elif converted is not None:
        data = converted
    else:
        raise FileNotFoundError(
            f"No .tortilla, V1 shard, or .hdf5 files under {directory}. Run `{command}`."
        )

    if len(data["lon"]) == 0:
        return GeoRecord(name=name, status="no_geo", reason="all coordinates were non-finite")

    return _build_record(name, data, _attribute_continents(data["lon"], data["lat"]))


def _record_path(name: str, store_dir: Path | None = None) -> Path:
    return (store_dir or STORE_DIR) / f"{name}.json"


def write_record(record: GeoRecord, store_dir: Path | None = None) -> Path:
    """Write one record to the store and return its path."""
    path = _record_path(record.name, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json(), sort_keys=True, separators=(",", ":")))
    return path


def build_index(store_dir: Path | None = None) -> dict:
    """Rebuild ``index.json`` from the per-dataset files and return it.

    The summary supports map legends and coverage warnings without loading per-dataset point lists.
    """
    directory = store_dir or STORE_DIR
    records = [
        GeoRecord.from_json(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
        if path.name != INDEX_NAME
    ]
    entries = [
        {
            "name": record.name,
            "status": record.status,
            "version": record.version,
            "n": record.n,
            "reason": record.reason,
            "alias_of": record.alias_of,
            "bbox": record.bbox,
            "continents": record.continents,
        }
        for record in records
    ]

    overall: dict[str, float] = {}
    for record in records:
        for continent, share in record.continents.items():
            overall[continent] = overall.get(continent, 0.0) + share * record.n / 100.0
    grand = sum(overall.values()) or 1

    index = {
        "bin_res": BIN_RES,
        "point_cap": POINT_CAP,
        "datasets": entries,
        "totals": {
            "datasets": len(entries),
            "extracted": sum(1 for record in records if record.status == "extracted"),
            "samples": sum(record.n for record in records),
            "continents": {
                k: round(100.0 * v / grand, 2)
                for k, v in sorted(overall.items(), key=lambda kv: (-kv[1], kv[0]))
            },
        },
    }
    (directory / INDEX_NAME).write_text(json.dumps(index, sort_keys=True, separators=(",", ":")))
    return index


def load_geography(name: str, store_dir: Path | None = None) -> GeoRecord:
    """Load one dataset's record from the committed store.

    Raises:
        FileNotFoundError: If the dataset has no record yet.
    """
    path = _record_path(name, store_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No geography record for {name!r} at {path}. "
            "Run scripts/extract_dataset_geography.py to generate it."
        )
    return GeoRecord.from_json(json.loads(path.read_text()))


def list_geography(store_dir: Path | None = None) -> dict[str, GeoRecord]:
    """Load every record in the store, keyed by dataset name."""
    directory = store_dir or STORE_DIR
    if not directory.is_dir():
        return {}
    records = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == INDEX_NAME:
            continue
        record = GeoRecord.from_json(json.loads(path.read_text()))
        records[record.name] = record
    return records


def missing_datasets(store_dir: Path | None = None) -> set[str]:
    """Return registered datasets that have no record yet."""
    return set(list_datasets()) - set(list_geography(store_dir))


__all__ = [
    "GEO_ALIAS",
    "INDEX_NAME",
    "NO_GEO",
    "STORE_DIR",
    "GeoRecord",
    "build_index",
    "extract_geography",
    "list_geography",
    "load_geography",
    "missing_datasets",
    "write_record",
]
