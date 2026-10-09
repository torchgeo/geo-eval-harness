"""Download benchmark datasets into ``data/``.

Targets:

- ``geobench_v1``: JSON-metadata shards under ``<output>/classification_v1.0_wds/``.
- ``geobench_v2``: datasets from ``aialliance/<name>`` under ``<output>/geobenchv2/``.
- ``eurosat`` — torchgeo's EuroSAT downloader, into ``<output>/eurosat``.
- ``resisc45`` — torchgeo's NWPU-RESISC45 downloader, into ``<output>/resisc45``.
- ``aid`` — pinned ``isaaccorley/aid`` rehost, into ``<output>/aid``.
- ``ucmerced`` — torchgeo's UC Merced downloader, into ``<output>/ucmerced``.
- ``hotosm_buildings`` — pinned ``hotosm/vhr-building-segmentation`` parquet, converted
  into compact per-split files under ``<output>/hotosm_buildings``.
- ``open_cities`` — Open Cities AI Challenge tier-1 scenes from source.coop, plus native-grid
  masks built from the labels, into ``<output>/open_cities``.

V1 uses the pinned ``calebrob6/geobenchv1-webdataset`` mirror.

Use ``--datasets`` to select a GeoBench subset.
"""

import logging
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from torchgeo.datasets import RESISC45, EuroSAT, EuroSATSpatial, UCMerced

from geo_eval_harness.datasets import open_cities
from geo_eval_harness.datasets._v1_webdataset import download_sharded_root
from geo_eval_harness.datasets.geobench_v2 import V2_REVISIONS, list_v2_datasets, v2_checksums
from geo_eval_harness.datasets.hotosm_buildings import (
    UPSTREAM_SPLITS,
    HOTOSMBuildings,
    convert_hotosm_split,
)
from geo_eval_harness.hub import fetch_pinned, load_checksums, verify_checksums

logger = logging.getLogger(__name__)

GEOBENCH_V2_REPO_PREFIX = "aialliance"

AID_REPO = "isaaccorley/aid"
AID_REVISION = "e6767964e40f567a18eac6e6ca5fce397e4ce411"

HOTOSM_REPO = "hotosm/vhr-building-segmentation"
HOTOSM_REVISION = "8d3e64e5c69aa37209953cce3a48df1092bc7c94"

DEFAULT_V2_DATASETS: tuple[str, ...] = tuple(list_v2_datasets())

V1_DATASETS: tuple[str, ...] = (
    "m-eurosat",
    "m-forestnet",
    "m-so2sat",
    "m-pv4ger",
    "m-brick-kiln",
    "m-bigearthnet",
)
TORCHGEO_DATASETS: tuple[str, ...] = ("eurosat", "resisc45", "ucmerced")
DIRECT_DATASETS: tuple[str, ...] = ("aid", "hotosm_buildings", "open_cities")
DOWNLOADABLE_DATASETS: tuple[str, ...] = (
    V1_DATASETS + DEFAULT_V2_DATASETS + TORCHGEO_DATASETS + DIRECT_DATASETS
)


def _validate_names(names: list[str]) -> list[str]:
    """Validate and deduplicate names before creating download roots."""
    if not names:
        raise ValueError("datasets must contain at least one dataset name")
    unique = list(dict.fromkeys(names))
    unknown = sorted(set(unique) - set(DOWNLOADABLE_DATASETS))
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(DOWNLOADABLE_DATASETS)}"
        )
    return unique


def download_geobench_v1(output_dir: Path, datasets: list[str] | None = None) -> None:
    """Download verified, pickle-free GeoBench V1 shards to ``output_dir``.

    Args:
        output_dir: Benchmark data root (typically ``data/``).
        datasets: Specific dataset names to fetch from the sharded mirror.
            ``None`` downloads all six classification datasets.

    Raises:
        ValueError: If dataset names are invalid or an archive checksum fails.
    """
    download_sharded_root(Path(output_dir) / "classification_v1.0_wds", datasets)


def download_geobench_v2_dataset(name: str, v2_root: Path) -> None:
    """Download and verify a pinned GeoBench V2 dataset into ``v2_root/<name>``."""
    if name not in V2_REVISIONS:
        raise ValueError(f"Unknown GeoBench V2 dataset: {name}")
    target = v2_root / name
    target.mkdir(parents=True, exist_ok=True)
    repo_id = f"{GEOBENCH_V2_REPO_PREFIX}/{name}"
    logger.info("Downloading %s@%s -> %s", repo_id, V2_REVISIONS[name], target)
    fetch_pinned(
        repo_id, V2_REVISIONS[name], target, v2_checksums(name), complete_glob="*.tortilla"
    )


def download_geobench_v2(output_dir: Path, datasets: list[str] | None = None) -> None:
    """Download GeoBench V2 datasets into ``output_dir/geobenchv2/<name>``.

    Args:
        output_dir: Benchmark data root (typically ``data/``).
        datasets: Specific dataset names to fetch. ``None`` downloads
            :data:`DEFAULT_V2_DATASETS`.
    """
    if datasets is None:
        names = list(DEFAULT_V2_DATASETS)
    elif not datasets:
        raise ValueError("datasets must contain at least one GeoBench V2 dataset name.")
    else:
        names = list(dict.fromkeys(datasets))

    unknown = sorted(set(names) - set(DEFAULT_V2_DATASETS))
    if unknown:
        raise ValueError(
            f"Unknown GeoBench V2 dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(DEFAULT_V2_DATASETS)}"
        )
    v2_root = Path(output_dir) / "geobenchv2"
    v2_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %d GeoBench v2 dataset(s) to %s", len(names), v2_root)
    for name in names:
        download_geobench_v2_dataset(name, v2_root)
    logger.info("GeoBench v2 download complete.")


def download_eurosat(output_dir: Path) -> None:
    """Download EuroSAT imagery and both standard/spatial split definitions."""
    target = Path(output_dir) / "eurosat"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading torchgeo EuroSAT -> %s", target)
    for dataset_cls in (EuroSAT, EuroSATSpatial):
        for split in ("train", "val", "test"):
            dataset_cls(root=str(target), split=split, download=True)
    logger.info("EuroSAT download complete.")


def download_resisc45(output_dir: Path) -> None:
    """Download torchgeo's NWPU-RESISC45 into ``output_dir/resisc45`` for all splits.

    All splits share one 427 MB archive; verify its checksum before loading samples.
    """
    target = Path(output_dir) / "resisc45"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading torchgeo RESISC45 -> %s", target)
    for split in ("train", "val", "test"):
        RESISC45(root=str(target), split=split, download=True, checksum=True)
    logger.info("RESISC45 download complete.")


def download_aid(output_dir: Path) -> None:
    """Download the pinned ``isaaccorley/aid`` rehost into ``output_dir/aid``."""
    target = Path(output_dir) / "aid"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s@%s -> %s", AID_REPO, AID_REVISION, target)
    fetch_pinned(AID_REPO, AID_REVISION, target, load_checksums("aid.sha256"))
    with zipfile.ZipFile(target / "AID.zip") as archive:
        archive.extractall(target)
    logger.info("AID download complete.")


def download_hotosm_buildings(output_dir: Path) -> None:
    """Download the pinned HOTOSM parquet and convert it into ``output_dir/hotosm_buildings``.

    The upstream shards (5.9 GB) are deleted after conversion (about 1.3 GB).
    """
    target = Path(output_dir) / "hotosm_buildings"
    upstream = target / "upstream"
    logger.info("Downloading %s@%s -> %s", HOTOSM_REPO, HOTOSM_REVISION, upstream)
    fetch_pinned(
        HOTOSM_REPO,
        HOTOSM_REVISION,
        upstream,
        load_checksums("hotosm_buildings.sha256"),
        allow_patterns=["data/*.parquet"],
    )
    for split, prefix in UPSTREAM_SPLITS.items():
        rows = convert_hotosm_split(
            (upstream / "data").glob(f"{prefix}-*.parquet"),
            target / f"{split}.parquet",
            expected_rows=HOTOSMBuildings.split_sizes[split],
        )
        logger.info("Converted HOTOSM buildings %s split (%d rows).", split, rows)
    shutil.rmtree(upstream)
    logger.info("HOTOSM buildings download complete.")


def fetch_open_cities_file(relative: str, root: Path) -> Path:
    """Download one upstream ``<city>/<scene>.tif|.geojson`` into ``root``, resuming partial files.

    Existing complete files are not downloaded again.
    """
    dst = root / relative
    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": open_cities.USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(open_cities.upstream_url(relative), headers=headers)
    with urllib.request.urlopen(request) as response:
        # A server that ignores the range answers 200 with the whole file.
        mode = "ab" if response.status == 206 else "wb"
        with part.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=1 << 20)
    part.replace(dst)
    return dst


def _check_open_cities_masks(root: Path, index: pd.DataFrame, tolerance: float = 0.002) -> None:
    """Compare each split's building fraction in the written masks with the index."""
    fractions = np.zeros(len(index))
    for scene, rows in index.groupby(["city", "scene"]).groups.items():
        with rasterio.open(root / scene[0] / f"{scene[1]}_mask.tif") as mask:
            for i in rows:
                chip = mask.read(
                    1,
                    window=Window(
                        index.col_off[i],
                        index.row_off[i],
                        open_cities.CHIP_SIZE,
                        open_cities.CHIP_SIZE,
                    ),
                )
                fractions[i] = (chip == 1).mean()
    for split, measured in index.assign(measured=fractions).groupby("split"):
        delta = abs(measured.measured.mean() - measured.building_frac.mean())
        if delta > tolerance:
            raise ValueError(
                f"Open Cities {split}: mask building fraction differs from the index by {delta:.4f}. "
                f"Remove the *_mask.tif files under {root} and retry the download."
            )
        logger.info("Open Cities %s masks match the index (|delta| = %.4f).", split, delta)


def download_open_cities(output_dir: Path, workers: int = 4) -> None:
    """Download the tier-1 Open Cities scenes and build their masks in ``output_dir/open_cities``.

    Imagery (about 34 GB) is kept byte-identical and verified against the packaged
    checksums. Masks are rasterised on each scene's native grid for the indexed chips only.
    """
    target = Path(output_dir) / "open_cities"
    checksums = load_checksums("open_cities.sha256")
    logger.info("Downloading %d Open Cities files -> %s", len(checksums), target)

    def fetch(item: tuple[str, str]) -> None:
        fetch_open_cities_file(item[0], target)
        verify_checksums(target, {item[0]: item[1]})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, checksums.items()))

    index = open_cities.load_index()

    def build(scene: tuple[str, str]) -> None:
        city, name = scene
        rows = index[(index.city == city) & (index.scene == name)]
        open_cities.build_scene_mask(
            target / city / f"{name}.tif", target / city / f"{name}.geojson", rows
        )
        logger.info("Built mask for %s/%s (%d chips).", city, name, len(rows))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(build, index[["city", "scene"]].drop_duplicates().itertuples(index=False)))
    _check_open_cities_masks(target, index)
    logger.info("Open Cities download complete.")


def download_ucmerced(output_dir: Path) -> None:
    """Download verified UC Merced imagery and splits into ``output_dir/ucmerced``."""
    target = Path(output_dir) / "ucmerced"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading torchgeo UCMerced -> %s", target)
    for split in ("train", "val", "test"):
        UCMerced(root=str(target), split=split, download=True, checksum=True)
    logger.info("UCMerced download complete.")


def download_datasets(names: list[str], output_dir: Path = Path("data")) -> None:
    """Download individually named benchmark datasets.

    Args:
        names: Dataset names from :data:`DOWNLOADABLE_DATASETS`.
        output_dir: Benchmark data root.

    Raises:
        ValueError: If a name is unknown or *names* is empty.
    """
    selected = _validate_names(names)
    v1 = [name for name in selected if name in V1_DATASETS]
    v2 = [name for name in selected if name in DEFAULT_V2_DATASETS]
    for name in v1:
        download_geobench_v1(output_dir, datasets=[name])
    if v2:
        download_geobench_v2(output_dir, datasets=v2)
    if "eurosat" in selected:
        download_eurosat(output_dir)
    if "resisc45" in selected:
        download_resisc45(output_dir)
    if "aid" in selected:
        download_aid(output_dir)
    if "ucmerced" in selected:
        download_ucmerced(output_dir)
    if "hotosm_buildings" in selected:
        download_hotosm_buildings(output_dir)
    if "open_cities" in selected:
        download_open_cities(output_dir)
