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

V1 uses the pinned ``calebrob6/geobenchv1-webdataset`` mirror.

Use ``--datasets`` to select a GeoBench subset.
"""

import logging
import shutil
import zipfile
from pathlib import Path

from torchgeo.datasets import RESISC45, EuroSAT, EuroSATSpatial, UCMerced

from geo_eval_harness.datasets._v1_webdataset import download_sharded_root
from geo_eval_harness.datasets.geobench_v2 import V2_REVISIONS, list_v2_datasets, v2_checksums
from geo_eval_harness.datasets.hotosm_buildings import (
    UPSTREAM_SPLITS,
    HOTOSMBuildings,
    convert_hotosm_split,
)
from geo_eval_harness.hub import fetch_pinned, load_checksums

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
DIRECT_DATASETS: tuple[str, ...] = ("aid", "hotosm_buildings")
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
