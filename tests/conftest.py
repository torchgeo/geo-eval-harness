"""Pytest configuration and fixtures for geo-eval-harness tests."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

# Dataset paths are relative to the directory where pytest is run.
GEOBENCH_ROOT = Path("data/classification_v1.0_wds")
GEOBENCH_V2_ROOT = Path("data/geobenchv2")
EUROSAT_ROOT = Path("data/eurosat")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("accuracy baselines")
    group.addoption(
        "--accuracy-device",
        default="cpu",
        help="Device for accuracy_check benchmarks (default: cpu).",
    )
    group.addoption(
        "--accuracy-timeout",
        type=int,
        default=600,
        help="Timeout in seconds per accuracy_check benchmark (default: 600).",
    )


@pytest.fixture
def geobench_root() -> str:
    if not (GEOBENCH_ROOT / "m-eurosat").exists():
        pytest.skip(f"GeoBench V1 data not found at {GEOBENCH_ROOT}")
    return str(GEOBENCH_ROOT)


@pytest.fixture
def geobench_v2_root() -> str:
    if not GEOBENCH_V2_ROOT.exists():
        pytest.skip(f"GeoBench V2 data not found at {GEOBENCH_V2_ROOT}")
    return str(GEOBENCH_V2_ROOT)


@pytest.fixture
def small_partition() -> str:
    return "0.01x_train"


@pytest.fixture
def small_openearthmap_sar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound native raster size and official memberships for offline SAR workflows."""
    from geo_eval_harness.datasets.openearthmap_sar import OpenEarthMapSAR

    monkeypatch.setattr(OpenEarthMapSAR, "split_sizes", {"train": 3, "val": 2, "test": 2})
    monkeypatch.setattr(OpenEarthMapSAR, "raster_size", 16)


@pytest.fixture
def small_loveda(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Use 2/1/2 tiles; torchgeo requires both scene folders in each release."""
    from geo_eval_harness.datasets.loveda import LoveDA

    members = {
        "train": "Train/Rural/0.png Train/Urban/2.png",
        "val": "Train/Urban/3.png",
        "test": "Val/Rural/4.png Val/Urban/5.png",
    }
    directory = tmp_path_factory.mktemp("loveda_splits")
    for split, paths in members.items():
        (directory / f"{split}.txt").write_text(paths)
    monkeypatch.setattr(LoveDA, "splits_dir", directory)


@pytest.fixture
def small_levir_cd_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use 16 x 16 rasters cut into four 8 x 8 tiles, with 3/2/2 images per split."""
    from geo_eval_harness.datasets.levir_cd_post import LEVIRCDPost

    monkeypatch.setattr(LEVIRCDPost, "image_counts", {"train": 3, "val": 2, "test": 2})
    monkeypatch.setattr(LEVIRCDPost, "split_sizes", {"train": 12, "val": 8, "test": 8})
    monkeypatch.setattr(LEVIRCDPost, "raster_size", 16)
    monkeypatch.setattr(LEVIRCDPost, "tile_size", 8)


@pytest.fixture
def small_sen1floods11(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the official split sizes for offline Sen1Floods11 workflows."""
    from geo_eval_harness.datasets.sen1floods11 import Sen1Floods11Base

    monkeypatch.setattr(Sen1Floods11Base, "split_sizes", {"train": 3, "val": 2, "test": 2})


@pytest.fixture
def small_bright_post_sar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install tiny BRIGHT memberships and raster size for offline workflows."""
    from geo_eval_harness.datasets.bright_post_sar import BRIGHTPostSAR

    ids = {
        "train": ["a_1", "a_2", "b_1"],
        "val": ["a_3", "b_2"],
        "test": ["b_3", "c_1"],
    }
    monkeypatch.setattr(BRIGHTPostSAR, "split_ids", classmethod(lambda cls, split: ids[split]))
    monkeypatch.setattr(BRIGHTPostSAR, "raster_size", 16)


@pytest.fixture(scope="session", autouse=True)
def test_environment() -> Iterator[None]:
    """Keep child processes on this checkout and avoid CPU oversubscription."""
    previous = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as patch:
        root = Path(__file__).resolve().parents[1]
        patch.setenv(
            "PYTHONPATH", os.pathsep.join((str(root / "src"), str(root))), prepend=os.pathsep
        )
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            patch.setenv(name, "1")
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous)
