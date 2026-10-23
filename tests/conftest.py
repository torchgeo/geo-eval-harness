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
