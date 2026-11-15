"""Shared in-process command runner for the offline integration scenarios."""

from collections.abc import Callable
from pathlib import Path

import pytest

from geo_eval_harness.cli import main
from tests.support.numerical import isolated_torch_rng as isolated_torch_rng


@pytest.fixture
def cli_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_torch_rng: None
) -> Callable[[list[str]], None]:
    """Return the CLI entry point with ``tmp_path`` as its offline working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return main
