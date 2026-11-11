"""Workflow coverage for independently reviewable stacked pull requests."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


@pytest.mark.parametrize("name", ["ci", "docs"])
def test_checks_cover_every_pull_request_base(name: str) -> None:
    # BaseLoader keeps GitHub's "on" key instead of treating it as a YAML 1.1 boolean.
    config = yaml.load((WORKFLOWS / f"{name}.yaml").read_text(), Loader=yaml.BaseLoader)
    events = config["on"]
    assert "pull_request" in events
    selection = events["pull_request"] or {}
    assert "branches" not in selection
    assert "branches-ignore" not in selection
    assert events["push"]["branches"] == ["main"]


def test_pull_requests_cannot_deploy_documentation() -> None:
    config = yaml.load((WORKFLOWS / "docs.yaml").read_text(), Loader=yaml.BaseLoader)
    not_pull_request = "github.event_name != 'pull_request'"
    assert config["jobs"]["deploy"]["if"] == not_pull_request
    upload = next(
        step
        for step in config["jobs"]["build"]["steps"]
        if step.get("name") == "Upload Pages artifact"
    )
    assert upload["if"] == not_pull_request


@pytest.mark.parametrize(
    "requested_version", [f"{sys.version_info.major}.{sys.version_info.minor}", "0.0"]
)
def test_ci_uses_and_verifies_matrix_python(requested_version: str) -> None:
    config = yaml.load((WORKFLOWS / "ci.yaml").read_text(), Loader=yaml.BaseLoader)
    job = config["jobs"]["test"]
    assert job["env"]["UV_PYTHON"] == "${{ matrix.python-version }}"
    cache = next(step for step in job["steps"] if step.get("name") == "Cache .venv")
    assert cache["with"]["key"].startswith("venv-matrix-v2-")
    verification = next(
        step for step in job["steps"] if step.get("name") == "Verify Python version"
    )
    assert verification["if"] == "env.SETUP == 'true'"
    command = shlex.split(verification["run"])
    assert command[:4] == ["uv", "run", "python", "-c"]
    assert len(command) == 5
    result = subprocess.run(
        [sys.executable, "-c", command[4]],
        env={**os.environ, "UV_PYTHON": requested_version},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert sys.version in result.stdout
    if requested_version == "0.0":
        assert result.returncode == 1
        assert "Expected Python 0.0, got" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
