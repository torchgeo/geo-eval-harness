"""Subprocess harnesses for the console and both canonical module entry points."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(
    command: list[str], *, cwd: Path, timeout: int, offline: bool
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[variable] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    return subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False
    )


def run_cli(
    *arguments: str, cwd: Path, timeout: int = 120, offline: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run the canonical ``python -m geo_eval_harness`` interface."""
    return run_module_cli("geo_eval_harness", *arguments, cwd=cwd, timeout=timeout, offline=offline)


def run_module_cli(
    module: str, *arguments: str, cwd: Path, timeout: int = 120, offline: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run an explicitly selected module without confusing it with the console script."""
    return _run(
        [sys.executable, "-m", module, *arguments], cwd=cwd, timeout=timeout, offline=offline
    )


def run_public_cli(
    *arguments: str, cwd: Path, timeout: int = 120, offline: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run the installed console script, including its installed entry-point mapping."""
    executable = shutil.which("geo-eval-harness", path=str(Path(sys.executable).parent))
    if executable is None:
        raise FileNotFoundError("Install geo-eval-harness in the pytest environment first.")
    return _run([executable, *arguments], cwd=cwd, timeout=timeout, offline=offline)


def cli_output(result: subprocess.CompletedProcess[str]) -> str:
    """Describe a failed invocation without dropping either captured output stream."""
    return (
        f"Command: {result.args!r}\nExit code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
