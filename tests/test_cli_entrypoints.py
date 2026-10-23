"""Subprocess coverage of the canonical image CLI entry points."""

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml


def run_entrypoint(
    module: str | None, arguments: list[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", module] if module is not None else ["geo-eval-harness"]
    return subprocess.run(
        [*command, *arguments],
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("module", [None, "geo_eval_harness", "geo_eval_harness.cli"])
def test_version_prints_installed_version_without_subcommand(
    module: str | None, tmp_path: Path
) -> None:
    completed = run_entrypoint(module, ["--version"], tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"geo-eval-harness {version('geo-eval-harness')}\n"
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("arguments", "status", "expected"),
    [
        (["--help"], 0, "Run image benchmarks"),
        (["run", "--help"], 0, "--config"),
        (["models", "rcf"], 0, "target:"),
        (["datasets", "m-eurosat"], 0, "task: classification"),
        ([], 2, "required: command"),
        (["run", "--unknown-option"], 2, "unrecognized arguments: --unknown-option"),
    ],
)
def test_package_module_matches_console_script(
    arguments: list[str], status: int, expected: str, tmp_path: Path
) -> None:
    package = run_entrypoint("geo_eval_harness", arguments, tmp_path)
    canonical = run_entrypoint("geo_eval_harness.cli", arguments, tmp_path)
    assert package.returncode == canonical.returncode == status
    assert package.stdout == canonical.stdout
    assert package.stderr == canonical.stderr
    assert expected in package.stdout + package.stderr


@pytest.mark.parametrize("use_config", [False, True])
def test_package_module_dry_run_matches_console_script(tmp_path: Path, *, use_config: bool) -> None:
    if use_config:
        path = tmp_path / "run.yaml"
        path.write_text(
            "model: {name: rcf}\ndatasets: [m-eurosat]\n"
            "runtime: {device: cpu, seed: 8}\noutput: {resume: true}\n",
            encoding="utf-8",
        )
        arguments = ["run", "--config", str(path)]
    else:
        arguments = ["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu"]
    arguments.extend(["--seed", "3", "--no-resume", "--dry-run"])
    package = run_entrypoint("geo_eval_harness", arguments, tmp_path)
    canonical = run_entrypoint("geo_eval_harness.cli", arguments, tmp_path)
    assert package.returncode == canonical.returncode == 0, package.stderr + canonical.stderr
    assert package.stdout == canonical.stdout
    assert package.stderr == canonical.stderr == ""
    config = yaml.safe_load(package.stdout)
    assert config["model"]["name"] == "rcf"
    assert config["datasets"] == ["m-eurosat"]
    assert config["runtime"] == {"device": "cpu", "seed": 3}
    assert config["output"]["resume"] is False


@pytest.mark.parametrize("module", [None, "geo_eval_harness", "geo_eval_harness.cli"])
def test_profile_rejects_repeated_dataset_flags(module: str | None, tmp_path: Path) -> None:
    completed = run_entrypoint(
        module,
        ["profile", "--model", "rcf", "-d", "m-eurosat", "--dataset=so2sat", "--dry-run"],
        tmp_path,
    )
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert "usage: geo-eval-harness profile" in completed.stderr
    assert "error:" in completed.stderr
    assert "may only be specified once" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("module", [None, "geo_eval_harness", "geo_eval_harness.cli"])
@pytest.mark.parametrize("use_config", [False, True])
def test_profile_accepts_one_dataset_flag(
    module: str | None, tmp_path: Path, *, use_config: bool
) -> None:
    if use_config:
        path = tmp_path / "profile.yaml"
        path.write_text("model: {name: rcf}\ndataset: m-eurosat\n", encoding="utf-8")
        arguments = ["profile", "--config", str(path)]
    else:
        arguments = ["profile", "--model", "rcf"]
    completed = run_entrypoint(module, [*arguments, "--dataset=so2sat", "--dry-run"], tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    config = yaml.safe_load(completed.stdout)
    assert config["model"]["name"] == "rcf"
    assert config["dataset"] == "so2sat"


@pytest.mark.parametrize("module", [None, "geo_eval_harness", "geo_eval_harness.cli"])
def test_run_accepts_repeated_dataset_flags(module: str | None, tmp_path: Path) -> None:
    completed = run_entrypoint(
        module,
        ["run", "-m", "rcf", "-d", "m-eurosat", "--dataset=so2sat", "--dry-run"],
        tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert yaml.safe_load(completed.stdout)["datasets"] == ["m-eurosat", "so2sat"]


@pytest.mark.parametrize("module", [None, "geo_eval_harness", "geo_eval_harness.cli"])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    ("flags", "diagnostic"),
    [
        (["--bands", "B99"], "MEurosat: unknown band 'B99'; available:"),
        (["--normalization", "model"], "'rcf' does not support --normalization model"),
    ],
)
def test_run_rejects_semantic_input_errors_before_execution(
    module: str | None,
    flags: list[str],
    diagnostic: str,
    tmp_path: Path,
    *,
    dry_run: bool,
) -> None:
    arguments = ["run", "--model", "rcf", "--dataset", "m-eurosat", *flags]
    if dry_run:
        arguments.append("--dry-run")
    completed = run_entrypoint(module, arguments, tmp_path)
    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("error: ")
    assert diagnostic in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()
