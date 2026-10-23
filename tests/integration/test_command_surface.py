"""Process-level entry-point contracts, without importing or faking CLI dispatch."""

from pathlib import Path

import pytest
import yaml

from tests.support.cli import cli_output, run_module_cli, run_public_cli

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("module", ["console", "geo_eval_harness", "geo_eval_harness.cli"])
def test_every_entrypoint_and_subcommand_has_help(tmp_path: Path, module: str) -> None:
    commands = ("run", "models", "datasets", "download", "profile", "flops", "coord")
    for arguments in (("--help",), *((command, "--help") for command in commands)):
        if module == "console":
            result = run_public_cli(*arguments, cwd=tmp_path)
        else:
            result = run_module_cli(module, *arguments, cwd=tmp_path)
        assert result.returncode == 0, cli_output(result)
        assert "usage:" in result.stdout, cli_output(result)
        assert not result.stderr, cli_output(result)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (("models",), "timm/resnet18"),
        (("models", "rcf"), "target: geo_eval_harness.models.RCFBench"),
        (("datasets",), "m-eurosat"),
        (("datasets", "caffe"), "task: segmentation"),
        (("run", "--config-help"), "schema_version"),
    ],
)
def test_public_catalogs_and_schema(
    tmp_path: Path, arguments: tuple[str, ...], expected: str
) -> None:
    result = run_public_cli(*arguments, cwd=tmp_path)
    assert result.returncode == 0, cli_output(result)
    assert expected in result.stdout, cli_output(result)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("arguments", "exit_code", "message"),
    [
        ((), 2, "required"),
        (("run", "--model", "rcf"), 2, "datasets"),
        (("models", "missing-preset"), 1, "unknown model"),
        (("download", "missing-dataset"), 1, "Unknown dataset"),
        (("profile", "--model", "rcf"), 2, "dataset"),
    ],
)
def test_public_validation_fails_without_side_effects(
    tmp_path: Path, arguments: tuple[str, ...], exit_code: int, message: str
) -> None:
    result = run_public_cli(*arguments, cwd=tmp_path)
    assert result.returncode == exit_code, cli_output(result)
    assert message in result.stderr, cli_output(result)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


def test_public_dry_run_is_reusable_yaml(tmp_path: Path) -> None:
    result = run_public_cli(
        "run",
        "--model",
        "rcf",
        "--dataset",
        "m-eurosat",
        "--device",
        "cpu",
        "--image-size",
        "16",
        "--methods",
        "knn",
        "--dry-run",
        cwd=tmp_path,
    )
    assert result.returncode == 0, cli_output(result)
    config = yaml.safe_load(result.stdout)
    assert config["model"] == {"name": "rcf"}
    assert config["datasets"] == ["m-eurosat"]
    assert config["classification"]["methods"] == ["knn"]
    assert config["input"]["image_size"] == 16
    path = tmp_path / "config.yaml"
    path.write_text(result.stdout)
    replay = run_public_cli("run", "--config", str(path), "--dry-run", cwd=tmp_path)
    assert replay.returncode == 0, cli_output(replay)
    assert yaml.safe_load(replay.stdout) == config
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("module", ["console", "geo_eval_harness", "geo_eval_harness.cli"])
@pytest.mark.parametrize("command", ["run", "flops", "profile", "coord"])
@pytest.mark.parametrize("override", ["model=rcf", "+model.features=8", "++model.pool=cls"])
def test_all_entrypoints_reject_retired_overrides(
    tmp_path: Path, module: str, command: str, override: str
) -> None:
    if module == "console":
        result = run_public_cli(command, override, cwd=tmp_path)
    else:
        result = run_module_cli(module, command, override, cwd=tmp_path)
    assert result.returncode == 2, cli_output(result)
    assert "key=value and +key=value overrides have been retired" in result.stderr
    assert "--config run.yaml" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()
