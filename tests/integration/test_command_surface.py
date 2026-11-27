"""Entry-point contracts: installed launchers match ``cli.main`` and reject bad input cleanly."""

from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

from geo_eval_harness.cli import main
from tests.support.cli import cli_output, run_module_cli, run_public_cli

pytestmark = pytest.mark.integration

DRY_RUN = ("run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu", "--dry-run")


class TestEntryPoints:
    @pytest.mark.parametrize("entry_point", ["console", "geo_eval_harness", "geo_eval_harness.cli"])
    def test_version_and_dry_run_agree(
        self, entry_point: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(list(DRY_RUN))
        expected = {
            ("--version",): f"geo-eval-harness {version('geo-eval-harness')}\n",
            DRY_RUN: capsys.readouterr().out,
        }
        for arguments, stdout in expected.items():
            if entry_point == "console":
                result = run_public_cli(*arguments, cwd=tmp_path)
            else:
                result = run_module_cli(entry_point, *arguments, cwd=tmp_path)
            assert result.returncode == 0, cli_output(result)
            assert result.stdout == stdout, cli_output(result)
            assert result.stderr == "", cli_output(result)
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "results").exists()

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (("--dry-run", "unexpected=value"), "key=value and +key=value overrides"),
            (("--bands", "B99"), "error: MEurosat: unknown band 'B99'; available:"),
        ],
        ids=["retired-override", "unknown-band"],
    )
    def test_invalid_run_fails_without_traceback_or_side_effects(
        self, tmp_path: Path, arguments: tuple[str, ...], message: str
    ) -> None:
        result = run_public_cli(
            "run", "--model", "rcf", "--dataset", "m-eurosat", *arguments, cwd=tmp_path
        )
        assert result.returncode == 2, cli_output(result)
        assert result.stdout == "", cli_output(result)
        assert message in result.stderr, cli_output(result)
        assert "Traceback" not in result.stderr, cli_output(result)
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "results").exists()


class TestMain:
    @pytest.mark.parametrize(
        "command",
        [None, "run", "models", "datasets", "download", "profile", "flops", "coord", "embeddings"],
    )
    def test_help(self, command: str | None, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as error:
            main(["--help"] if command is None else [command, "--help"])
        assert error.value.code == 0
        captured = capsys.readouterr()
        assert captured.out.startswith("usage: geo-eval-harness")
        assert captured.err == ""

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ([], "the following arguments are required: command"),
            (
                ["run", "-m", "rcf", "-d", "m-eurosat", "--normalization", "model", "--dry-run"],
                "error: 'rcf' does not support --normalization model",
            ),
            (
                ["profile", "-m", "rcf", "-d", "m-eurosat", "--dataset=so2sat", "--dry-run"],
                "--dataset may only be specified once",
            ),
        ],
        ids=["no-command", "unsupported-normalization", "repeated-profile-dataset"],
    )
    def test_invalid_input_exits_with_message(
        self, arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert message in captured.err

    def test_profile_dataset_flag_overrides_yaml(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "profile.yaml"
        path.write_text("model: {name: rcf}\ndataset: m-eurosat\n", encoding="utf-8")
        main(["profile", "--config", str(path), "--dataset=so2sat", "--dry-run"])
        assert yaml.safe_load(capsys.readouterr().out)["dataset"] == "so2sat"
