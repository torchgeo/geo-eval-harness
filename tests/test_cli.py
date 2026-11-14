"""Unit tests for CLI entrypoints."""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from geo_eval_harness import commands
from geo_eval_harness.cli import _parse_args
from geo_eval_harness.cli import main as cli_main
from geo_eval_harness.commands._config import parse_image_size
from geo_eval_harness.commands.run_arguments import add_run_arguments
from geo_eval_harness.config import list_model_configs
from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.presets import ModelPreset
from geo_eval_harness.config.run import RunConfig


@pytest.mark.parametrize("command", ["run", "profile", "flops", "coord"])
def test_cli_dispatches_commands_directly(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[argparse.Namespace] = []
    monkeypatch.setattr(commands, command, received.append)
    cli_main([command])
    assert len(received) == 1
    assert received[0].command == command


def test_run_dispatches_typed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[RunConfig] = []
    monkeypatch.setattr(commands, "_run_runtime", SimpleNamespace(run=received.append))
    cli_main(
        ["run", "--model", "rcf", "--dataset", "m-eurosat", "--batch-size", "8", "--device", "cpu"]
    )
    assert len(received) == 1
    cfg = received[0]
    assert isinstance(cfg, RunConfig)
    assert cfg.model.name == "rcf"
    assert cfg.datasets == ["m-eurosat"]
    assert cfg.runtime.batch_size == 8
    assert cfg.runtime.device == "cpu"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["run", "-m", "rcf", "-d", "m-eurosat", "--device", "cpu"], "WARNING: problem\n"),
        (
            ["run", "-m", "rcf", "-d", "m-eurosat", "--device", "cpu", "--verbose"],
            "INFO: progress\nWARNING: problem\n",
        ),
        (
            ["coord", "--model", "sincos", "--dataset", "country"],
            "INFO: progress\nWARNING: problem\n",
        ),
        (["profile", "-m", "rcf", "-d", "m-eurosat"], "WARNING: problem\n"),
    ],
)
def test_commands_log_to_stderr_at_the_selected_level(
    argv: list[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def runtime(_config: object) -> None:
        logger = logging.getLogger("geo_eval_harness.main")
        logger.info("progress")
        logger.warning("problem")

    root = logging.getLogger()
    level = root.level
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(commands, "_run_runtime", SimpleNamespace(run=runtime))
    monkeypatch.setattr(commands, "_profile_runtime", SimpleNamespace(run=runtime))
    monkeypatch.setattr("geo_eval_harness.coordbench.run.run_coordbench", runtime)
    try:
        cli_main(argv)
    finally:
        root.setLevel(level)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == expected


@pytest.mark.parametrize("command", ["run", "flops"])
def test_flags_override_yaml_and_preserve_omitted_values(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = {
        "model": {"name": "rcf", "kwargs": {"features": 8}},
        "runtime": {"device": "cuda:0", "seed": 17, "verbose": True},
        "output": {"resume": True},
    }
    if command == "run":
        settings["datasets"] = ["m-eurosat"]
    path = tmp_path / "run=1.yaml"
    path.write_text(yaml.safe_dump(settings))
    cli_main(
        [
            command,
            "--config",
            str(path),
            "--device",
            "cpu",
            "--no-resume",
            "--no-verbose",
            "--dry-run",
        ]
    )
    schema = RunConfig if command == "run" else FlopsConfig
    config = schema.model_validate(yaml.safe_load(capsys.readouterr().out))
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 17
    assert not config.runtime.verbose
    assert config.model.kwargs == {"features": 8}
    assert not config.output.resume


class TestRunArguments:
    @pytest.fixture
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_run_arguments(parser)
        return parser

    @staticmethod
    def dry_run(capsys: pytest.CaptureFixture[str], command: str, *flags: str) -> dict:
        """Return the YAML a command prints for an rcf dry run with extra flags."""
        selection = [] if command == "flops" else ["--dataset", "m-eurosat"]
        cli_main([command, "--model", "rcf", *selection, *flags, "--dry-run"])
        return yaml.safe_load(capsys.readouterr().out)

    def test_omitted_flags_do_not_set_config_defaults(
        self, parser: argparse.ArgumentParser
    ) -> None:
        assert vars(parser.parse_args([])) == {}

    def test_explicit_values_preserve_types_and_dataset_order(
        self, parser: argparse.ArgumentParser
    ) -> None:
        args = parser.parse_args(
            [
                "--config",
                "run.yaml",
                "-m",
                "rcf",
                "-d",
                "m-eurosat",
                "--dataset",
                "burn_scars",
                "--image-size",
                "none",
                "--workers",
                "0",
                "--no-resume",
                "--methods",
                "linear",
                "--dry-run",
            ]
        )
        assert vars(args) == {
            "config": Path("run.yaml"),
            "model": "rcf",
            "datasets": ["m-eurosat", "burn_scars"],
            "image_size": None,
            "workers": 0,
            "resume": False,
            "methods": ["linear"],
            "dry_run": True,
        }

    @pytest.mark.parametrize("command", ["run", "flops"])
    @pytest.mark.parametrize(
        "output_args",
        [
            ["--output", "results/run=1.csv"],
            ["-o", "results/run=1.csv"],
            ["--output=results/run=1.csv"],
        ],
    )
    def test_output_flag_value_can_contain_equals(
        self, command: str, output_args: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.dry_run(capsys, command, *output_args)["output"]["file"] == "results/run=1.csv"

    @pytest.mark.parametrize("command", ["run", "flops"])
    @pytest.mark.parametrize(
        "flags",
        [
            ["--seed", "1", "--device", "cpu", "--seed", "2"],
            ["--seed=1", "--device=cpu", "--seed=2"],
            ["--seed", "1", "--device", "cpu", "--seed", "1", "--seed", "2"],
        ],
    )
    def test_repeated_flags_keep_their_order(
        self, command: str, flags: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.dry_run(capsys, command, *flags)["runtime"] == {"device": "cpu", "seed": 2}

    def test_run_repeated_dataset_flags_preserve_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: list[RunConfig] = []
        monkeypatch.setattr(commands, "_run_runtime", SimpleNamespace(run=received.append))
        cli_main(["run", "-m", "rcf", "-d", "m-eurosat", "--dataset", "m-so2sat"])
        assert received[0].datasets == ["m-eurosat", "m-so2sat"]

    @pytest.mark.parametrize(
        "argv",
        [
            ["run", "--unknown=value", "--dry-run"],
            ["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run", "--unknown=value"],
            ["flops", "--model", "rcf", "--dry-run", "--unknown=value"],
        ],
    )
    def test_unrecognized_arguments_are_rejected(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            cli_main(argv)
        assert exc.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_run_dry_run_prints_reusable_typed_config(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = RunConfig.model_validate(self.dry_run(capsys, "run"))
        assert config.model.name == "rcf"
        assert config.datasets == ["m-eurosat"]

    def test_constructor_options_are_explicit_yaml_kwargs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "custom.yaml"
        path.write_text(
            "model:\n  name: custom\n  target: not_installed.Model\n"
            "  kwargs: {pool: cls, gsd: 1.0}\ndatasets: [m-eurosat]\n"
        )
        cli_main(["run", "--config", str(path), "--dry-run"])
        config = RunConfig.model_validate(yaml.safe_load(capsys.readouterr().out))
        assert config.model.target == "not_installed.Model"
        assert config.model.kwargs == {"pool": "cls", "gsd": 1.0}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("none", None), ("None", None), ("NONE", None), ("1", 1), ("224", 224)],
    )
    def test_image_size_parser(self, value: str, expected: int | None) -> None:
        assert parse_image_size(value) == expected

    @pytest.mark.parametrize("value", ["0", "-1", "2.5", "null", "bad", ""])
    def test_invalid_image_size(self, value: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer or none"):
            parse_image_size(value)

    @pytest.mark.parametrize("command", ["run", "profile"])
    def test_commands_report_the_same_image_size_error(
        self, command: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as error:
            _parse_args([command, "--image-size", "2.5"])
        assert error.value.code == 2
        assert "image-size must be a positive integer or none" in capsys.readouterr().err

    @pytest.mark.parametrize("command", ["run", "profile"])
    def test_omitted_image_size_stays_unset(self, command: str) -> None:
        assert not hasattr(_parse_args([command]), "image_size")

    @pytest.mark.parametrize("command", ["run", "profile"])
    @pytest.mark.parametrize(("value", "expected"), [("none", None), ("17", 17)])
    def test_dry_run_preserves_explicit_resize(
        self,
        command: str,
        value: str,
        expected: int | None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = self.dry_run(capsys, command, "--image-size", value)
        assert config["input"]["image_size"] == expected

    def test_flops_requires_a_concrete_size(self) -> None:
        with pytest.raises(SystemExit) as error:
            _parse_args(["flops", "--image-size", "none"])
        assert error.value.code == 2


class TestCatalog:
    def test_model_catalog_matches_packaged_presets(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli_main(["models"])
        names = capsys.readouterr().out.splitlines()
        assert names == list_model_configs()
        # Model names are CLI identifiers and must use forward slashes even on Windows.
        assert "torchgeo/scalemae_large_fmow" in names
        assert not any("\\" in name for name in names)

    def test_run_list_datasets(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli_main(["datasets"])
        out = capsys.readouterr().out
        assert "m-eurosat" in out
        assert "ucmerced" in out

    def test_run_model_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli_main(["models", "rcf"])
        preset = ModelPreset.model_validate(yaml.safe_load(capsys.readouterr().out))
        assert preset.name == "rcf"
        assert preset.target == "geo_eval_harness.models.RCFBench"
        assert preset.track == "image"

    @pytest.mark.parametrize(
        ("dataset", "task"), [("m-eurosat", "classification"), ("burn_scars", "segmentation")]
    )
    def test_dataset_catalog_details(
        self, dataset: str, task: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli_main(["datasets", dataset])
        assert yaml.safe_load(capsys.readouterr().out) == {"name": dataset, "task": task}

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["models", "not-a-real-model"], "unknown model 'not-a-real-model'"),
            (["models", "../flops_config"], "unknown model"),
            (["datasets", "unknown"], "unknown dataset 'unknown'"),
        ],
    )
    def test_unknown_name_errors(self, argv: list[str], message: str) -> None:
        with pytest.raises(SystemExit, match=message):
            cli_main(argv)


class TestDownloadCommand:
    @pytest.fixture
    def backend(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path, list[str] | None]]:
        """Record downloader calls in place of the real downloads."""
        calls: list[tuple[str, Path, list[str] | None]] = []
        for name in ("geobench_v1", "geobench_v2", "eurosat", "resisc45", "ucmerced"):

            def record(path: Path, datasets: list[str] | None = None, name: str = name) -> None:
                calls.append((name, path, datasets))

            monkeypatch.setattr(f"geo_eval_harness.download.download_{name}", record)
        return calls

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["geobench_v1"], [("geobench_v1", None)]),
            (
                ["geobench_v2", "--datasets", "burn_scars,benv2"],
                [("geobench_v2", ["burn_scars", "benv2"])],
            ),
            (["eurosat"], [("eurosat", None)]),
            (["resisc45"], [("resisc45", None)]),
            (["ucmerced"], [("ucmerced", None)]),
            (
                ["m-eurosat", "burn_scars", "eurosat", "resisc45"],
                [
                    ("geobench_v1", ["m-eurosat"]),
                    ("geobench_v2", ["burn_scars"]),
                    ("eurosat", None),
                    ("resisc45", None),
                ],
            ),
        ],
    )
    def test_targets_reach_their_downloaders(
        self,
        argv: list[str],
        expected: list[tuple[str, list[str] | None]],
        backend: list[tuple[str, Path, list[str] | None]],
    ) -> None:
        cli_main(["download", *argv])
        assert backend == [(name, Path("data"), datasets) for name, datasets in expected]

    def test_output_dir_reaches_the_downloader(
        self, tmp_path: Path, backend: list[tuple[str, Path, list[str] | None]]
    ) -> None:
        cli_main(["download", "ucmerced", "--output-dir", str(tmp_path)])
        assert backend == [("ucmerced", tmp_path, None)]

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["bogus"], "error: Unknown dataset"),
            (["geobench_v1", "eurosat"], "collection targets cannot be mixed"),
            (["geobench_v1", "geobench_v2"], "collection targets cannot be mixed"),
            (["eurosat", "--datasets", "m-eurosat"], "only supported for GeoBench"),
            (["geobench_v1", "--datasets", ","], "at least one dataset name"),
            (["geobench_v1", "--datasets", "m-eurosat,unknown"], "error: Unknown GeoBench"),
            (["geobench_v2", "--datasets", "caffe,unknown"], "error: Unknown GeoBench"),
        ],
    )
    def test_invalid_request_exits_before_writing(
        self, argv: list[str], message: str, tmp_path: Path
    ) -> None:
        output = tmp_path / "downloads"
        with pytest.raises(SystemExit, match=message):
            cli_main(["download", *argv, "--output-dir", str(output)])
        assert not output.exists()

    @pytest.mark.parametrize("collection", ["geobench_v1", "geobench_v2"])
    def test_collection_propagates_unexpected_errors(
        self, monkeypatch: pytest.MonkeyPatch, collection: str
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("unexpected download failure")

        monkeypatch.setattr(f"geo_eval_harness.download.download_{collection}", fail)
        with pytest.raises(RuntimeError, match="unexpected download failure"):
            cli_main(["download", collection])


@pytest.mark.parametrize("command", ["flops", "profile"])
def test_command_without_model_errors_cleanly(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main([command])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "model" in message
    assert "Field required" in message


def test_flops_dispatches_typed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[FlopsConfig] = []
    monkeypatch.setattr(commands, "run_flops", received.append)
    cli_main(["flops", "-m", "rcf", "--device", "cpu"])
    assert isinstance(received[0], FlopsConfig)
    assert received[0].classification.num_classes == 10
    assert received[0].runtime.device == "cpu"
