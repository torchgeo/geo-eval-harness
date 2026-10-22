"""Unit tests for CLI entrypoints."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from torchgeo_bench import commands
from torchgeo_bench.cli import main as cli_main
from torchgeo_bench.config.flops import FlopsConfig
from torchgeo_bench.config.presets import ModelPreset
from torchgeo_bench.config.run import RunConfig


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
    command: str, output_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    selection = ["--dataset", "m-eurosat"] if command == "run" else []
    cli_main([command, "--model", "rcf", *selection, *output_args, "--dry-run"])
    schema = RunConfig if command == "run" else FlopsConfig
    config = schema.model_validate(yaml.safe_load(capsys.readouterr().out))
    assert config.output.file == "results/run=1.csv"


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
    command: str, flags: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    selection = ["--dataset", "m-eurosat"] if command == "run" else []
    cli_main([command, "--model", "rcf", *selection, "--dry-run", *flags])
    schema = RunConfig if command == "run" else FlopsConfig
    config = schema.model_validate(yaml.safe_load(capsys.readouterr().out))
    assert config.runtime.seed == 2
    assert config.runtime.device == "cpu"


@pytest.mark.parametrize("command", ["run", "flops", "profile", "coord"])
@pytest.mark.parametrize(
    "arguments",
    [
        ["model=rcf"],
        ["+model.features=8"],
        ["++model.pool=cls", "+model.gsd=1.0"],
        ["typo.key=1"],
        ["seed=1", "--device", "cpu", "seed=2"],
        ["seed=1", "--device", "cpu", "--", "seed=2"],
    ],
)
def test_old_override_grammar_reports_migration(
    command: str, arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main([command, *arguments])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "key=value and +key=value overrides have been retired" in message
    assert "--model rcf --dataset m-eurosat" in message
    assert "--config run.yaml" in message
    assert "Traceback" not in message


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--unknown=value", "--dry-run"],
        ["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run", "--unknown=value"],
        ["flops", "--model", "rcf", "--dry-run", "--unknown=value"],
    ],
)
def test_unrecognized_arguments_are_rejected(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(argv)
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_run_repeated_dataset_flags_preserve_order(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[RunConfig] = []
    monkeypatch.setattr(commands, "_run_runtime", SimpleNamespace(run=received.append))
    cli_main(["run", "-m", "rcf", "-d", "m-eurosat", "--dataset", "m-so2sat"])
    assert received[0].datasets == ["m-eurosat", "m-so2sat"]


def test_run_unknown_model_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["run", "--model", "not_a_model", "--dataset", "m-eurosat"])
    assert error.value.code == 2
    assert "not_a_model" in capsys.readouterr().err


def test_run_unknown_yaml_key_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\ntypo_key: 1\n")
    with pytest.raises(SystemExit) as error:
        cli_main(["run", "--config", str(path)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "typo_key" in message
    assert "Extra inputs are not permitted" in message
    assert "Traceback" not in message


def test_run_dry_run_prints_reusable_typed_config(capsys: pytest.CaptureFixture[str]) -> None:
    cli_main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run"])
    config = RunConfig.model_validate(yaml.safe_load(capsys.readouterr().out))
    assert config.model.name == "rcf"
    assert config.datasets == ["m-eurosat"]


def test_run_list_models(capsys) -> None:
    cli_main(["models"])
    out = capsys.readouterr().out
    assert "rcf" in out
    assert "torchgeo/scalemae_large_fmow" in out


def test_run_list_datasets(capsys) -> None:
    cli_main(["datasets"])
    out = capsys.readouterr().out
    assert "m-eurosat" in out
    assert "ucmerced" in out


def test_run_model_help(capsys) -> None:
    cli_main(["models", "rcf"])
    preset = ModelPreset.model_validate(yaml.safe_load(capsys.readouterr().out))
    assert preset.name == "rcf"
    assert preset.target == "torchgeo_bench.models.RCFBench"
    assert preset.track == "image"


def test_run_model_help_unknown_model_errors() -> None:
    with pytest.raises(SystemExit, match="unknown model 'not-a-real-model'"):
        cli_main(["models", "not-a-real-model"])


def test_run_model_help_rejects_path_traversal() -> None:
    with pytest.raises(SystemExit, match="unknown model"):
        cli_main(["models", "../flops_config"])


@pytest.mark.parametrize(
    ("dataset", "task"), [("m-eurosat", "classification"), ("burn_scars", "segmentation")]
)
def test_dataset_catalog_details(
    dataset: str, task: str, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_main(["datasets", dataset])
    assert yaml.safe_load(capsys.readouterr().out) == {"name": dataset, "task": task}


def test_download_invalid_target() -> None:
    with pytest.raises(SystemExit, match="Unknown dataset"):
        cli_main(["download", "bogus"])


@pytest.mark.parametrize(
    ("collection", "selection"),
    [
        ("geobench_v1", "unknown"),
        ("geobench_v1", "m-eurosat,unknown"),
        ("geobench_v2", "unknown"),
        ("geobench_v2", "caffe,unknown"),
    ],
)
def test_download_invalid_collection_selection(
    tmp_path: Path,
    collection: str,
    selection: str,
) -> None:
    output = tmp_path / "downloads"
    with pytest.raises(SystemExit, match="error: Unknown GeoBench"):
        cli_main(["download", collection, "--datasets", selection, "--output-dir", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("collection", ["geobench_v1", "geobench_v2"])
def test_download_collection_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
    collection: str,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected download failure")

    monkeypatch.setattr(f"torchgeo_bench.download.download_{collection}", fail)
    with pytest.raises(RuntimeError, match="unexpected download failure"):
        cli_main(["download", collection])


def test_download_geobench_v1(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def _fake_download(path, datasets=None) -> None:
        calls.append((str(path), datasets))

    monkeypatch.setattr("torchgeo_bench.download.download_geobench_v1", _fake_download)
    cli_main(["download", "geobench_v1"])
    assert calls == [("data", None)]


def test_download_geobench_v2_with_datasets(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def _fake_download(path, datasets=None) -> None:
        calls.append((str(path), datasets))

    monkeypatch.setattr("torchgeo_bench.download.download_geobench_v2", _fake_download)
    cli_main(["download", "geobench_v2", "--datasets", "burn_scars,benv2"])
    assert calls == [("data", ["burn_scars", "benv2"])]


@pytest.mark.parametrize(
    "names",
    [
        ["m-eurosat", "burn_scars"],
        ["m-eurosat", "eurosat"],
        ["eurosat", "resisc45"],
        ["ucmerced"],
        ["eurosat", "ucmerced"],
        ["m-eurosat", "burn_scars", "eurosat", "resisc45"],
    ],
)
def test_download_named_datasets(monkeypatch, names: list[str]) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_download(names, path) -> None:
        calls.append((names, str(path)))

    monkeypatch.setattr("torchgeo_bench.download.download_datasets", _fake_download)
    cli_main(["download", *names])
    assert calls == [(names, "data")]


@pytest.mark.parametrize("names", [["geobench_v1", "eurosat"], ["geobench_v1", "geobench_v2"]])
def test_download_rejects_mixed_collection_targets(names: list[str]) -> None:
    with pytest.raises(SystemExit, match="collection targets cannot be mixed"):
        cli_main(["download", *names])


@pytest.mark.parametrize("invalid", ["unknown", "not_a_dataset"])
def test_download_named_dataset_rejects_unknown_before_backend(monkeypatch, invalid: str) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "torchgeo_bench.hub.snapshot_download", lambda *args, **kwargs: calls.append(args)
    )
    with pytest.raises(SystemExit, match="Unknown dataset"):
        cli_main(["download", "m-eurosat", invalid])
    assert calls == []


def test_download_eurosat(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "torchgeo_bench.download.download_eurosat", lambda path: calls.append(str(path))
    )
    cli_main(["download", "eurosat"])
    assert calls == ["data"]


def test_download_rejects_empty_dataset_list() -> None:
    with pytest.raises(SystemExit, match="at least one dataset name"):
        cli_main(["download", "geobench_v1", "--datasets", ","])


def test_download_resisc45(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "torchgeo_bench.download.download_resisc45", lambda path: calls.append(str(path))
    )
    cli_main(["download", "resisc45"])
    assert calls == ["data"]


def test_download_ucmerced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    monkeypatch.setattr("torchgeo_bench.download.download_ucmerced", calls.append)
    cli_main(["download", "ucmerced", "--output-dir", str(tmp_path)])
    assert calls == [tmp_path]


def test_download_rejects_datasets_for_ucmerced() -> None:
    with pytest.raises(SystemExit, match="only supported for GeoBench"):
        cli_main(["download", "ucmerced", "--datasets", "ucmerced"])


def test_download_rejects_datasets_for_resisc45() -> None:
    with pytest.raises(SystemExit):
        cli_main(["download", "resisc45", "--datasets", "resisc45"])


def test_download_rejects_datasets_for_eurosat() -> None:
    with pytest.raises(SystemExit, match="only supported for GeoBench"):
        cli_main(["download", "eurosat", "--datasets", "m-eurosat"])


def test_flops_without_model_errors_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["flops"])
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


def test_unknown_model_is_reported_without_suggestions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(["run", "-m", "resnet50", "-d", "m-eurosat"])
    assert error.value.code == 2
    assert capsys.readouterr().err == "error: Unknown model config 'resnet50'.\n"


def test_constructor_options_are_explicit_yaml_kwargs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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


def test_model_names_are_posix_on_every_platform(capsys: pytest.CaptureFixture[str]) -> None:
    """Model names are CLI identifiers and must use forward slashes even on Windows."""
    cli_main(["models"])
    names = capsys.readouterr().out.splitlines()
    assert "torchgeo/scalemae_large_fmow" in names
    assert not any("\\" in n for n in names)
