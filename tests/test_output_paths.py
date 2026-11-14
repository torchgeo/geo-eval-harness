"""Consistent nonblank output paths across image, FLOPs, and coordinate settings."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geo_eval_harness.cli import main
from geo_eval_harness.config.flops import FlopsConfig, FlopsOutputConfig
from geo_eval_harness.config.profile import ProfileConfig, ProfileOutputConfig
from geo_eval_harness.config.run import OutputConfig, RunConfig
from geo_eval_harness.config.schema import StrictModel, resolve_output_path
from geo_eval_harness.coordbench.config import CoordConfig, CoordOutputConfig


@pytest.mark.parametrize(
    ("config_class", "field", "default"),
    [
        (OutputConfig, "directory", "results"),
        (OutputConfig, "file", None),
        (FlopsOutputConfig, "directory", "results"),
        (FlopsOutputConfig, "file", None),
        (CoordOutputConfig, "directory", "results"),
        (CoordOutputConfig, "file", None),
        (ProfileOutputConfig, "directory", None),
        (ProfileOutputConfig, "file", None),
    ],
)
class TestOutputPaths:
    def test_defaults(
        self, config_class: type[StrictModel], field: str, default: str | None
    ) -> None:
        assert config_class().model_dump()[field] == default

    @pytest.mark.parametrize("value", ["", " ", "\t\n", 1, False, 1.5])
    def test_blank_or_non_string_path_is_rejected(
        self, config_class: type[StrictModel], field: str, default: str | None, value: object
    ) -> None:
        with pytest.raises(ValidationError, match=field) as error:
            config_class.model_validate({field: value})
        assert error.value.errors()[0]["loc"] == (field,)

    @pytest.mark.parametrize(
        "value", ["results.csv", " path with spaces.csv ", "missing/tree/out.csv"]
    )
    def test_valid_path_is_not_rewritten(
        self, config_class: type[StrictModel], field: str, default: str | None, value: str
    ) -> None:
        assert config_class.model_validate({field: value}).model_dump()[field] == value

    def test_only_optional_paths_accept_null(
        self, config_class: type[StrictModel], field: str, default: str | None
    ) -> None:
        if default is None:
            assert config_class.model_validate({field: None}).model_dump()[field] is None
        else:
            with pytest.raises(ValidationError, match=field):
                config_class.model_validate({field: None})


@pytest.mark.parametrize("yaml_output", [False, True])
@pytest.mark.parametrize(
    "flags",
    [
        ["--output-dir"],
        ["--output"],
        ["--output-dir", "--output"],
        ["--output", "--output-dir"],
    ],
)
@pytest.mark.parametrize(
    ("command", "config_class", "selection"),
    [
        ("run", RunConfig, {"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}),
        ("flops", FlopsConfig, {"model": {"name": "rcf"}}),
        ("coord", CoordConfig, {"model": {"name": "sincos"}}),
        ("profile", ProfileConfig, {"model": {"name": "rcf"}, "dataset": "m-eurosat"}),
    ],
)
def test_output_flags_preserve_other_settings_and_round_trip(  # noqa: PLR0913 - parameterized CLI matrix
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    command: str,
    config_class: type[RunConfig | FlopsConfig | CoordConfig | ProfileConfig],
    selection: dict,
    *,
    yaml_output: bool,
) -> None:
    values = dict(selection)
    values["output"] = {"directory": "yaml/root", "file": "yaml/file"} if yaml_output else {}
    expected = config_class.model_validate(values).output
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    arguments = []
    for flag in flags:
        arguments.extend([flag, "custom measurements"])
        field = "directory" if flag == "--output-dir" else "file"
        expected = expected.model_copy(update={field: "custom measurements"})
    main([command, "--config", str(config_path), *arguments, "--dry-run"])
    output = capsys.readouterr().out
    assert config_class.model_validate(yaml.safe_load(output)).output == expected

    config_path.write_text(output, encoding="utf-8")
    main([command, "--config", str(config_path), "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out) == yaml.safe_load(output)


def test_relative_output_file_is_not_under_output_directory() -> None:
    assert resolve_output_path("scratch", "chosen.json", "profile.json") == "chosen.json"


@pytest.mark.parametrize("flag", ["--results-dir", "--profile-dir", "--intrinsic-dim-dir"])
def test_removed_directory_flags_are_rejected(flag: str) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", flag, "scratch", "--dry-run"])
    assert error.value.code == 2


@pytest.mark.parametrize("field", ["profile_directory", "intrinsic_dim_directory"])
def test_separate_yaml_directories_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        OutputConfig.model_validate({field: "scratch"})
