"""Consistent image-size flags for image runs and standalone profiling."""

import argparse

import pytest
import yaml

from geo_eval_harness.cli import _parse_args, main
from geo_eval_harness.commands._config import parse_image_size


@pytest.mark.parametrize(
    ("value", "expected"), [("none", None), ("None", None), ("NONE", None), ("1", 1), ("224", 224)]
)
def test_image_size_parser(value: str, expected: int | None) -> None:
    assert parse_image_size(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "null", "bad", ""])
def test_invalid_image_size(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer or none"):
        parse_image_size(value)


@pytest.mark.parametrize("command", ["run", "profile"])
@pytest.mark.parametrize("value", ["0", "-1", "2.5", "null", "bad"])
def test_commands_report_the_same_image_size_error(
    command: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        _parse_args([command, "--image-size", value])
    assert error.value.code == 2
    assert "image-size must be a positive integer or none" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "profile"])
def test_omitted_image_size_stays_unset(command: str) -> None:
    assert not hasattr(_parse_args([command]), "image_size")


@pytest.mark.parametrize("command", ["run", "profile"])
@pytest.mark.parametrize(
    ("value", "expected"), [("none", None), ("None", None), ("NONE", None), ("17", 17)]
)
def test_dry_run_preserves_explicit_resize(
    command: str, value: str, expected: int | None, capsys: pytest.CaptureFixture[str]
) -> None:
    main([command, "--model", "rcf", "--dataset", "m-eurosat", "--image-size", value, "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out)["input"]["image_size"] == expected


def test_flops_requires_a_concrete_size() -> None:
    with pytest.raises(SystemExit) as error:
        _parse_args(["flops", "--image-size", "none"])
    assert error.value.code == 2
