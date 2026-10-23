"""Shared flags-over-YAML loading for command handlers."""

import argparse
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, NoReturn

import yaml

from ..config.presets import merge_settings
from ..config.schema import _UniqueKeyLoader, load_yaml

EXPECTED_CONFIG_ERRORS = (OSError, ValueError, yaml.YAMLError)


def parse_image_size(value: str) -> int | None:
    """Parse a positive image size or case-insensitive ``none`` to disable resizing."""
    if value.lower() == "none":
        return None
    try:
        size = int(value)
    except ValueError as error:  # allow-except: report invalid sizes as argparse input errors
        raise argparse.ArgumentTypeError("image-size must be a positive integer or none") from error
    if size <= 0:
        raise argparse.ArgumentTypeError("image-size must be a positive integer or none")
    return size


@dataclass(frozen=True)
class FlagOverride:
    """Map one explicitly supplied argparse value into a configuration path."""

    name: str
    path: tuple[str, ...]
    transform: Callable[[Any], Any] | None = None
    replace_roots: tuple[str, ...] = field(default_factory=tuple)


OUTPUT_FLAG_OVERRIDES = (
    FlagOverride("output", ("output", "file")),
    FlagOverride("output_dir", ("output", "directory")),
)


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Register common directory and explicit-file output flags."""
    parser.add_argument(
        "-o", "--output", default=argparse.SUPPRESS, help="Output file (overrides --output-dir)"
    )
    parser.add_argument(
        "--output-dir", default=argparse.SUPPRESS, help="Root directory for all command outputs"
    )


def comma_separated_bands(value: Any) -> Any:
    """Convert explicit comma-separated bands while leaving named selections alone."""
    if isinstance(value, str) and value not in {"rgb", "all"}:
        return [band.strip() for band in value.split(",")]
    return value


def yaml_mapping(value: str) -> dict[str, Any]:
    """Parse an inline YAML mapping from a flag value."""
    parsed = yaml.load(value, Loader=_UniqueKeyLoader)
    if not isinstance(parsed, dict):
        raise ValueError("value must be a YAML mapping")  # noqa: TRY004 - preserve the CLI configuration-error contract
    return parsed


def set_path(values: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a nested path in a plain dictionary."""
    section = values
    for name in path[:-1]:
        section = section.setdefault(name, {})
    section[path[-1]] = value


def flag_overrides(
    args: argparse.Namespace, fields: Iterable[FlagOverride]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Collect only argparse values that were explicitly supplied."""
    values: dict[str, Any] = {}
    replace_roots: list[str] = []
    for field_config in fields:
        if not hasattr(args, field_config.name):
            continue
        value = getattr(args, field_config.name)
        if field_config.transform is not None:
            value = field_config.transform(value)
        set_path(values, field_config.path, value)
        replace_roots.extend(field_config.replace_roots)
    return values, tuple(dict.fromkeys(replace_roots))


def _require_mapping(value: Any, path: tuple[str, ...]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{'.'.join(path)} must be a YAML mapping")  # noqa: TRY004 - preserve the CLI configuration-error contract


def _check_override_parents(
    base: dict[str, Any], overrides: dict[str, Any], path: tuple[str, ...] = ()
) -> None:
    for key, value in overrides.items():
        if not isinstance(value, dict):
            continue
        current = (*path, key)
        if key not in base:
            existing = {}
        else:
            existing = base[key]
            _require_mapping(existing, current)
        _check_override_parents(existing, value, current)


def apply_flag_overrides(
    base: dict[str, Any], overrides: dict[str, Any], *, replace_roots: Iterable[str] = ()
) -> dict[str, Any]:
    """Apply explicit flags after checking YAML sections are mergeable mappings."""
    values = dict(base)
    for root in replace_roots:
        if root in values:
            _require_mapping(values[root], (root,))
        values[root] = {}
    _check_override_parents(values, overrides)
    return merge_settings(values, overrides)


def load_from_flags[ConfigT](
    args: argparse.Namespace,
    fields: Iterable[FlagOverride],
    validate: Callable[[dict[str, Any]], ConfigT],
) -> ConfigT:
    """Load YAML, apply explicit flags, and validate one command config."""
    path = getattr(args, "config", None)
    base = load_yaml(path) if path is not None else {}
    overrides, replace_roots = flag_overrides(args, fields)
    return validate(apply_flag_overrides(base, overrides, replace_roots=replace_roots))


def exit_config_error(error: BaseException, *, path: object | None = None) -> NoReturn:
    """Report an expected user configuration error with argparse's status code."""
    message = str(error)
    if path is not None and str(path) not in message:
        message = f"{path}: {message}"
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2) from error


def load_config_or_exit[ConfigT](
    args: argparse.Namespace, loader: Callable[[argparse.Namespace], ConfigT]
) -> ConfigT:
    """Run a config loader and convert expected input errors into exit status 2."""
    try:
        return loader(args)
    except EXPECTED_CONFIG_ERRORS as error:  # allow-except: CLI configuration errors
        exit_config_error(error, path=getattr(args, "config", None))
