"""Validate release metadata and packaged resources without importing ML dependencies."""

import argparse
import configparser
import logging
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def check_metadata(data: bytes, project: dict) -> None:
    """Reject mismatched identities and dependency URLs that PyPI does not accept."""
    metadata = BytesParser().parsebytes(data)
    for field, expected in (
        ("Name", project["name"]),
        ("Version", project["version"]),
        ("Requires-Python", project["requires-python"]),
        ("License-Expression", project["license"]),
    ):
        if metadata[field] != expected:
            raise ValueError(f"{field}: expected {expected!r}, got {metadata[field]!r}")
    for requirement in metadata.get_all("Requires-Dist", []):
        if "@" in requirement:
            raise ValueError(f"PyPI does not accept direct URL dependencies: {requirement}")
    if "LICENSE" not in metadata.get_all("License-File", []):
        raise ValueError("Distribution metadata must include LICENSE")


def check_distributions(directory: Path, *, tag: str | None = None) -> None:
    """Check both archives against the source package, including configuration and checksums."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if tag is not None and tag != f"v{project['version']}":
        raise ValueError(f"Tag {tag!r} does not match project version {project['version']!r}")
    stem = f"geo_eval_harness-{project['version']}"
    wheel_path = directory / f"{stem}-py3-none-any.whl"
    sdist_path = directory / f"{stem}.tar.gz"
    with zipfile.ZipFile(wheel_path) as wheel, tarfile.open(sdist_path) as sdist:
        info = f"{stem}.dist-info"
        check_metadata(wheel.read(f"{info}/METADATA"), project)
        metadata = sdist.extractfile(f"{stem}/PKG-INFO")
        if metadata is None:
            raise ValueError("Source distribution has no PKG-INFO")
        check_metadata(metadata.read(), project)
        if wheel.read(f"{info}/licenses/LICENSE") != (ROOT / "LICENSE").read_bytes():
            raise ValueError("Wheel license does not match LICENSE")
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(wheel.read(f"{info}/entry_points.txt").decode())
        if dict(entrypoints["console_scripts"]) != project["scripts"]:
            raise ValueError("Wheel console scripts do not match project.scripts")
        for path in sorted((ROOT / "src" / "geo_eval_harness").rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT / "src").as_posix()
            source = sdist.extractfile(f"{stem}/src/{relative}")
            if source is None or source.read() != path.read_bytes():
                raise ValueError(f"Source distribution is missing or changed: {relative}")
            if wheel.read(relative) != path.read_bytes():
                raise ValueError(f"Wheel is missing or changed: {relative}")
    logger.info("Validated %s and %s", wheel_path.name, sdist_path.name)


def main() -> None:
    """Validate built artifacts before installation or publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", help="Release tag, which must equal v<project.version>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_distributions(args.directory, tag=args.tag)


if __name__ == "__main__":
    main()
