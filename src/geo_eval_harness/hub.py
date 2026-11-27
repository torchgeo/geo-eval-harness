"""Exact-commit Hugging Face downloads verified against packaged SHA-256 manifests."""

import hashlib
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def parse_checksums(text: str) -> dict[str, str]:
    """Parse ``sha256sum`` output into ``{relative path: digest}``."""
    return {name: digest for digest, name in (line.split() for line in text.splitlines())}


@cache
def load_checksums(manifest: str) -> dict[str, str]:
    """Read ``checksums/<manifest>`` in ``sha256sum`` format into ``{relative path: digest}``."""
    return parse_checksums(files("geo_eval_harness").joinpath("checksums", manifest).read_text())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checksums(
    root: Path, checksums: Mapping[str, str], *, complete_glob: str | None = None
) -> None:
    """Require each listed file under ``root`` to match its SHA-256 digest.

    Args:
        root: Directory that the manifest paths are relative to.
        checksums: Expected digests keyed by relative path.
        complete_glob: If set, files matching this pattern must all be listed, so that stale
            files from another revision are rejected rather than read.
    """
    if complete_glob is not None:
        listed = {path.relative_to(root).as_posix() for path in root.glob(complete_glob)}
        if extra := listed - checksums.keys():
            raise ValueError(f"Unexpected files in {root}: {sorted(extra)}")
    for name, expected in checksums.items():
        path = root / name
        if sha256_file(path) != expected:
            raise ValueError(f"SHA-256 checksum mismatch: {path}. Remove it and download again.")


def fetch_pinned(  # noqa: PLR0913 - snapshot_download options stay explicit
    repo_id: str,
    revision: str,
    local_dir: Path,
    checksums: Mapping[str, str],
    *,
    complete_glob: str | None = None,
    allow_patterns: list[str] | None = None,
    cache_dir: Path | None = None,
) -> None:
    """Download a dataset repository at an exact commit, then verify the listed files.

    Args:
        repo_id: Hugging Face dataset repository.
        revision: Full commit hash, never a branch or tag.
        local_dir: Download destination; manifest paths are relative to it.
        checksums: Expected digests keyed by path relative to ``local_dir``.
        complete_glob: Pattern of files that must all be listed; see :func:`verify_checksums`.
        allow_patterns: Optional repository paths to download.
        cache_dir: Optional Hugging Face download cache.
    """
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
    )
    verify_checksums(local_dir, checksums, complete_glob=complete_glob)


def fetch_pinned_file(
    repo_id: str, filename: str, revision: str, sha256: str, *, local_path: str | Path | None = None
) -> Path:
    """Return one model repository file at an exact commit after verifying its SHA-256 digest.

    Args:
        repo_id: Hugging Face model repository.
        filename: Path of the file in the repository.
        revision: Full commit hash, never a branch or tag.
        sha256: Expected digest of the file.
        local_path: An existing copy to verify and return instead of downloading.
    """
    path = (
        Path(hf_hub_download(repo_id, filename, revision=revision))
        if local_path is None
        else Path(local_path).expanduser()
    )
    verify_checksums(path.parent, {path.name: sha256})
    return path
