"""Exact-commit Hugging Face downloads verified against packaged SHA-256 manifests."""

import hashlib
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from pathlib import Path

from huggingface_hub import snapshot_download


@cache
def load_checksums(manifest: str) -> dict[str, str]:
    """Read ``checksums/<manifest>`` in ``sha256sum`` format into ``{relative path: digest}``."""
    text = files("torchgeo_bench").joinpath("checksums", manifest).read_text()
    return {name: digest for digest, name in (line.split() for line in text.splitlines())}


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
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
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
