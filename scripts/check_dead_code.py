"""Run vulture for dead-code detection, gating only on findings outside ``tests/``.

Pytest fixtures, ``unittest.mock`` attributes, and stdlib-signature overrides
(e.g. ``BaseHTTPRequestHandler.log_message``) make test files structurally
noisy for vulture, even though scanning them also resolves many real
references in ``src/`` (dataset/model registry classes, dataclass fields)
that would otherwise look unused. Tests stay in vulture's scanned paths for
that cross-reference signal, but findings located under ``tests/`` are
dropped before this gate decides pass or fail.

Usage::

    python scripts/check_dead_code.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATED_EXIT_CODES = (0, 3)


def main() -> int:
    """Run vulture from the repo root and report findings outside ``tests/``."""
    result = subprocess.run(
        [sys.executable, "-m", "vulture"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stderr.write(result.stderr)

    if result.returncode not in GATED_EXIT_CODES:
        sys.stdout.write(result.stdout)
        return result.returncode

    gated_lines = [line for line in result.stdout.splitlines() if not line.startswith("tests/")]
    for line in gated_lines:
        print(line)
    return 1 if gated_lines else 0


if __name__ == "__main__":
    sys.exit(main())
