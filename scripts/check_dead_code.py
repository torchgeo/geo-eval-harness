"""Scan tests for references and report dead code outside the test suite."""

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

    gated_lines = [
        line for line in result.stdout.splitlines() if not line.startswith(("tests/", "tests\\"))
    ]
    for line in gated_lines:
        print(line)
    return 1 if gated_lines else 0


if __name__ == "__main__":
    sys.exit(main())
