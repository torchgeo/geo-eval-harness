"""Lightweight logging shared by command handlers."""

import logging


def setup_logging(*, verbose: bool = False) -> None:
    """Configure command logging without importing benchmark modules."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
