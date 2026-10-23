"""Small model inputs shared by model contract tests."""

from geo_eval_harness.datasets.base import BandSpec


def bands(n: int = 2) -> list[BandSpec]:
    """Return distinct per-channel statistics without loading a dataset."""
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=float(10 * (i + 1)),
            std=float(2 * (i + 1)),
            min=0.0,
            max=255.0,
        )
        for i in range(n)
    ]
