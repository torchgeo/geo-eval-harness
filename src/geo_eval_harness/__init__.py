"""Public package exports for geo-eval-harness.

Benchmark dependencies load on first use to keep package imports and CLI startup fast.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geo-eval-harness")
except PackageNotFoundError:  # allow-except: source-only imports before package installation
    __version__ = "0.5.0"

__author__ = "geo-eval-harness contributors"

__all__: list[str] = [
    "bootstrap_map",
    "evaluate_knn",
    "evaluate_logistic",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from geo_eval_harness import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
