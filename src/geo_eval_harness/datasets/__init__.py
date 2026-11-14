"""Benchmark dataset registry for geo-eval-harness.

Dataset classes load only when requested, keeping imports and CLI startup fast.

Public API
----------
.. autofunction:: get_datasets
.. autofunction:: get_bench_dataset_class
.. autofunction:: get_dataset_task
.. autofunction:: list_datasets
.. autoclass:: BandSpec
.. autoclass:: BenchDataset
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import BandSpec, BenchDataset
from .loading import (
    _REGISTRY_SPEC,
    get_bench_dataset_class,
    get_dataset_task,
    get_datasets,
    list_datasets,
)

if TYPE_CHECKING:
    from .open_cities import OpenCities

__all__ = [
    "AID",
    "BENV2",
    "FLAIR2",
    "PASTIS",
    "PLANTEUR",
    "RESISC45",
    "BRIGHTPostSAR",
    "BRIGHTPostSARRGB",
    "BandSpec",
    "BenchDataset",
    "BurnScars",
    "CaFFe",
    "CloudSEN12",
    "DynamicEarthNet",
    "EuroSAT",
    "EuroSATSpatial",
    "FieldsOfTheWorld",
    "Forestnet",
    "HOTOSMBuildings",
    "KuroSiwo",
    "LEVIRCDPost",
    "LoveDA",
    "MBigEarthNet",
    "MBrickKiln",
    "MEurosat",
    "MForestnet",
    "MPv4ger",
    "MSo2Sat",
    "OSCDPost",
    "OpenCities",
    "OpenEarthMapSAR",
    "PLANTEURLOIOGuadeloupe",
    "PLANTEURLOIOGuyane",
    "PLANTEURLOIOMartinique",
    "PLANTEURLOIOMayotte",
    "PLANTEURLOIOReunion",
    "Sen1Floods11Base",
    "Sen1Floods11S1",
    "Sen1Floods11S2",
    "So2Sat",
    "SpaceNet2",
    "SpaceNet7",
    "TreeSatAI",
    "UCMerced",
    "get_bench_dataset_class",
    "get_dataset_task",
    "get_datasets",
    "list_datasets",
]

_LAZY_CLASSES: dict[str, str] = {
    class_name: module for module, class_name, _ in _REGISTRY_SPEC.values()
}
_LAZY_CLASSES["Sen1Floods11Base"] = "sen1floods11"


def __getattr__(name: str) -> object:
    if name in _LAZY_CLASSES:
        cls = getattr(import_module(f".{_LAZY_CLASSES[name]}", __name__), name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_CLASSES))
