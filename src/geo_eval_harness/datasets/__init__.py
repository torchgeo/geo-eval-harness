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

from .base import BandSpec, BenchDataset
from .loading import (
    get_bench_dataset_class,
    get_dataset_task,
    get_datasets,
    list_datasets,
)

__all__ = [
    "AID",
    "BENV2",
    "FLAIR2",
    "PASTIS",
    "PLANTEUR",
    "RESISC45",
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
    "MBigEarthNet",
    "MBrickKiln",
    "MEurosat",
    "MForestnet",
    "MPv4ger",
    "MSo2Sat",
    "PLANTEURLOIOGuadeloupe",
    "PLANTEURLOIOGuyane",
    "PLANTEURLOIOMartinique",
    "PLANTEURLOIOMayotte",
    "PLANTEURLOIOReunion",
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
    "AID": "aid",
    "BENV2": "benv2",
    "BurnScars": "burn_scars",
    "CaFFe": "caffe",
    "CloudSEN12": "cloudsen12",
    "DynamicEarthNet": "dynamic_earthnet",
    "EuroSAT": "eurosat",
    "EuroSATSpatial": "eurosat",
    "FLAIR2": "flair2",
    "Forestnet": "forestnet",
    "FieldsOfTheWorld": "fotw",
    "HOTOSMBuildings": "hotosm_buildings",
    "KuroSiwo": "kuro_siwo",
    "MBigEarthNet": "m_bigearthnet",
    "MBrickKiln": "m_brick_kiln",
    "MEurosat": "m_eurosat",
    "MForestnet": "m_forestnet",
    "MPv4ger": "m_pv4ger",
    "MSo2Sat": "m_so2sat",
    "PASTIS": "pastis",
    "PLANTEUR": "planteur",
    "PLANTEURLOIOGuadeloupe": "planteur",
    "PLANTEURLOIOMartinique": "planteur",
    "PLANTEURLOIOGuyane": "planteur",
    "PLANTEURLOIOReunion": "planteur",
    "PLANTEURLOIOMayotte": "planteur",
    "RESISC45": "resisc45",
    "So2Sat": "so2sat",
    "SpaceNet2": "spacenet2",
    "SpaceNet7": "spacenet7",
    "TreeSatAI": "treesatai",
    "UCMerced": "ucmerced",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_CLASSES:
        cls = getattr(import_module(f".{_LAZY_CLASSES[name]}", __name__), name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_CLASSES))
