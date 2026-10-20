"""Benchmark model implementations and exports."""

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy
from .handcrafted import HandcraftedBench
from .image_stats import ImageStatsBench
from .interface import BenchModel
from .olmoearth import OlmoEarthBenchModel
from .rcf import RCFBench, RCFModelSettings
from .sam3 import SAM3Encoder
from .segmentation_heads import (
    ConvBlockHead,
    DPTHead,
    FPNHead,
    LinearHead,
    PatchLinearHead,
    UPerNetHead,
)
from .terrafm import TerraFMBench
from .terratorch_models import (
    TerraTorchClayBench,
    TerraTorchPrithviBench,
    TerraTorchTerraMindBench,
)
from .timm import TimmModelSettings, TimmPatchBenchModel
from .torchgeo_models import (
    TorchGeoCromaBench,
    TorchGeoDEOBench,
    TorchGeoDOFABench,
    TorchGeoEarthLocBench,
    TorchGeoPanopticonBench,
    TorchGeoResNetBench,
    TorchGeoScaleMAEBench,
    TorchGeoSwinBench,
)
from .universat import UniverSatBenchModel

__all__: list[str] = [
    "BenchModel",
    "ConvBlockHead",
    "DPTHead",
    "FPNHead",
    "HandcraftedBench",
    "ImageStatsBench",
    "InputUnit",
    "LinearHead",
    "NormalizationStrategy",
    "OlmoEarthBenchModel",
    "PatchLinearHead",
    "RCFBench",
    "RCFModelSettings",
    "SAM3Encoder",
    "TerraFMBench",
    "TerraTorchClayBench",
    "TerraTorchPrithviBench",
    "TerraTorchTerraMindBench",
    "TimmModelSettings",
    "TimmPatchBenchModel",
    "TorchGeoCromaBench",
    "TorchGeoDEOBench",
    "TorchGeoDOFABench",
    "TorchGeoEarthLocBench",
    "TorchGeoPanopticonBench",
    "TorchGeoResNetBench",
    "TorchGeoScaleMAEBench",
    "TorchGeoSwinBench",
    "UPerNetHead",
    "UniverSatBenchModel",
]
