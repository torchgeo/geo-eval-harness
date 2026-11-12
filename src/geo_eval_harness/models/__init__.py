"""Benchmark model implementations and exports."""

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy
from .anysat import AnySatBenchModel
from .eosfm import EoSFMBenchModel
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
    NeckedHead,
    PatchLinearHead,
    UNetHead,
    UPerNetHead,
    ViTNeck,
)
from .terrafm import TerraFMBench
from .terratorch_models import (
    TerraTorchClayBench,
    TerraTorchPrithviBench,
    TerraTorchTerraMindBench,
)
from .thor import ThorBenchModel
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
    "AnySatBenchModel",
    "BenchModel",
    "ConvBlockHead",
    "DPTHead",
    "EoSFMBenchModel",
    "FPNHead",
    "HandcraftedBench",
    "ImageStatsBench",
    "InputUnit",
    "LinearHead",
    "NeckedHead",
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
    "ThorBenchModel",
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
    "UNetHead",
    "UPerNetHead",
    "UniverSatBenchModel",
    "ViTNeck",
]
