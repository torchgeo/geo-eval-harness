"""UniverSat (AnySat v2) wrapper for geo-eval-harness.

UniverSat (https://github.com/gastruc/UniverSat) takes a ``{modality: tensor}``
dict and embeds each channel by its wavelength/sensor code, accepting arbitrary
band counts per sensor. The wrapper groups input channels by ``BandSpec.sensor``,
runs ``model.encode(...)``, and mean-pools the spatial tokens to ``(B, 768)``.

Model code loads via ``torch.hub`` and weights from the HuggingFace Hub
(``g-astruc/UniverSat``); no pip package. Inputs use the framework's
``bandspec_zscore`` normalization, matching UniverSat's own GeoBench loader.
"""

import logging
from collections import defaultdict
from typing import ClassVar, Protocol, Self, cast

import torch
import torch.nn.functional as F

from geo_eval_harness.datasets.base import BandSpec

from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

# Pin model code for reproducible torch.hub loads.
UNIVERSAT_REPO = "gastruc/UniverSat"
UNIVERSAT_REF = "d4e8712d29651c34c4fdb9f9d2c0b168dd5c678b"

# BandSpec.sensor -> UniverSat modality name (must be a key the released
# checkpoint built a projector for; see DEFAULT_MODALITIES_DICT in the repo).
_SENSOR_TO_MODALITY: dict[str, str] = {
    "s2": "s2",
    "sentinel2": "s2",
    "sentinel-2": "s2",
    "s1": "s1",
    "sentinel1": "s1",
    "sar": "s1",
    "naip": "naip",
    "aerial": "aerial",
    "spot": "spot",
    "landsat": "l8",
}

# Physical resolution (m/px) and sub-patch factor per UniverSat modality,
# from the model's modality_registry.
_MODALITY_INPUT_RES: dict[str, float] = {
    "s2": 10.0,
    "s1": 10.0,
    "naip": 1.25,
    "aerial": 0.2,
    "spot": 1.0,
    "l8": 10.0,
    "l7": 30.0,
}
_MODALITY_SUBPATCH: dict[str, int] = {
    "s2": 1,
    "s1": 1,
    "naip": 10,
    "aerial": 10,
    "spot": 10,
    "l8": 1,
    "l7": 1,
}


# SAR bands are sensor codes (not wavelengths) in UniverSat's s1 modality.
def _sar_code(name: str) -> str:
    n = name.lower()
    if "ratio" in n or "vv_vh" in n or "vh_vv" in n:
        return "Ratio_VV_VH"
    if "vv" in n:
        return "VV"
    if "vh" in n:
        return "VH"
    raise ValueError(f"Cannot map SAR band {name!r} to a UniverSat s1 code (VV/VH/Ratio_VV_VH).")


def _build_sensor_groups(
    bands: list[BandSpec],
    modality: str | None = None,
    input_res: float | None = None,
) -> list[dict]:
    """Group channels by sensor and resolve each group's UniverSat modality.

    Returns one dict per sensor (first-seen order) with keys ``modality``,
    ``indices`` (channel positions in ``bands``), ``wavelengths`` (floats, or
    s1 sensor codes), ``input_res``, and ``subpatch``. ``modality`` / ``input_res``
    overrides apply only when the input has a single sensor.
    """
    sensor_to_indices: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for i, b in enumerate(bands):
        s = b.sensor.lower()
        if s not in sensor_to_indices:
            order.append(s)
        sensor_to_indices[s].append(i)
    single_sensor = len(order) == 1

    groups: list[dict] = []
    seen: set[str] = set()
    for sensor in order:
        mod = modality if (modality and single_sensor) else _SENSOR_TO_MODALITY.get(sensor)
        if mod is None:
            raise ValueError(
                f"No UniverSat modality mapping for sensor {sensor!r}. "
                f"Pass `modality=` explicitly. Known: {sorted(_SENSOR_TO_MODALITY)}."
            )
        if mod in seen:
            raise ValueError(
                f"Two sensors map to the same UniverSat modality {mod!r}; "
                "cannot disambiguate the encode() dict."
            )
        seen.add(mod)
        indices = sensor_to_indices[sensor]
        group_bands = [bands[i] for i in indices]
        if mod == "s1":
            wavelengths: list[float | str] = [_sar_code(b.name) for b in group_bands]
        else:
            band_wavelengths = [b.wavelength_um for b in group_bands]
            if any(w is None for w in band_wavelengths):
                missing = [b.name for b in group_bands if b.wavelength_um is None]
                raise ValueError(
                    f"UniverSat needs a wavelength per channel; BandSpecs {missing} "
                    f"have wavelength_um=None."
                )
            wavelengths = [w for w in band_wavelengths if w is not None]
        res = input_res if (input_res is not None and single_sensor) else _MODALITY_INPUT_RES[mod]
        groups.append(
            {
                "modality": mod,
                "indices": indices,
                "wavelengths": wavelengths,
                "input_res": res,
                "subpatch": _MODALITY_SUBPATCH.get(mod, 1),
            }
        )
    return groups


class UniverSatEncoder(Protocol):
    """The inference methods exposed by UniverSat's torch.hub model."""

    def eval(self) -> Self:
        """Switch to inference mode."""
        ...

    def encode(  # noqa: PLR0913 - upstream model API
        self,
        x: dict[str, torch.Tensor],
        *,
        patch_size: float,
        output_grid: int | None,
        wavelengths: dict[str, list[float | str]],
        input_res: dict[str, float],
        subpatches: dict[str, int],
    ) -> tuple[torch.Tensor, object]:
        """Encode modality tensors into spatial tokens."""
        ...


class UniverSatBenchModel(BenchModel):
    """BenchModel wrapper for the UniverSat (AnySat v2) EO encoder.

    Channels are grouped by ``BandSpec.sensor`` and each group is routed to its
    UniverSat modality, so single- and multi-sensor inputs (e.g. S2+SAR) both
    work without a fixed channel layout.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        modality: Force the UniverSat modality (single-sensor inputs only);
            ``None`` (default) maps from ``BandSpec.sensor``.
        patch_size: Patch size in metres passed to ``encode`` (default 10, i.e.
            one token per Sentinel-2 pixel -- matches UniverSat's own
            ``scale_base=1`` GeoBench recipe for its one 10 m-GSD S2 dataset,
            m-brick-kiln; https://github.com/gastruc/UniverSat's other GeoBench
            configs tune this per dataset GSD and have no S2 classification
            entry to match otherwise).
        output_grid: Side ``G`` of the ``GxG`` token grid. ``None`` (default)
            lets UniverSat infer the natural patch grid; tokens are mean-pooled.
        input_res: Override the physical resolution (m/px), single-sensor only;
            ``None`` uses the registry value per modality.
        normalize: If True, L2-normalize the output embeddings.
        repo: torch.hub source for the model code.
        repo_ref: Pinned git ref for the torch.hub load.
    """

    # Output embedding dimension of the released Base model.
    embed_dim: ClassVar[int] = 768

    def __init__(  # noqa: PLR0913 - public YAML options
        self,
        bands: list[BandSpec],
        *,
        modality: str | None = None,
        patch_size: float = 10.0,
        output_grid: int | None = None,
        input_res: float | None = None,
        normalize: bool = False,
        repo: str = UNIVERSAT_REPO,
        repo_ref: str | None = UNIVERSAT_REF,
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization=normalization, **_kwargs)

        self._groups = _build_sensor_groups(self.bands, modality=modality, input_res=input_res)

        self.patch_size = patch_size
        self.output_grid = output_grid
        self.do_normalize = normalize

        source = f"{repo}:{repo_ref}" if repo_ref else repo
        # Single-pass extraction does not amortize torch.compile's max-autotune warmup.
        # Varying dataset shapes can also exhaust its recompile limit.
        # torch.hub's fork check only accepts a commit that is a branch or tag head,
        # so the pinned commit fails once upstream main moves past it.
        self.model = cast(
            UniverSatEncoder,
            torch.hub.load(
                source,
                "from_pretrained",
                trust_repo=True,
                skip_validation=True,
                compile=False,
            ),
        ).eval()
        logger.info(
            "UniverSat loaded (modalities=%s, %d channels, patch_size=%sm)",
            [g["modality"] for g in self._groups],
            self.num_channels,
            self.patch_size,
        )

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, 768)`` mean-pooled tile features."""
        x, wavelengths, input_res, subpatches = {}, {}, {}, {}
        for g in self._groups:
            mod = g["modality"]
            x[mod] = images[:, g["indices"]]
            wavelengths[mod] = g["wavelengths"]
            input_res[mod] = g["input_res"]
            subpatches[mod] = g["subpatch"]
        tokens, _ = self.model.encode(
            x,
            patch_size=self.patch_size,
            output_grid=self.output_grid,
            wavelengths=wavelengths,
            input_res=input_res,
            subpatches=subpatches,
        )
        embeddings = tokens.mean(dim=1)
        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
