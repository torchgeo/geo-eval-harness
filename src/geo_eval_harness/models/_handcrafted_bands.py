"""Sensor-local spectral maps and their reproducible source metadata."""

from dataclasses import dataclass
from urllib.parse import quote

import torch

from geo_eval_harness.bands import BandSpec

from ._band_mapping import canonical_band_name, resolve_src_indices

EPS = 1e-6
INDEX_ROLES = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "ndbi": ("swir1", "nir"),
    "nbr": ("nir", "swir2"),
}
_OPTICAL_SENSORS = {"s2", "landsat", "aerial", "planet", "worldview"}
_NON_OPTICAL_SENSORS = {"s1", "sar", "dem", "elevation"}


@dataclass(frozen=True)
class FeatureMap:
    """Describe one raw band or a normalized-difference map."""

    name: str
    sensor: str
    sources: tuple[int, ...]
    roles: tuple[str, ...]

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract this map without resampling or synthesizing source channels.

        Args:
            images: Input images in ``(B, C, H, W)`` order.

        Returns:
            One raw or derived map per image, with shape ``(B, H, W)``.
        """
        first = images[:, self.sources[0]]
        if len(self.sources) == 1:
            return first
        return normalized_difference(first, images[:, self.sources[1]])

    def metadata(self, bands: list[BandSpec]) -> dict[str, object]:
        """Return JSON-serializable source indices and original band identifiers.

        Args:
            bands: Metadata in input-channel order.

        Returns:
            The map identity and the actual band supplying each canonical role.
        """
        return {
            "name": self.name,
            "sensor": self.sensor,
            "kind": "band" if len(self.sources) == 1 else "index",
            "available": True,
            "sources": [
                {
                    "role": role,
                    "channel": channel,
                    "name": bands[channel].name,
                    "source_name": bands[channel].source_name,
                    "canonical_name": canonical_band_name(bands[channel].name),
                }
                for role, channel in zip(self.roles, self.sources, strict=True)
            ],
        }


def normalized_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compute a ratio, returning zero for zero or nearly cancelling denominators.

    Both operands are divided by their shared absolute maximum first. This
    prevents overflow without changing the ratio. Denominators with magnitude
    at most ``1e-6 * (abs(first) + abs(second))`` produce zero, not an epsilon
    division. Finite negative measurements are accepted; ratios are not clipped.

    Args:
        first: First input map.
        second: Second input map in the same sensor units.

    Returns:
        The guarded normalized difference ``(first - second) / (first + second)``.
    """
    scale = torch.maximum(first.abs(), second.abs())
    scale = torch.where(scale > 0, scale, 1.0)
    first, second = first / scale, second / scale
    denominator = first + second
    valid = denominator.abs() > EPS * (first.abs() + second.abs())
    return torch.where(valid, (first - second) / torch.where(valid, denominator, 1.0), 0.0)


def _optical(band: BandSpec) -> bool:
    sensor = band.sensor.lower()
    return sensor not in _NON_OPTICAL_SENSORS and (
        sensor in _OPTICAL_SENSORS or band.wavelength_um is not None
    )


def _identifier(value: str) -> str:
    return quote(value, safe="").replace(".", "%2E")


def _sensor_maps(
    bands: list[BandSpec], sensor: str, channels: list[int]
) -> tuple[list[FeatureMap], list[dict[str, object]]]:
    candidates = [channel for channel in channels if _optical(bands[channel])]
    resolved = resolve_src_indices([bands[channel] for channel in candidates])
    sources = {role: candidates[index] for role, index in resolved.items()}
    if "nir" not in sources and "nir_narrow" in sources:
        sources["nir"] = sources["nir_narrow"]
    maps: list[FeatureMap] = []
    skipped: list[dict[str, object]] = []
    for index, roles in INDEX_ROLES.items():
        name = f"{_identifier(sensor)}.index.{index}"
        missing = [role for role in roles if role not in sources]
        if missing:
            skipped.append(
                {
                    "name": name,
                    "sensor": sensor,
                    "kind": "index",
                    "available": False,
                    "missing_roles": missing,
                    "reason": "missing optical bands" if candidates else "non-optical sensor",
                    "sources": [],
                }
            )
        else:
            maps.append(FeatureMap(name, sensor, tuple(sources[role] for role in roles), roles))
    return maps, skipped


def build_maps(bands: list[BandSpec]) -> tuple[list[FeatureMap], list[dict[str, object]]]:
    """Build raw maps followed by available sensor-local indices.

    Args:
        bands: Ordered input-channel metadata.

    Returns:
        Available map definitions and explicit skipped-index metadata.
    """
    groups: dict[str, list[int]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    maps: list[FeatureMap] = []
    for channel, band in enumerate(bands):
        groups.setdefault(band.sensor, []).append(channel)
        key = (band.sensor, band.name)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        suffix = f"#{occurrence + 1}" if occurrence else ""
        name = f"{_identifier(band.sensor)}.band.{_identifier(band.name)}{suffix}"
        maps.append(FeatureMap(name, band.sensor, (channel,), (canonical_band_name(band.name),)))
    skipped: list[dict[str, object]] = []
    for sensor, channels in groups.items():
        indices, missing = _sensor_maps(bands, sensor, channels)
        maps.extend(indices)
        skipped.extend(missing)
    return maps, skipped
