"""Deterministic, sensor-aware handcrafted baseline with cumulative complexity."""

from collections.abc import Iterator
from copy import deepcopy

import torch

from geo_eval_harness.bands import BandSpec

from ._handcrafted_bands import build_maps
from ._handcrafted_structure import STRUCTURE_NAMES, structure
from ._handcrafted_texture import STAT_NAMES, TEXTURE_NAMES, statistics, texture
from ._normalization import NormalizationStrategy
from .interface import BenchModel

_MAP_CHUNK_SIZE = 2
_LEVEL_NAMES = (STAT_NAMES, TEXTURE_NAMES, STRUCTURE_NAMES)


class HandcraftedBench(BenchModel):
    """Extract generic statistics, spectral indices, texture and spatial structure.

    For ``M`` raw channels plus available index maps, levels 1/2/3 return
    ``9*M``, ``21*M`` and ``37*M`` features respectively. Each level is a strict
    prefix of the next, independent of spatial dimensions. All raw channels,
    including SAR, contribute at every level. There are no learned parameters,
    image gradients, pretrained weights, random operations or fitted statistics.

    Args:
        bands: Ordered input-channel metadata. Index roles resolve independently
            within each sensor, using semantic names, never nearest-band filling.
        level: Cumulative complexity, one of 1, 2 or 3.
        normalization: Base-class input normalization. Defaults to identity.
            Physical spectral indices require raw, consistently scaled optical
            bands; the presets use ``input.normalization: none`` by default.
            Other strategies are honored, not silently recorded as identity.
        **kwargs: Additional benchmark arguments forwarded to :class:`BenchModel`.

    Attributes:
        feature_names: Ordered, sensor-prefixed output column names.

    Raises:
        ValueError: If the level is not an integer in 1..3 or bands is empty.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        level: int = 2,
        normalization: NormalizationStrategy | str = NormalizationStrategy.IDENTITY,
        **kwargs: object,
    ) -> None:
        if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2, 3):
            raise ValueError("HandcraftedBench level must be an integer in {1, 2, 3}.")
        super().__init__(bands=bands, normalization=normalization, **kwargs)
        self.level = level
        self._maps, self._skipped_indices = build_maps(self.bands)
        self.feature_names: tuple[str, ...] = tuple(
            f"{feature_map.name}.{stat}"
            for names in _LEVEL_NAMES[:level]
            for feature_map in self._maps
            for stat in names
        )

    @property
    def num_features(self) -> int:
        """Number of output columns, fixed by bands and level at construction."""
        return len(self.feature_names)

    @property
    def feature_metadata(self) -> list[dict[str, object]]:
        """JSON-serializable map records, including available and skipped indices.

        Each available record gives source channels, actual band labels, canonical
        roles and its output feature names/column indices. Skipped indices have
        ``available=False``, missing roles, a reason, and no output columns.
        Narrow-NIR substitution is explicit in the source's canonical name.
        """
        records: list[dict[str, object]] = []
        columns = {name: index for index, name in enumerate(self.feature_names)}
        for feature_map in self._maps:
            record = feature_map.metadata(self.bands)
            names = [
                f"{feature_map.name}.{stat}"
                for level_names in _LEVEL_NAMES[: self.level]
                for stat in level_names
            ]
            record.update(feature_names=names, columns=[columns[name] for name in names])
            records.append(record)
        records.extend(
            {**deepcopy(skipped), "feature_names": [], "columns": []}
            for skipped in self._skipped_indices
        )
        return records

    def _map_chunks(self, images: torch.Tensor) -> Iterator[torch.Tensor]:
        for start in range(0, len(self._maps), _MAP_CHUNK_SIZE):
            yield torch.stack(
                [
                    mapping.extract(images)
                    for mapping in self._maps[start : start + _MAP_CHUNK_SIZE]
                ],
                dim=1,
            )

    def _validate_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != self.num_channels:
            raise ValueError(f"Expected (B, {self.num_channels}, H, W) input.")
        if min(images.shape) < 1:
            raise ValueError("Batch and spatial dimensions must be positive.")
        if images.is_complex() or not torch.isfinite(images).all():
            raise ValueError("HandcraftedBench requires finite, real input values.")

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return float32 features on the input device, without image gradients.

        Args:
            images: Normalized input with shape ``(B, C, H, W)``. No resizing
                occurs here; the benchmark owns input interpolation.

        Returns:
            A tensor of shape ``(B, num_features)``. Outside caller inference mode,
            it can train a downstream head even under caller autocast.

        Raises:
            ValueError: For invalid dimensions, channels or nonfinite input.
            FloatingPointError: If values overflow float32 feature arithmetic.
        """
        self._validate_images(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            images = images.float()
            levels: list[list[torch.Tensor]] = [[] for _ in range(self.level)]
            for maps in self._map_chunks(images):
                stats = statistics(maps)
                levels[0].append(stats.flatten(1))
                if self.level >= 2:
                    levels[1].append(texture(maps).flatten(1))
                if self.level >= 3:
                    levels[2].append(structure(maps, stats).flatten(1))
            features = torch.cat([torch.cat(parts, dim=1) for parts in levels], dim=1)
        if not torch.isfinite(features).all():
            raise FloatingPointError(
                "Handcrafted features overflowed; check the input units/range."
            )
        return features
