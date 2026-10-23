"""Dataset metadata and split-loading contracts.

Each :class:`BenchDataset` declares metadata and loads a PyTorch dataset for each split.

Each family fixes its data root under ``data/``, relative to the working directory.

Torch is imported only for type checking so metadata stays available without loading it.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from geo_eval_harness.bands import BandSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.utils.data import Dataset

__all__ = ["BandSpec", "BenchDataset"]

logger = logging.getLogger(__name__)


class BenchDataset(ABC):
    """Abstract base class for benchmark datasets.

    Subclasses must define the class-level metadata attributes listed below
    and implement :meth:`get_dataset` and :meth:`data_root`.

    Attributes:
        name: Dataset identifier used on the command line (e.g. ``"m-eurosat"``).
        task: ``"classification"`` or ``"segmentation"``.
        num_classes: Number of output classes.
        bands: Ordered list of all available spectral bands with statistics.
        rgb_bands: Short names of the bands to use for RGB-only mode.
        split_sizes: Number of samples per split for the *default* partition,
            keyed by ``"train"``, ``"val"``, ``"test"``.
        multilabel: Whether labels are multi-hot (e.g. BigEarthNet).
        supports_partitions: Whether the dataset honours a non-default
            ``partition`` argument (V1 GeoBench datasets do; V2 does not).
    """

    name: str
    task: Literal["classification", "segmentation"]
    num_classes: int
    bands: ClassVar[list[BandSpec]]
    rgb_bands: ClassVar[list[str]]
    split_sizes: ClassVar[dict[str, int]]
    multilabel: bool = False
    supports_partitions: bool = False

    @property
    def num_channels(self) -> int:
        """Total number of spectral bands."""
        return len(self.bands)

    @property
    def rgb_indices(self) -> list[int]:
        """Indices into :attr:`bands` for the RGB subset."""
        names = [b.name for b in self.bands]
        return [names.index(s) for s in self.rgb_bands if s in names]

    @classmethod
    @abstractmethod
    def data_root(cls) -> Path:
        """Return the directory the upstream loader expects.

        For V1/V2 wrappers this is the *parent* directory containing per-dataset
        subdirectories (e.g. ``data/classification_v1.0``); for torchgeo
        wrappers it is the dataset's own root (e.g. ``data/eurosat``).
        """

    @classmethod
    def select_band_specs(cls, bands: "Iterable[str] | None") -> list[BandSpec]:
        """Return the :class:`BandSpec` entries matching *bands*.

        Preserves the order given by *bands*. Raises ``ValueError`` if any
        requested band is not declared on the dataset.
        """
        if bands is None:
            return list(cls.bands)
        by_name = {b.name: b for b in cls.bands}
        result: list[BandSpec] = []
        for name in bands:
            if name not in by_name:
                raise ValueError(
                    f"{cls.__name__}: unknown band {name!r}; available: {sorted(by_name)}"
                )
            result.append(by_name[name])
        return result

    @classmethod
    def resolve_band_specs(cls, selection: "str | Iterable[str]") -> list[BandSpec]:
        """Resolve ``rgb``, ``all``, or explicit names using this dataset's metadata.

        Band order and objects are preserved without loading dataset samples.
        """
        if not isinstance(selection, str):
            return cls.select_band_specs(selection)
        if selection == "rgb":
            return cls.select_band_specs(cls.rgb_bands)
        if selection == "all":
            return cls.select_band_specs(None)
        raise ValueError(
            f"Unknown band selection {selection!r}; use rgb, all, or explicit band names"
        )

    @abstractmethod
    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: "Callable | None" = None,
    ) -> "Dataset":
        """Return a PyTorch :class:`~torch.utils.data.Dataset` for a split.

        Datasets always emit raw float32 values; normalization is the
        :class:`~geo_eval_harness.models.interface.BenchModel`'s job.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            partition: Partition name (V1 only, e.g. ``"0.01x_train"``).
                Ignored by datasets where :attr:`supports_partitions` is
                ``False``.
            bands: Tuple of canonical band names to load. ``None`` loads all.
            transform: Optional sample transform callable.
        """
