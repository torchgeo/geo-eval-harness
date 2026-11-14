"""GeoBench V2 adapters with raw sensor values and shared wrapper defaults."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..hub import load_checksums
from .base import BenchDataset

logger = logging.getLogger(__name__)

V2_ROOT = Path("data/geobenchv2")


_V2_REGISTRY: dict[str, str] = {
    "benv2": "GeoBenchBENV2",
    "burn_scars": "GeoBenchBurnScars",
    "caffe": "GeoBenchCaFFe",
    "cloudsen12": "GeoBenchCloudSen12",
    "dynamic_earthnet": "GeoBenchDynamicEarthNet",
    "flair2": "GeoBenchFLAIR2",
    "forestnet": "GeoBenchForestnet",
    "fotw": "GeoBenchFieldsOfTheWorld",
    "kuro_siwo": "GeoBenchKuroSiwo",
    "pastis": "GeoBenchPASTIS",
    "so2sat": "GeoBenchSo2Sat",
    "spacenet2": "GeoBenchSpaceNet2",
    "spacenet7": "GeoBenchSpaceNet7",
    "treesatai": "GeoBenchTreeSatAI",
}

# Pinned ``aialliance/<name>`` commits; archive digests are in checksums/geobench_v2.sha256.
V2_REVISIONS: dict[str, str] = {
    "benv2": "20ac91314d2959d585faef744e07a2beb219759d",
    "burn_scars": "a31f44bd17e4889c33777576068a14461c9e436a",
    "caffe": "9c8b0e8886cbb9a91c4ef90d8676df9462d9afbd",
    "cloudsen12": "7b72e755b787231c2263ad70c9d04eba38bdbb93",
    "dynamic_earthnet": "b5374f7ad00aeedf8436029aab09d23362e319cd",
    "flair2": "e81a78d922f6e60953e5bab6404b1fc21896fdd1",
    "forestnet": "06c344cebeba8cc176688dceb48ff1a0b98327d4",
    "fotw": "53840eaa234c6e9dee8a21776b872b208998a946",
    "kuro_siwo": "25997dc683df721de44fe6f68a4ce1b36f4562eb",
    "pastis": "b4918975bf6650250715a0f922fd496266737a96",
    "so2sat": "dcaca83bf7acb78d92e5161499a1bb5d09716536",
    "spacenet2": "6e31423205fe0f4d11f7c3f29b2ab9590d0f97c1",
    "spacenet7": "fd2305b12db018e365c0d9831041e4a9e395d4bf",
    "treesatai": "f3305edc75e6bb4679129f068f877dbff04e6e43",
}

# KuroSiwo expects "val"; the other upstream loaders expect "validation".
_V2_VAL_AS_VAL: frozenset[str] = frozenset({"kuro_siwo"})


def list_v2_datasets() -> list[str]:
    """Return the sorted set of dataset names handled by the V2 adapter."""
    return sorted(_V2_REGISTRY)


def v2_checksums(dataset: str) -> dict[str, str]:
    """Return one dataset's archive digests, keyed by file name within its directory."""
    prefix = f"{dataset}/"
    return {
        name.removeprefix(prefix): digest
        for name, digest in load_checksums("geobench_v2.sha256").items()
        if name.startswith(prefix)
    }


class _ChainedTransform:
    """Canonicalize an upstream V2 sample, then apply the framework transform."""

    def __init__(self, canonicalize: Callable[[dict], dict], transform: Callable | None) -> None:
        self.canonicalize = canonicalize
        self.transform = transform

    def __call__(self, sample: dict) -> dict:
        sample = self.canonicalize(sample)
        if self.transform is None:
            return sample
        if "image" in sample:
            return self.transform(sample)
        image_keys = [k for k in sample if k.startswith("image_")]
        for index, key in enumerate(image_keys):
            wrapped = {"image": sample[key]}
            if index == 0 and "mask" in sample:
                wrapped["mask"] = sample["mask"]
            wrapped = self.transform(wrapped)
            sample[key] = wrapped["image"]
            if "mask" in wrapped:
                sample["mask"] = wrapped["mask"]
        return sample


class GeoBenchv2(Dataset):
    """Load a GeoBench V2 dataset through its upstream class.

    Args:
        root: Path to the GeoBench V2 collection root (the directory
            containing per-dataset subdirectories, e.g. ``data/geobenchv2``).
        dataset_name: One of :func:`list_v2_datasets`.
        split: ``"train"``, ``"val"``, or ``"test"``.
        band_order: Bands to load in upstream-expected shape (a flat ``list``
            for single-modality datasets, or ``dict[modality, list[str]]`` for
            multi-modality ones).
        transforms: Optional sample transform forwarded to the upstream class.
        **kwargs: Additional keyword arguments forwarded to the upstream class.
    """

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str,
        *,
        band_order: object | None = None,
        transforms: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if dataset_name not in _V2_REGISTRY:
            raise KeyError(
                f"Unknown GeoBench V2 dataset '{dataset_name}'. "
                f"Available: {', '.join(list_v2_datasets())}"
            )
        # Import GeoBench V2 only when needed to keep CLI startup fast.
        import geobench_v2.datasets as _gb_v2

        cls: type[Dataset] = getattr(_gb_v2, _V2_REGISTRY[dataset_name])
        upstream_split = (
            "val"
            if split == "val" and dataset_name in _V2_VAL_AS_VAL
            else "validation"
            if split == "val"
            else split
        )

        forward: dict[str, object] = {
            "root": Path(root) / dataset_name,
            "split": upstream_split,
            "transforms": transforms,
        }
        if band_order is not None:
            forward["band_order"] = band_order
        forward.update(kwargs)

        self._inner: Dataset = cls(**forward)
        self.dataset_name = dataset_name
        self.split = split

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict:
        return self._inner[index]


class _V2Dataset(BenchDataset):
    """Base class for every GeoBench V2 wrapper.

    Use ``band_order_strategy = "by_sensor"`` when the upstream loader groups bands by sensor.

    Override :meth:`canonicalize_sample` for different sample keys or temporal shapes.

    :attr:`upstream_kwargs` overrides loader arguments.
    """

    band_order_strategy: Literal["flat", "by_sensor"] = "flat"

    #: Override upstream loader defaults here, including ``return_stacked_image``.
    upstream_kwargs: ClassVar[dict[str, object]] = {}

    #: Whether the upstream loader accepts a time axis (``num_time_steps``).
    multi_temporal: ClassVar[bool] = False

    @classmethod
    def data_root(cls) -> Path:
        return V2_ROOT

    def build_band_order(self, bands: tuple[str, ...] | None) -> object:
        """Translate canonical band names into the upstream loader's shape."""
        specs = self.select_band_specs(bands)
        if self.band_order_strategy == "by_sensor":
            grouped: dict[str, list[str]] = {}
            for spec in specs:
                grouped.setdefault(spec.sensor, []).append(spec.source_name)
            return grouped
        return [spec.source_name for spec in specs]

    def canonicalize_sample(self, sample: dict) -> dict:
        """Adapt an upstream sample to ``image`` and ``label``/``mask`` keys.

        The default is a no-op; override for image pairs or temporal stacks.
        """
        return sample

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        time_steps: int | None = None,
    ) -> Dataset:
        """Return raw sensor values for a split.

        Use ``nn.Identity`` upstream; ``BenchModel`` owns per-channel normalization.
        ``time_steps`` requests a time series and needs :attr:`multi_temporal`.
        """
        del partition
        band_order = self.build_band_order(bands)

        kwargs: dict[str, object] = {
            "data_normalizer": nn.Identity,
            "download": False,
        }
        if self.band_order_strategy == "by_sensor":
            kwargs["return_stacked_image"] = True
        kwargs.update(self.upstream_kwargs)
        if time_steps is not None:
            if not self.multi_temporal:
                raise ValueError(f"{self.name} is not multi-temporal; drop time_steps.")
            kwargs["num_time_steps"] = int(time_steps)
            kwargs["temporal_output_format"] = "TCHW"

        return GeoBenchv2(
            root=self.data_root(),
            dataset_name=self.name,
            split=split,
            band_order=band_order,
            transforms=_ChainedTransform(self.canonicalize_sample, transform),
            **kwargs,
        )


class _OffsetMaskV2Dataset(_V2Dataset):
    """Restore SpaceNet's native two-class building masks.

    Upstream shifts native ``{0, 1}`` masks to ``{1, 2}``, reserving unused class 0.

    Remove that offset so the probe learns only the two real classes.

    See the SpaceNet2/7 loaders in https://github.com/The-AI-Alliance/GEO-Bench-2.
    """

    def canonicalize_sample(self, sample: dict) -> dict:
        """Reverse GeoBench's ``+1`` offset: mask ``1 -> 0``, ``2 -> 1``.

        Clamp reserved class 0 to no-building, avoiding negative labels before resizing.
        """
        mask = sample.get("mask")
        if mask is not None:
            mask = torch.as_tensor(mask)
            sample["mask"] = (mask - 1).clamp_(min=0)
        return sample
