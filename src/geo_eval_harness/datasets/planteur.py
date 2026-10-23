"""PLANTEUR crop-type segmentation over France's overseas territories, from Hugging Face."""

import csv
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset

logger = logging.getLogger(__name__)

NUM_MONTHS = 12
PATCH_SIZE = 128
SPLITS = ("train", "val", "test")
# Linear backscatter floor before the dB conversion, as in OlmoEarth's own S1 pipeline.
S1_LINEAR_FLOOR = 1e-10
# File name prefix and per-sample array shape of each converted sensor array.
SENSOR_ARRAYS: dict[str, tuple[str, tuple[int, ...], type]] = {
    "s2": ("S2", (NUM_MONTHS, 12, PATCH_SIZE, PATCH_SIZE), np.uint16),
    "s1": ("S1", (NUM_MONTHS, 2, PATCH_SIZE, PATCH_SIZE), np.float32),
}


def _patch_centres(metadata_path: Path) -> dict[int, tuple[str, float, float]]:
    """Return ``{patch id: (island, lon, lat)}`` from the release's ``metadata.geojson``."""
    features = json.loads(metadata_path.read_text())["features"]
    centres = {}
    for feature in features:
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        lon, lat = ring[:-1].mean(axis=0)
        properties = feature["properties"]
        centres[int(properties["ID_PATCH"])] = (properties["Island"], float(lon), float(lat))
    return centres


def to_db(backscatter: np.ndarray) -> np.ndarray:
    """Convert linear Sentinel-1 backscatter to dB, flooring no-data zeros."""
    return 10 * np.log10(np.clip(backscatter, S1_LINEAR_FLOOR, None))


def _write_split(upstream: Path, target: Path, split: str, ids: list[int]) -> None:
    """Fill ``<split>_<name>.npy.tmp`` memory maps for *ids*, releasing them on return."""
    outputs = {
        sensor: np.lib.format.open_memmap(
            target / f"{split}_{sensor}.npy.tmp", mode="w+", dtype=dtype, shape=(len(ids), *shape)
        )
        for sensor, (_, shape, dtype) in SENSOR_ARRAYS.items()
    }
    mask = np.lib.format.open_memmap(
        target / f"{split}_mask.npy.tmp",
        mode="w+",
        dtype=np.uint8,
        shape=(len(ids), PATCH_SIZE, PATCH_SIZE),
    )
    for index, patch_id in enumerate(ids):
        for sensor, (prefix, shape, dtype) in SENSOR_ARRAYS.items():
            array = np.load(upstream / f"DATA_{prefix}_MONTHLY" / f"{prefix}_{patch_id}.npy")
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"PLANTEUR {prefix}_{patch_id}: unexpected {array.dtype} array {array.shape}."
                )
            outputs[sensor][index] = to_db(array) if sensor == "s1" else array
        mask[index] = np.load(upstream / "ANNOTATIONS" / f"TARGET_{patch_id}.npy")[0]
    for memmap in (*outputs.values(), mask):
        memmap.flush()


def convert_planteur(upstream: Path, target: Path, *, expected_sizes: dict[str, int]) -> None:
    """Stack the release's per-patch monthly arrays into one memory-mappable file per split.

    Writes ``<split>_s2.npy``, ``<split>_s1.npy`` (backscatter in dB) and ``<split>_mask.npy`` plus
    ``metadata.csv`` (patch id, split, island, centre lon/lat) under *target*, and copies
    the leave-one-island-out lists to ``loio.json``.

    Args:
        upstream: Hugging Face snapshot holding ``DATA_S2_MONTHLY``, ``DATA_S1_MONTHLY``,
            ``ANNOTATIONS``, ``splits/443.json``, ``splits/loio.json`` and ``metadata.geojson``.
        target: Output directory.
        expected_sizes: Required number of patches per split.

    Raises:
        ValueError: If the split sizes or any per-patch array shape differ from the release.
    """
    splits = json.loads((upstream / "splits" / "443.json").read_text())["splits"]
    centres = _patch_centres(upstream / "metadata.geojson")
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        ids = splits[split]
        if len(ids) != expected_sizes[split]:
            raise ValueError(
                f"PLANTEUR {split}: expected {expected_sizes[split]} patches, got {len(ids)}."
            )
        _write_split(upstream, target, split, ids)
        # Rename only after _write_split returns: Windows cannot replace a memory-mapped file.
        for name in (*SENSOR_ARRAYS, "mask"):
            (target / f"{split}_{name}.npy.tmp").replace(target / f"{split}_{name}.npy")
        for patch_id in ids:
            island, lon, lat = centres[patch_id]
            rows.append((patch_id, split, island, f"{lon:.6f}", f"{lat:.6f}"))
        logger.info("Converted PLANTEUR %s split (%d patches).", split, len(ids))
    with (target / "metadata.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id_patch", "split", "island", "lon", "lat"))
        writer.writerows(rows)
    shutil.copyfile(upstream / "splits" / "loio.json", target / "loio.json")


def _split_positions(root: Path) -> dict[int, tuple[str, int]]:
    """Return ``{patch id: (default split, row)}`` for the converted arrays under *root*."""
    positions: dict[int, tuple[str, int]] = {}
    rows = dict.fromkeys(SPLITS, 0)
    with (root / "metadata.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            positions[int(record["id_patch"])] = (record["split"], rows[record["split"]])
            rows[record["split"]] += 1
    return positions


class PLANTEUR(BenchDataset, Dataset):
    """Crop-type semantic segmentation (8 classes) in France's overseas territories.

    Guadeloupe, Martinique, Guyane, Réunion, Mayotte and Saint-Martin: 2,750 patches of
    128x128 px at 10 m with twelve 2019 monthly mosaics of Sentinel-2 L2A (12 bands) and
    Sentinel-1 RTC (VV, VH backscatter in dB), labelled from the French Land Parcel
    Identification System (RPG 2019). Classes: 0 Background, 1 Meadow,
    2 Fruits/vegetables/flowers, 3 Orchard, 4 Sugarcane, 5 Banana, 6 Pineapple,
    7 Tropical tuber; 255 is void (crops outside the taxonomy), which the segmentation
    probe ignores. Uses the official spatial-block, territory-stratified split; the
    ``planteur_loio_*`` subclasses hold out one territory each.
    Source: https://huggingface.co/datasets/allenai/planteur-benchmark (ODC-BY).

    Samples hold the month chosen by ``time_steps`` as in PASTIS: ``None`` or 1 returns
    the first month as ``(C, H, W)``; ``T > 1`` returns ``T`` evenly spaced months as
    ``(T, C, H, W)``.
    """

    name = "planteur"
    task = "segmentation"
    num_classes = 8
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 1896, "val": 411, "test": 443}
    supports_partitions = False

    # Ordered as stored: the S2 mosaic channels, then S1. Statistics over all months of the
    # default train split, which every leave-one-island-out fold shares.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b02", "B02", mean=1741.05, std=2596.19, min=0, max=19952, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=1865.16, std=2393.66, min=0, max=18608, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=1700.7, std=2363.49, min=0, max=17760, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=2223.34, std=2390.93, min=0, max=17342, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=3359.48, std=2057.37, min=0, max=17089, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=3780.22, std=2006.34, min=0, max=16919, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=3766.73, std=1996.27, min=0, max=16736, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=3962.25, std=1954.83, min=0, max=16622, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=2432.54, std=1440.8, min=0, max=15440, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=1606.93, std=1266.72, min=0, max=15257, wavelength_um=2.19),
        BandSpec("s2", "b01", "B01", mean=1765.41, std=2687.13, min=1, max=21078, wavelength_um=0.443),
        BandSpec("s2", "b09", "B09", mean=4655.6, std=3305.04, min=1, max=17183, wavelength_um=0.945),
        BandSpec("s1", "vv", "VV", mean=-8.0434, std=3.5684, min=-100, max=35.975),
        BandSpec("s1", "vh", "VH", mean=-14.0961, std=3.441, min=-100, max=27.7915),
    ]
    # fmt: on

    def __init__(
        self,
        root: Path | None = None,
        rows: list[tuple[str, int]] | None = None,
        *,
        band_indices: list[int] | None = None,
        months: list[int] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        """Index *rows* (default split, row) of the converted arrays under *root*."""
        self.root = root
        self.rows = [] if rows is None else rows
        self.months = [0] if months is None else months
        self.transform = transform
        band_indices = list(range(len(self.bands))) if band_indices is None else band_indices
        # Read each sensor's selected channels in storage order, then permute to the request.
        self._reads: list[tuple[str, list[int]]] = []
        read_order: list[int] = []
        offset = 0
        for sensor, (_, shape, _) in SENSOR_ARRAYS.items():
            channels = [i - offset for i in band_indices if offset <= i < offset + shape[1]]
            if channels:
                self._reads.append((sensor, channels))
                read_order += [c + offset for c in channels]
            offset += shape[1]
        self._permutation = [read_order.index(i) for i in band_indices]
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self) -> dict:
        """Drop open memory maps so workers reopen them instead of copying the split."""
        return {**self.__dict__, "_arrays": {}}

    def _array(self, split: str, name: str) -> np.ndarray:
        key = f"{split}_{name}"
        if key not in self._arrays:
            if self.root is None:
                raise IndexError("PLANTEUR instance holds no split.")
            self._arrays[key] = np.load(self.root / f"{key}.npy", mmap_mode="r")
        return self._arrays[key]

    def __len__(self) -> int:
        """Return the number of loaded patches."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return ``image`` (float32, raw values) and ``mask`` (int64 HW, 255 = void)."""
        split, row = self.rows[index]
        stacks = [
            torch.from_numpy(
                self._array(split, sensor)[row, self.months][:, channels].astype(np.float32)
            )
            for sensor, channels in self._reads
        ]
        image = torch.cat(stacks, dim=1)[:, self._permutation]
        sample = {
            "image": image[0] if len(self.months) == 1 else image,
            "mask": torch.from_numpy(self._array(split, "mask")[row].astype(np.int64)),
        }
        return sample if self.transform is None else self.transform(sample)

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data") / "planteur"``, shared by every fold."""
        return Path("data") / "planteur"

    def _split_rows(self, split: str) -> list[tuple[str, int]]:
        """Return the converted-array rows of *split*."""
        size = np.load(self.data_root() / f"{split}_mask.npy", mmap_mode="r").shape[0]
        return [(split, row) for row in range(size)]

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[dict], dict] | None = None,
        time_steps: int | None = None,
    ) -> Self:
        """Return a new instance holding *split*, with *time_steps* evenly spaced months."""
        del partition
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        steps = 1 if time_steps is None else int(time_steps)
        if not 1 <= steps <= NUM_MONTHS:
            raise ValueError(
                f"PLANTEUR time_steps must be between 1 and {NUM_MONTHS}, got {time_steps}."
            )
        path = self.data_root() / f"{split}_mask.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"PLANTEUR split not found: {path}. Run `geo-eval-harness download planteur` first."
            )
        indices = [self.bands.index(spec) for spec in self.select_band_specs(bands)]
        months = [int(i * NUM_MONTHS / steps) for i in range(steps)]
        return type(self)(
            self.data_root(),
            self._split_rows(split),
            band_indices=indices,
            months=months,
            transform=transform,
        )


class _PLANTEURLOIO(PLANTEUR):
    """PLANTEUR leave-one-island-out fold: test is every patch of :attr:`held_out`.

    Train and val split the other territories (about 88/12) and share 0.1° blocks, so val
    scores are optimistic; test never shares a block with them. Saint-Martin stays in train.
    """

    held_out: ClassVar[str]

    def _split_rows(self, split: str) -> list[tuple[str, int]]:
        """Return the converted-array rows of this fold's *split*, in ``loio.json`` order."""
        root = self.data_root()
        ids = json.loads((root / "loio.json").read_text())["folds"][self.held_out][split]
        positions = _split_positions(root)
        return [positions[patch_id] for patch_id in ids]


class PLANTEURLOIOGuadeloupe(_PLANTEURLOIO):
    """PLANTEUR with Guadeloupe held out."""

    name = "planteur_loio_guadeloupe"
    held_out = "Guadeloupe"
    split_sizes: ClassVar[dict[str, int]] = {"train": 1750, "val": 265, "test": 735}


class PLANTEURLOIOMartinique(_PLANTEURLOIO):
    """PLANTEUR with Martinique held out."""

    name = "planteur_loio_martinique"
    held_out = "Martinique"
    split_sizes: ClassVar[dict[str, int]] = {"train": 1950, "val": 250, "test": 550}


class PLANTEURLOIOGuyane(_PLANTEURLOIO):
    """PLANTEUR with Guyane held out."""

    name = "planteur_loio_guyane"
    held_out = "Guyane"
    split_sizes: ClassVar[dict[str, int]] = {"train": 1960, "val": 284, "test": 506}


class PLANTEURLOIOReunion(_PLANTEURLOIO):
    """PLANTEUR with Réunion held out."""

    name = "planteur_loio_reunion"
    held_out = "Réunion"
    split_sizes: ClassVar[dict[str, int]] = {"train": 1720, "val": 217, "test": 813}


class PLANTEURLOIOMayotte(_PLANTEURLOIO):
    """PLANTEUR with Mayotte held out."""

    name = "planteur_loio_mayotte"
    held_out = "Mayotte"
    split_sizes: ClassVar[dict[str, int]] = {"train": 2296, "val": 314, "test": 140}
