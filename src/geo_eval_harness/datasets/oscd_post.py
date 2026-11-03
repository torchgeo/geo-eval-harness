"""OSCD post-change Sentinel-2 L1C image segmented against the urban change map."""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchgeo.datasets.utils import Sample

from .base import BandSpec, BenchDataset

RELEASE = "Onera Satellite Change Detection dataset"
#: Official OSCD training cities, in the order of the release's ``train.txt``.
OFFICIAL_TRAIN_CITIES: tuple[str, ...] = (
    "aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes", "saclay_e",
    "abudhabi", "cupertino", "pisa", "beihai", "hongkong", "beirut", "mumbai",
)  # fmt: skip
#: Official OSCD test cities, in the order of the release's ``test.txt``.
OFFICIAL_TEST_CITIES: tuple[str, ...] = (
    "brasilia", "montpellier", "norcia", "rio", "saclay_w",
    "valencia", "dubai", "lasvegas", "milano", "chongqing",
)  # fmt: skip
#: ``sorted(random.Random(0).sample(sorted(OFFICIAL_TRAIN_CITIES), 3))``.
VAL_CITIES: tuple[str, ...] = ("cupertino", "rennes", "saclay_e")


def read_window(path: Path, window: Window | None = None) -> np.ndarray:
    """Read band 1, zero-filling any part of *window* outside the raster."""
    # The resampled ``imgs_2_rect`` rasters carry no CRS or transform.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as raster:
            return raster.read(1, window=window, boundless=window is not None, fill_value=0)


def _tile_starts(length: int, size: int) -> list[tuple[int, int]]:
    """Return ``(grid start, read start)`` per tile; the last tile ends at the border."""
    return [(start, max(min(start, length - size), 0)) for start in range(0, length, size)]


class OSCDTileDataset(Dataset):
    """Read one post-image tile and its change mask per index."""

    def __init__(
        self,
        root: Path,
        tiles: list[tuple[str, int, int, int, int]],
        bands: list[str],
        transform: Callable[[Sample], Sample] | None,
    ) -> None:
        self.root = root
        self.tiles = tiles
        self.bands = bands
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> Sample:
        city, top, left, overlap_rows, overlap_cols = self.tiles[index]
        size = OSCDPost.tile_size
        window = Window(left, top, size, size)
        image = np.stack(
            [
                read_window(OSCDPost.image_dir(self.root, city) / f"{b}.tif", window)
                for b in self.bands
            ]
        )
        label_path = OSCDPost.label_path(self.root, city)
        native = read_window(label_path, window)
        if native.max() > 2:
            raise ValueError(f"OSCD: native labels outside {{1, 2}} in {label_path}")
        # Native 1/2 are no change/change; 0 only appears as padding.
        mask = np.where(native == 0, 255, native.astype(np.int64) - 1)
        # Pixels already covered by the previous tile in the row or column are ignored.
        mask[:overlap_rows] = 255
        mask[:, :overlap_cols] = 255
        sample = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask),
        }
        return self.transform(sample) if self.transform is not None else sample


class OSCDPost(BenchDataset):
    """OSCD urban change detection, posed as single-image segmentation of the post image.

    Source: the Onera Satellite Change Detection dataset (Caye Daudt et al., IGARSS 2018), as
    downloaded by torchgeo's ``OSCD``. Each of the 24 cities has a Sentinel-2 **Level-1C** image
    pair; this benchmark uses only the later image (``imgs_2_rect``: all 13 bands resampled by the
    authors to a common 10 m grid) and segments the urban change map against it. A model sees
    only the post-change image, so scores are not comparable with bi-temporal OSCD results.

    Splits are by city: ``test`` is the 10 official test cities, ``val`` three of the 14 official
    training cities (``VAL_CITIES``, drawn with ``random.Random(0)``) and ``train`` the other 11.
    Cities are cut row-major into 256 px tiles (2.56 km). Edge tiles are shifted inward to end
    at the image border, so they show real imagery instead of padding; the part overlapping the
    previous tile is masked 255 and every pixel is scored exactly once. Only a city smaller than
    a tile is zero-padded (mask 255). The dataset is tiny (119 / 31 / 69 tiles) and noisy.

    Labels come from ``<city>-cm.tif`` (1 = no change, 2 = change), remapped to 0 / 1; the
    ``cm.png`` previews are not used because the aguasclaras one is anti-aliased.

    Images are raw L1C DN (reflectance x 10^4) as float32. ``BandSpec`` has no processing-level
    field, so pick an L1C model stem explicitly (e.g. ``terrafm/base_s2_l1c``). Change labels are
    CC BY-NC-SA; imagery contains modified Copernicus Sentinel data 2015-2018.
    """

    name = "oscd-post"
    task = "segmentation"
    num_classes = 2
    multilabel = False
    supports_partitions = False
    rgb_bands: ClassVar[list[str]] = ["b04", "b03", "b02"]
    tile_size: ClassVar[int] = 256
    split_cities: ClassVar[dict[str, tuple[str, ...]]] = {
        "train": tuple(sorted(set(OFFICIAL_TRAIN_CITIES) - set(VAL_CITIES))),
        "val": VAL_CITIES,
        "test": tuple(sorted(OFFICIAL_TEST_CITIES)),
    }
    split_sizes: ClassVar[dict[str, int]] = {"train": 119, "val": 31, "test": 69}

    # Mean/std/min/max over every pixel of the 11 train-city rasters, i.e. every train pixel
    # scored once (5,087,269 per band).
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("s2", "b01", "B01", mean=1571.8888, std=283.9952, min=0, max=3988, wavelength_um=0.443),
        BandSpec("s2", "b02", "B02", mean=1369.8927, std=425.382, min=111, max=27108, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=1295.0977, std=575.419, min=474, max=28000, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=1336.7313, std=827.5685, min=274, max=28000, wavelength_um=0.665),
        BandSpec("s2", "b05", "B05", mean=1457.9584, std=793.6916, min=0, max=29449, wavelength_um=0.705),
        BandSpec("s2", "b06", "B06", mean=1820.1416, std=813.0226, min=0, max=25552, wavelength_um=0.74),
        BandSpec("s2", "b07", "B07", mean=2014.163, std=882.0069, min=0, max=29756, wavelength_um=0.783),
        BandSpec("s2", "b08", "B08", mean=1937.9653, std=902.4788, min=148, max=28000, wavelength_um=0.842),
        BandSpec("s2", "b8a", "B8A", mean=2134.839, std=946.8174, min=0, max=33758, wavelength_um=0.865),
        BandSpec("s2", "b09", "B09", mean=663.5543, std=373.0631, min=0, max=2699, wavelength_um=0.945),
        BandSpec("s2", "b10", "B10", mean=13.6214, std=7.594, min=0, max=175, wavelength_um=1.375),
        BandSpec("s2", "b11", "B11", mean=1981.9397, std=1124.211, min=0, max=24927, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=1472.574, std=980.3392, min=0, max=24012, wavelength_um=2.19),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return the torchgeo ``OSCD`` root."""
        return Path("data") / "oscd-post"

    @staticmethod
    def image_dir(root: Path, city: str) -> Path:
        """Return the directory holding a city's 10 m post-image bands."""
        return root / f"{RELEASE} - Images" / city / "imgs_2_rect"

    @classmethod
    def label_path(cls, root: Path, city: str) -> Path:
        """Return a city's change map."""
        kind = "Test" if city in cls.split_cities["test"] else "Train"
        return root / f"{RELEASE} - {kind} Labels" / city / "cm" / f"{city}-cm.tif"

    @classmethod
    def split_tiles(cls, root: Path, split: str) -> list[tuple[str, int, int, int, int]]:
        """Return ``(city, top, left, overlap_rows, overlap_cols)`` in city then row-major order."""
        tiles = []
        for city in cls.split_cities[split]:
            paths = [cls.label_path(root, city)]
            paths += [cls.image_dir(root, city) / f"{b.source_name}.tif" for b in cls.bands]
            if missing := [path for path in paths if not path.is_file()]:
                raise FileNotFoundError(
                    f"OSCD: missing {missing[0]}. Run: geo-eval-harness download oscd-post"
                )
            height, width = read_window(paths[0]).shape
            size = cls.tile_size
            tiles += [
                (city, y, x, grid_y - y, grid_x - x)
                for grid_y, y in _tile_starts(height, size)
                for grid_x, x in _tile_starts(width, size)
            ]
        if len(tiles) != cls.split_sizes[split]:
            raise ValueError(
                f"OSCD: {split} has {len(tiles)} tiles, expected {cls.split_sizes[split]}"
            )
        return tiles

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Load raw L1C DN tiles as float32 and change masks as int64 (255 = ignore)."""
        if partition != "default":
            raise ValueError("OSCD supports only the default partition")
        names = [spec.source_name for spec in self.select_band_specs(bands)]
        root = self.data_root()
        return OSCDTileDataset(root, self.split_tiles(root, split), names, transform)
