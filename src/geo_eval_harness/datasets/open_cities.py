"""Open Cities AI Challenge building segmentation, from GFDRR Labs on source.coop.

The upstream drone scenes stay byte-identical on disk. A committed chip index
(``open_cities_index.csv.gz``) fixes 512 px chips on each scene's native pixel grid and assigns
them to train/val/test by 500 m geographic block. Masks are rasterised once, on the same
native grid, into one sparse GeoTIFF per scene. Chips are read by window; nothing is resampled.
"""

import logging
import os
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import Window
from torch.utils.data import Dataset

from ._transforms import select_bands
from .base import BandSpec, BenchDataset

logger = logging.getLogger(__name__)

BASE_URL = "https://data.source.coop/open-cities/ai-challenge/train_tier_1"
# source.coop answers 403 to Python's default urllib User-Agent.
USER_AGENT = "geo-eval-harness"
CHIP_SIZE = 512
IGNORE_INDEX = 255
# The alpha band is JPEG-compressed, so nodata edges carry alpha 1-127; four Zanzibar
# scenes also store nodata as black RGB under opaque alpha.
ALPHA_MIN = 128
BLACK_MAX = 10
INDEX_FILE = "open_cities_index.csv.gz"
CITY_NAME: dict[str, str] = {
    "acc": "Accra",
    "dar": "Dar es Salaam",
    "kam": "Kampala",
    "mon": "Monrovia",
    "nia": "Niamey",
    "ptn": "Pointe-Noire",
    "znz": "Zanzibar",
}


def is_valid(rgba: np.ndarray) -> np.ndarray:
    """Return the valid-data mask of a ``(4, H, W)`` uint8 RGBA array."""
    return (rgba[3] >= ALPHA_MIN) & (rgba[:3].max(0) > BLACK_MAX)


def load_index() -> pd.DataFrame:
    """Return the committed chip index shipped with the package."""
    path = resources.files(__package__).joinpath(INDEX_FILE)
    with path.open("rb") as stream:
        return pd.read_csv(stream, comment="#", compression="gzip")


def load_labels(path: Path, crs: CRS) -> np.ndarray:
    """Read a label GeoJSON, reproject it to ``crs``, and repair invalid polygons.

    Repairs can yield line or point fragments; only the polygonal parts are kept.
    """
    geoms = shapely.get_parts(shapely.make_valid(gpd.read_file(path).to_crs(crs).geometry.values))
    return geoms[shapely.get_type_id(geoms) == shapely.GeometryType.POLYGON]


def upstream_url(relative: str) -> str:
    """Map a local ``<city>/<scene>.tif|.geojson`` path to its source.coop URL."""
    city, filename = relative.split("/")
    scene, suffix = filename.split(".")
    folder = scene if suffix == "tif" else f"{scene}-labels"
    return f"{BASE_URL}/{city}/{folder}/{filename}"


def build_scene_mask(
    image_path: Path, label_path: Path, windows: pd.DataFrame, chip_size: int = CHIP_SIZE
) -> Path:
    """Write ``<scene>_mask.tif`` next to the image, filling only the indexed windows.

    The mask shares the image's grid. Values: 0 background, 1 building, and
    :data:`IGNORE_INDEX` where :func:`is_valid` is false. Unwritten tiles stay sparse and
    read back as :data:`IGNORE_INDEX`.

    Args:
        image_path: Upstream RGBA scene GeoTIFF.
        label_path: The scene's label GeoJSON.
        windows: Index rows with ``col_off`` and ``row_off`` for this scene.
        chip_size: Window size in pixels; must be a multiple of 16.

    Returns:
        Path of the written mask.
    """
    out = image_path.with_name(f"{image_path.stem}_mask.tif")
    tmp = out.with_suffix(".tif.tmp")
    with rasterio.open(image_path) as src:
        geoms = load_labels(label_path, src.crs)
        tree = shapely.STRtree(geoms)
        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": IGNORE_INDEX,
            "tiled": True,
            "blockxsize": chip_size,
            "blockysize": chip_size,
            "compress": "deflate",
            "sparse_ok": True,
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            for col_off, row_off in zip(windows.col_off, windows.row_off, strict=True):
                window = Window(col_off, row_off, chip_size, chip_size)
                transform = src.window_transform(window)
                hits = tree.query(shapely.box(*rasterio.windows.bounds(window, src.transform)))
                mask = np.zeros((chip_size, chip_size), dtype=np.uint8)
                if len(hits):
                    mask = rasterize(
                        ((geom, 1) for geom in geoms[hits]),
                        out_shape=mask.shape,
                        transform=transform,
                        dtype=np.uint8,
                    )
                mask[~is_valid(src.read(window=window))] = IGNORE_INDEX
                dst.write(mask, 1, window=window)
    tmp.replace(out)
    return out


class _OpenCitiesSplit(Dataset):
    """Native-resolution chips read by window from the scene and mask GeoTIFFs."""

    def __init__(
        self,
        root: Path,
        index: pd.DataFrame,
        transform: Callable[[dict], dict] | None,
        chip_size: int = CHIP_SIZE,
    ) -> None:
        self.root = root
        self.scenes = index.city.str.cat(index.scene, sep="/").to_numpy()
        self.offsets = index[["col_off", "row_off"]].to_numpy()
        self.transform = transform
        self.chip_size = chip_size
        self._handles: dict[str, rasterio.DatasetReader] = {}
        self._pid = os.getpid()
        for scene in np.unique(self.scenes):
            for path in (root / f"{scene}.tif", root / f"{scene}_mask.tif"):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Open Cities file not found: {path}. "
                        "Run `geo-eval-harness download open_cities` first."
                    )

    def __len__(self) -> int:
        return len(self.scenes)

    def _open(self, path: Path) -> rasterio.DatasetReader:
        # Handles must not cross a fork: each DataLoader worker opens its own.
        if os.getpid() != self._pid:
            self._handles, self._pid = {}, os.getpid()
        key = str(path)
        if key not in self._handles:
            self._handles[key] = rasterio.open(path)
        return self._handles[key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scene = self.scenes[index]
        col_off, row_off = self.offsets[index]
        window = Window(col_off, row_off, self.chip_size, self.chip_size)
        image = self._open(self.root / f"{scene}.tif").read((1, 2, 3), window=window)
        mask = self._open(self.root / f"{scene}_mask.tif").read(1, window=window)
        sample = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
        }
        return sample if self.transform is None else self.transform(sample)


class OpenCities(BenchDataset):
    """Building footprint segmentation (2 classes) on drone imagery of 7 African cities.

    Open Cities AI Challenge (GFDRR Labs 2020, DOI 10.34911/rdnt.f94cxb), tier 1 only:
    31 scenes over Accra, Dar es Salaam, Kampala, Monrovia, Niamey, Pointe-Noire and
    Zanzibar, labelled with OpenStreetMap / Zanzibar Mapping Initiative footprints.

    Samples are 512x512 RGB chips cut on each scene's native pixel grid, so the ground
    sampling distance varies from 0.02 m to 0.20 m (a chip spans 10-100 m). Resizing is left
    to the model's transforms. A chip is kept when at least half of it is valid imagery.
    Train/val/test (70/10/20 per city) are assigned per 500 m geographic block and fixed in
    the packaged ``open_cities_index.csv.gz``. Zanzibar blocks with under 1% building cover are
    dropped, as are chips duplicated by overlapping scenes. See
    ``docs/plans/2026-10-09-open-cities-building-segmentation.md`` for the label audit.

    Mask values: 0 background, 1 building, 255 nodata (transparent or black image pixels).
    Caveats: Dar es Salaam tier 1 misses about 5% of buildings; Monrovia scenes ``207cc7``
    and ``401175`` are stored in UTM 36N, so their ground pixels are not square.
    Source: https://source.coop/open-cities/ai-challenge. Labels ODbL 1.0; imagery
    CC BY 4.0 or ODbL depending on the scene.
    """

    name = "open_cities"
    task = "segmentation"
    num_classes = 2
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 76818, "val": 11236, "test": 22408}
    supports_partitions = False

    # Train-split statistics from scripts/compute_band_statistics.py --ignore-index 255.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", "red", "R", mean=126.5779, std=49.1320, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=122.1465, std=45.8053, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=100.8429, std=52.7224, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/open_cities")``."""
        return Path("data/open_cities")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ) -> Dataset:
        """Return the split with ``image`` (float32 CHW, 0-255) and ``mask`` (int64 HW)."""
        del partition
        if split not in self.split_sizes:
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        index = load_index()
        return _OpenCitiesSplit(
            self.data_root(),
            index[index.split == split].reset_index(drop=True),
            select_bands(indices, len(self.bands), transform),
        )
