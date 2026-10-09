"""Plot the committed Open Cities train/val/test geography.

Writes to ``figures/open_cities/`` (git-ignored; the figures stay local):

- ``split_map.png``: one panel per city, 500 m blocks coloured by split over a desaturated
  imagery thumbnail; dropped Zanzibar blocks dotted;
- ``split_map_<city>.png``: one larger map per city with chips per block and scene ids;
- ``chip_gallery.png``: native 512 px chips per city and split with label outlines;
- ``blocks.geojson``: block polygons (EPSG:4326) with city, split, chip count and building
  fraction, including the dropped Zanzibar blocks (``split = "dropped"``).

Reads the packaged index and the scenes under ``--data-dir`` if present, otherwise the
upstream files over HTTP (overviews only for the maps).

Usage::

    python scripts/plot_open_cities_splits.py --data-dir data/open_cities
"""

import argparse
import gzip
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.patches import Patch, Rectangle
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from geo_eval_harness.datasets import open_cities
from geo_eval_harness.datasets.open_cities import CHIP_SIZE, CITY_NAME, is_valid

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

os.environ.setdefault("GDAL_HTTP_USERAGENT", open_cities.USER_AGENT)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

BLOCK_M = 500.0
CITY_EPSG = {
    "acc": 32630,
    "dar": 32737,
    "kam": 32636,
    "mon": 32629,
    "nia": 32631,
    "ptn": 32732,
    "znz": 32737,
}
SPLITS = ("train", "val", "test")
COLOR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
INK, MUTED, SURFACE = "#2b2b29", "#8a8a85", "#fcfcfb"
# Thumbnail resolution (m/px) per city, sized to the city's extent.
THUMB_RES = {"znz": 12, "dar": 8, "acc": 4, "kam": 2, "mon": 2, "nia": 2, "ptn": 2}
PANEL_ORDER = ["znz", "dar", "acc", "kam", "mon", "ptn", "nia"]


def source(data_dir: Path, city: str, scene: str, ext: str) -> str:
    """Return the local path of an upstream file if downloaded, else its ``/vsicurl/`` URL."""
    local = data_dir / city / f"{scene}.{ext}"
    if local.is_file():
        return str(local)
    return "/vsicurl/" + open_cities.upstream_url(f"{city}/{scene}.{ext}")


def read_dropped_blocks(path: Path) -> pd.Series:
    """Read the dropped Zanzibar blocks and their building fraction from the index header."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("# znz_dropped_blocks:"):
                pairs = [item.split("=") for item in line.split(":", 1)[1].split()]
                return pd.Series({b: float(f) for b, f in pairs}, name="building_frac")
    raise ValueError(f"{path} has no znz_dropped_blocks header line")


def block_polygons(blocks: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return 500 m block squares in EPSG:4326 for rows with ``city`` and ``block`` columns."""
    parts = []
    for city, group in blocks.groupby("city"):
        ij = group.block.str.rsplit("_", n=2, expand=True).iloc[:, 1:].astype(int)
        i, j = ij[1].to_numpy(), ij[2].to_numpy()
        geom = shapely.box(i * BLOCK_M, j * BLOCK_M, (i + 1) * BLOCK_M, (j + 1) * BLOCK_M)
        parts.append(gpd.GeoDataFrame(group, geometry=geom, crs=CITY_EPSG[city]).to_crs(4326))
    return pd.concat(parts, ignore_index=True)


def blocks_table(index: pd.DataFrame, dropped: pd.Series) -> gpd.GeoDataFrame:
    """Return all blocks (kept and dropped) with their split, chip count and scenes."""
    kept = (
        index.groupby("block")
        .agg(
            city=("city", "first"),
            split=("split", "first"),
            n_chips=("chip_id", "size"),
            building_frac=("building_frac", "mean"),
            scene=("scene", lambda s: ",".join(sorted(s.unique()))),
        )
        .reset_index()
    )
    removed = pd.DataFrame(
        {
            "block": dropped.index,
            "city": "znz",
            "split": "dropped",
            "n_chips": 0,
            "building_frac": dropped.to_numpy(),
            "scene": "",
        }
    )
    table = pd.concat([kept, removed], ignore_index=True)
    table["building_frac"] = table.building_frac.round(5)
    return block_polygons(table)


def thumbnail(path: str, city: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return a desaturated RGBA thumbnail warped to the city's UTM, with its extent."""
    res = THUMB_RES[city]
    with (
        rasterio.open(path) as ds,
        WarpedVRT(ds, crs=f"EPSG:{CITY_EPSG[city]}", resampling=Resampling.average) as vrt,
    ):
        width = int((vrt.bounds.right - vrt.bounds.left) / res)
        height = int((vrt.bounds.top - vrt.bounds.bottom) / res)
        rgba = vrt.read(out_shape=(4, height, width), resampling=Resampling.average)
        extent = (vrt.bounds.left, vrt.bounds.right, vrt.bounds.bottom, vrt.bounds.top)
    rgba = np.moveaxis(rgba, 0, -1) / 255.0
    gray = rgba[..., :3].mean(-1, keepdims=True)
    rgba[..., :3] = 0.4 * gray + 0.6 * rgba[..., :3]
    return rgba, extent


def footprints(data_dir: Path, scenes: list[tuple[str, str]]) -> gpd.GeoDataFrame:
    """Return each scene's valid-data footprint (from its 1/32 overview) in EPSG:4326."""
    rows = []
    for city, scene in scenes:
        with rasterio.open(source(data_dir, city, scene, "tif")) as ds:
            h, w = ds.height // 32, ds.width // 32
            rgba = ds.read(out_shape=(4, h, w), resampling=Resampling.nearest)
            transform = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
            shapes = rasterio.features.shapes(
                is_valid(rgba).astype(np.uint8), mask=is_valid(rgba), transform=transform
            )
            geom = shapely.union_all([shapely.geometry.shape(s) for s, _ in shapes])
            geom = gpd.GeoSeries([geom.buffer(0)], crs=ds.crs).to_crs(4326).iloc[0]
        rows.append({"city": city, "scene": scene, "geometry": geom})
    return gpd.GeoDataFrame(rows, crs=4326)


def draw_city(ax, city, index, dropped, thumbs, fps, *, big=False):  # noqa: PLR0913
    """Draw one city's blocks coloured by split over its imagery thumbnails."""
    for (c, _), (img, ext) in thumbs.items():
        if c == city:
            ax.imshow(img, extent=ext, interpolation="bilinear")
    group = index[index.city == city]
    blocks = group.groupby("block").agg(split=("split", "first"), n=("split", "size"))
    xs, ys = [], []
    for name, row in blocks.iterrows():
        _, i, j = name.rsplit("_", 2)
        x, y = int(i) * BLOCK_M, int(j) * BLOCK_M
        xs.append(x)
        ys.append(y)
        ax.add_patch(
            Rectangle(
                (x, y),
                BLOCK_M,
                BLOCK_M,
                facecolor=COLOR[row.split],
                alpha=0.40,
                edgecolor=COLOR[row.split],
                lw=1.3,
            )
        )
        if big:
            ax.text(
                x + BLOCK_M / 2,
                y + BLOCK_M / 2,
                f"{row.n}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK,
            )
    for name in dropped.index if city == "znz" else []:
        _, i, j = name.rsplit("_", 2)
        x, y = int(i) * BLOCK_M, int(j) * BLOCK_M
        xs.append(x)
        ys.append(y)
        ax.add_patch(
            Rectangle((x, y), BLOCK_M, BLOCK_M, fill=False, edgecolor=MUTED, lw=0.6, ls=":")
        )
    fp = fps[fps.city == city].to_crs(CITY_EPSG[city])
    fp.boundary.plot(ax=ax, color=INK, lw=0.6, ls="--")
    if big:
        for _, row in fp.iterrows():
            p = row.geometry.representative_point()
            ax.text(
                p.x,
                p.y,
                row.scene,
                fontsize=7,
                color="white",
                bbox={"facecolor": INK, "alpha": 0.7, "pad": 1, "lw": 0},
            )
    ax.set_xlim(min(xs) - BLOCK_M, max(xs) + 2 * BLOCK_M)
    ax.set_ylim(min(ys) - BLOCK_M, max(ys) + 2 * BLOCK_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d9d8d2")
    counts = group.split.value_counts()
    ax.set_title(
        f"{CITY_NAME[city]}: train {counts.get('train', 0):,} · val {counts.get('val', 0):,} · "
        f"test {counts.get('test', 0):,} chips ({len(blocks)} blocks)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = 1000 if (x1 - x0) > 3000 else 500
    start = x0 + 0.05 * (x1 - x0)
    ax.plot([start, start + length], [y0 + 0.04 * (y1 - y0)] * 2, color=INK, lw=2)
    ax.text(start, y0 + 0.06 * (y1 - y0), f"{length} m", fontsize=8, color=INK)


def legend_handles(index: pd.DataFrame) -> list[Patch]:
    """Return legend entries naming each split with its chip count."""
    handles = [
        Patch(
            facecolor=COLOR[s],
            alpha=0.6,
            edgecolor=COLOR[s],
            label=f"{s} ({(index.split == s).sum():,} chips)",
        )
        for s in SPLITS
    ]
    handles.append(
        Patch(
            fill=False, edgecolor=MUTED, ls=":", label="Zanzibar block dropped (<1% building cover)"
        )
    )
    handles.append(Patch(fill=False, edgecolor=INK, ls="--", label="scene footprint"))
    return handles


def chip_gallery(index: pd.DataFrame, data_dir: Path, out: Path, per_split: int = 2) -> None:
    """Plot native 512 px chips with label outlines, ``per_split`` per split and city."""
    rng = np.random.default_rng(0)
    cities = sorted(index.city.unique())
    fig, axes = plt.subplots(
        len(cities),
        3 * per_split,
        figsize=(3 * per_split * 2.6, len(cities) * 2.8),
        facecolor=SURFACE,
    )
    for r, city in enumerate(cities):
        for c, split in enumerate(SPLITS):
            group = index[(index.city == city) & (index.split == split) & (index.building_frac > 0)]
            pick = group.iloc[
                rng.choice(len(group), size=min(per_split, len(group)), replace=False)
            ]
            for k in range(per_split):
                ax = axes[r, c * per_split + k]
                ax.set_xticks([])
                ax.set_yticks([])
                if k >= len(pick):
                    ax.axis("off")
                    continue
                row = pick.iloc[k]
                with rasterio.open(source(data_dir, city, row.scene, "tif")) as ds:
                    window = Window(row.col_off, row.row_off, CHIP_SIZE, CHIP_SIZE)
                    img = ds.read(window=window)
                    transform = ds.window_transform(window)
                    bounds = rasterio.windows.bounds(window, ds.transform)
                    crs = ds.crs
                bbox = gpd.GeoSeries([shapely.box(*bounds)], crs=crs).to_crs(4326).total_bounds
                labels = gpd.read_file(
                    source(data_dir, city, row.scene, "geojson"), bbox=tuple(bbox)
                ).to_crs(crs)
                mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
                if len(labels):
                    mask = rasterize(
                        ((g, 1) for g in shapely.make_valid(labels.geometry.values)),
                        out_shape=mask.shape,
                        transform=transform,
                    )
                ax.imshow(np.moveaxis(img[:3], 0, -1))
                ax.contour(mask, levels=[0.5], colors=["#e34948"], linewidths=0.7)
                nodata = ~is_valid(img)
                ax.imshow(
                    np.ma.masked_where(~nodata, nodata), cmap="Greys", vmin=0, vmax=1, alpha=0.5
                )
                for spine in ax.spines.values():
                    spine.set_color(COLOR[split])
                    spine.set_linewidth(3)
                ax.set_title(
                    f"{split} · {row.scene} · {row.gsd_m * CHIP_SIZE:.0f} m",
                    fontsize=7.5,
                    color=INK,
                )
        axes[r, 0].set_ylabel(CITY_NAME[city], fontsize=9, color=INK)
    fig.suptitle(
        "Native-resolution 512 px chips (no resampling); red = label outline, frame colour = split",
        fontsize=11,
        color=INK,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)


def plot_splits(index: pd.DataFrame, dropped: pd.Series, out_dir: Path, data_dir: Path) -> None:
    """Write the split maps, chip gallery and ``blocks.geojson`` for a chip index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_table(index, dropped).to_file(out_dir / "blocks.geojson", driver="GeoJSON")

    scenes = sorted(set(zip(index.city, index.scene, strict=True)))
    fps = footprints(data_dir, scenes)
    with ThreadPoolExecutor(8) as pool:
        images = pool.map(lambda s: thumbnail(source(data_dir, s[0], s[1], "tif"), s[0]), scenes)
        thumbs = dict(zip(scenes, images, strict=True))

    fig = plt.figure(figsize=(18, 13.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.3, 1, 1])
    cells = [grid[:, 0], grid[0, 1], grid[1, 1], grid[0, 2], grid[0, 3], grid[1, 2], grid[1, 3]]
    for cell, city in zip(cells, PANEL_ORDER, strict=True):
        draw_city(fig.add_subplot(cell), city, index, dropped, thumbs, fps)
    fig.legend(
        handles=legend_handles(index),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=5,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle(
        "Open Cities tier 1: 500 m blocks assigned to train / val / test, 512 px native chips",
        fontsize=13,
        color=INK,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "split_map.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    for city in PANEL_ORDER:
        fig, ax = plt.subplots(figsize=(11, 11), facecolor=SURFACE)
        draw_city(ax, city, index, dropped, thumbs, fps, big=True)
        ax.legend(
            handles=legend_handles(index[index.city == city]),
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=False,
            fontsize=9,
        )
        fig.text(
            0.01,
            0.005,
            "Numbers in blocks = chips per block. Dashed = scene footprint (label = scene id).",
            fontsize=8,
            color=MUTED,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"split_map_{city}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)

    chip_gallery(index, data_dir, out_dir / "chip_gallery.png")
    logger.info("Wrote split maps to %s", out_dir)


def main() -> None:
    """Plot the splits of the packaged index."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/open_cities"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/open_cities"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    index_path = Path(open_cities.__file__).with_name(open_cities.INDEX_FILE)
    plot_splits(
        open_cities.load_index(), read_dropped_blocks(index_path), args.out_dir, args.data_dir
    )


if __name__ == "__main__":
    main()
