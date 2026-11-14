"""Small, disjoint on-disk datasets read by the production dataset adapters."""

import hashlib
import io
import json
import shutil
import struct
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from geo_eval_harness.datasets import get_bench_dataset_class


def require_dataset_data(name: str) -> None:
    """Skip absent optional real data, never an incompatible existing cache."""
    if name.startswith("m-"):
        paths = [
            Path("data/classification_v1.0") / name,
            Path("data/classification_v1.0_wds") / name,
        ]
    elif name in ("eurosat", "eurosat-spatial"):
        paths = [Path("data/eurosat")]
    elif name in (
        "resisc45",
        "ucmerced",
        "hotosm_buildings",
        "openearthmap-sar",
        "open_cities",
        "loveda",
        "levir-cd-post",
        "bright-post-sar",
        "oscd-post",
    ):
        paths = [Path("data") / name]
    elif name.startswith("sen1floods11"):
        paths = [Path("data/sen1floods11")]
    elif name.startswith("planteur"):
        paths = [Path("data/planteur")]
    else:
        paths = [Path("data/geobenchv2") / name]
    if not any(path.exists() for path in paths):
        pytest.skip(f"{name} data not supplied; expected one of {paths}")


def write_classification_files(
    root: Path,
    dataset_name: str,
    labels: tuple[int | list[int], ...],
    *,
    all_bands: bool = False,
) -> Path:
    """Write separable V1 JSON-HDF5 samples with distinct IDs and pixels in each split."""
    directory = root / "data" / "classification_v1.0" / dataset_name
    directory.mkdir(parents=True)
    bench = get_bench_dataset_class(dataset_name)()
    specs = bench.select_band_specs(None if all_bands else tuple(bench.rgb_bands))
    partition: dict[str, list[str]] = {}
    for split_index, (split, per_class) in enumerate((("train", 12), ("valid", 4), ("test", 4))):
        partition[split] = []
        for class_index, label in enumerate(labels):
            for index in range(per_class):
                sample_id = f"{split}-{class_index}-{index}"
                partition[split].append(sample_id)
                with h5py.File(directory / f"{sample_id}.hdf5", "w") as sample:
                    for band in specs:
                        signal = 0.5 + 2.5 * class_index + 0.03 * split_index + 0.001 * index
                        sample.create_dataset(
                            band.source_name,
                            data=np.full((16, 16), band.mean + band.std * signal, np.float32),
                        )
                    sample.attrs["metadata_json"] = json.dumps(
                        {"label": label, "bands_order": [band.source_name for band in specs]}
                    )
    (directory / "default_partition.json").write_text(json.dumps(partition))
    return directory


def write_v1_shards(root: Path, dataset_name: str = "m-eurosat") -> Path:
    """Package the same samples in the download mirror's real JSON/NPZ tar format."""
    source = write_classification_files(root, dataset_name, (2, 7))
    target = root / "data" / "classification_v1.0_wds" / dataset_name
    target.mkdir(parents=True)
    (target / "default_partition.json").write_bytes(
        (source / "default_partition.json").read_bytes()
    )
    with tarfile.open(target / "shard_00000.tar", "w") as archive:
        for path in sorted(source.glob("*.hdf5")):
            with h5py.File(path) as sample:
                bands = io.BytesIO()
                np.savez(bands, **{name: sample[name][:] for name in sample})
                metadata = sample.attrs["metadata_json"].encode()
            for suffix, payload in (("bands.npz", bands.getvalue()), ("meta.json", metadata)):
                member = tarfile.TarInfo(f"{path.stem}.{suffix}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return target


def geotiff_bytes(pixels: np.ndarray) -> bytes:
    """Encode channel-first pixels as a real, georeferenced in-memory GeoTIFF."""
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=pixels.shape[-2],
            width=pixels.shape[-1],
            count=pixels.shape[0],
            dtype=pixels.dtype,
            crs="EPSG:4326",
            transform=Affine(0.01, 0, 10, 0, -0.01, 45),
        ) as dataset:
            dataset.write(pixels)
        return memory.read()


def write_openearthmap_sar_files(root: Path) -> Path:
    """Write tiny, disjoint SAR/label pairs using the active split descriptors."""
    from geo_eval_harness.datasets.openearthmap_sar import OpenEarthMapSAR

    directory = root / "data" / "openearthmap-sar"
    size = OpenEarthMapSAR.raster_size
    y, x = np.indices((size, size))
    for split_index, split in enumerate(OpenEarthMapSAR.split_sizes):
        for index, name in enumerate(reversed(OpenEarthMapSAR.split_filenames(split))):
            mask = ((x // 2 + y // 4 + index) % 8 + 1).astype(np.uint8)
            image = (mask * 25 + split_index * 3 + index).astype(np.uint8)
            # A 2x2 block keeps ignored pixels at the origin after 2x nearest downsampling.
            mask[:2, :2] = 0
            for kind, pixels in (("sar_images", image), ("labels", mask)):
                path = directory / split / kind / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(geotiff_bytes(pixels[None]))
    return directory


def write_sen1floods11_files(root: Path) -> Path:
    """Write 16 x 16 chips and split CSVs in the Sen1Floods11 v1.1 layout.

    Each chip has label -1 at ``[:2, :2]``, S1 NaN at ``[2:4, :2]`` and S2 zero fill at
    ``[:2, 2:4]``; the last val chip is entirely no data.
    """
    from geo_eval_harness.datasets.sen1floods11 import (
        SPLIT_DIR,
        SPLIT_FILES,
        Sen1Floods11Base,
        chip_path,
    )

    directory = root / "data" / "sen1floods11"
    y, x = np.indices((16, 16))
    for split, csv_name in SPLIT_FILES.items():
        chips = [f"{split}_{i}" for i in range(Sen1Floods11Base.split_sizes[split])]
        for i, chip in enumerate(chips):
            water = (x // 4 + y // 4 + i) % 2
            label = water.astype(np.int16)
            label[:2, :2] = -1
            if split == "val" and chip == chips[-1]:
                label[:] = -1
            s1 = np.stack([-8 - 14 * water, -15 - 13 * water]).astype(np.float32)
            s1[:, 2:4, :2] = np.nan
            s2 = np.stack([2500 - 2000 * water + band for band in range(13)]).astype(np.int16)
            s2[:, :2, 2:4] = 0
            for layer, pixels in (("S1Hand", s1), ("S2Hand", s2), ("LabelHand", label[None])):
                path = directory / chip_path(chip, layer)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(geotiff_bytes(pixels))
        path = directory / SPLIT_DIR / csv_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\r\n" for c in chips), newline="")
    return directory


def sen1floods11_checksums(directory: Path) -> dict[str, str]:
    """Return a SHA-256 manifest of a written Sen1Floods11 directory."""
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def write_openearthmap_sar_archives(root: Path) -> tuple:
    """Package tiny release ZIPs, preserving production URLs and directory ownership."""
    import hashlib
    from dataclasses import replace

    from geo_eval_harness.download import SAR_ARCHIVES

    directory = write_openearthmap_sar_files(root)
    artifacts = []
    for artifact in SAR_ARCHIVES:
        path = root / artifact.filename
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for prefix in artifact.prefixes:
                for member in (directory / prefix).glob("*.tif"):
                    archive.write(member, member.relative_to(directory).as_posix())
            archive.writestr("train/rgb_images/TrainArea_001.tif", b"not SAR")
            archive.writestr("__MACOSX/._ValArea_001.tif", b"metadata")
            archive.writestr("val/.DS_Store", b"metadata")
        artifacts.append(
            replace(
                artifact,
                size=path.stat().st_size,
                md5=hashlib.md5(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(artifacts)


def write_loveda_files(root: Path) -> Path:
    """Write tiny RGB/mask PNG pairs for the active LoveDA memberships."""
    from geo_eval_harness.datasets.loveda import LoveDA

    directory = root / "data" / "loveda"
    mask = (np.arange(64).reshape(8, 8) % 8).astype(np.uint8)  # native 0 (no-data) at the origin
    image = np.stack([mask * 30 + band for band in range(3)], axis=-1)
    for split in LoveDA.split_sizes:
        for member in LoveDA.split_members(split):
            release, scene, name = member.split("/")
            for kind, pixels in (("images_png", image), ("masks_png", mask)):
                path = directory / release / scene / kind / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
    return directory


def write_loveda_archives(root: Path) -> dict[str, str]:
    """Package the tiny tiles as torchgeo's ``Train.zip``/``Val.zip``; return their MD5s."""
    directory = write_loveda_files(root / "tiles")
    md5s = {}
    for release in ("Train", "Val"):
        path = root / f"{release}.zip"
        shutil.make_archive(str(path.with_suffix("")), "zip", directory, release)
        md5s[release.lower()] = hashlib.md5(path.read_bytes()).hexdigest()
    return md5s


def write_levir_cd_post_files(root: Path) -> Path:
    """Write random pre/post images and masks (including stray 156/254) in torchgeo's layout."""
    from geo_eval_harness.datasets.levir_cd_post import LEVIRCDPost

    directory = root / "data" / "levir-cd-post"
    size = LEVIRCDPost.raster_size
    rng = np.random.default_rng(0)
    for kind in ("A", "B", "label"):
        (directory / kind).mkdir(parents=True, exist_ok=True)
    for split, count in LEVIRCDPost.image_counts.items():
        for index in range(1, count + 1):
            name = f"{split}_{index}.png"
            image = Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
            image.save(directory / "A" / name)
            image.save(directory / "B" / name)
            mask = rng.choice(np.array([0, 0, 156, 254, 255], dtype=np.uint8), (size, size))
            Image.fromarray(mask).save(directory / "label" / name)
    return directory


def write_bright_post_sar_files(root: Path) -> Path:
    """Write tiny SAR tiles and four-class masks for the active BRIGHT splits."""
    from geo_eval_harness.datasets.bright_post_sar import BRIGHTPostSAR

    directory = root / "data" / "bright-post-sar"
    y, x = np.indices((BRIGHTPostSAR.raster_size,) * 2)
    for split in BRIGHTPostSAR.split_sizes:
        for index, tile in enumerate(BRIGHTPostSAR.split_ids(split)):
            mask = ((x // 2 + y // 4 + index) % 4).astype(np.uint8)
            for path, pixels in (
                (directory / "post-event" / f"{tile}_post_disaster.tif", mask * 60 + index),
                (directory / "target" / f"{tile}_building_damage.tif", mask),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(geotiff_bytes(pixels[None].astype(np.uint8)))
    return directory


def write_bright_post_sar_archives(root: Path) -> dict[str, str]:
    """Zip the toy tiles like the upstream release and return their SHA-256 manifest."""
    import hashlib

    directory = write_bright_post_sar_files(root / "files")
    checksums = {}
    for kind in ("post-event", "target"):
        path = root / f"{kind}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for member in (directory / kind).iterdir():
                archive.write(member, f"{kind}/{member.name}")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


#: Fixture cities; with 8-pixel tiles they give 7 / 4 / 4 train/val/test tiles.
OSCD_SHAPES = {"alpha": (12, 20), "bravo": (6, 8), "charlie": (10, 9), "delta": (16, 16)}


def write_oscd_post_files(root: Path) -> Path:
    """Write the ``small_oscd_post`` cities in torchgeo's OSCD layout."""
    from geo_eval_harness.datasets.oscd_post import OSCDPost

    directory = root / "data" / "oscd-post"
    for index, (city, shape) in enumerate(OSCD_SHAPES.items()):
        y, x = np.indices(shape)
        mask = ((x // 3 + y // 2 + index) % 2 + 1).astype(np.uint8)
        label = OSCDPost.label_path(directory, city)
        label.parent.mkdir(parents=True)
        label.write_bytes(geotiff_bytes(mask[None]))
        # torchgeo's OSCD also indexes the unused pre image and dates.txt.
        post = OSCDPost.image_dir(directory, city)
        for images in (post, post.parent / "imgs_1_rect"):
            images.mkdir(parents=True)
            for band_index, band in enumerate(OSCDPost.bands):
                pixels = 1000 + 100 * band_index + 50 * mask.astype(int) + index + x
                (images / f"{band.source_name}.tif").write_bytes(
                    geotiff_bytes(pixels.astype(np.uint16)[None])
                )
        (post.parent / "dates.txt").write_text("date_1: 20160101\ndate_2: 20180101\n")
    return directory


def write_oscd_post_archives(root: Path) -> list[Path]:
    """Zip the fixture into torchgeo's three OSCD release archives."""
    directory = write_oscd_post_files(root)
    return [
        Path(shutil.make_archive(str(root / folder.name), "zip", directory, folder.name))
        for folder in sorted(directory.iterdir())
    ]


def _tortilla_bytes(samples: list[tuple[dict[str, Any], bytes]]) -> bytes:
    # Tacoreader's 18-byte header stores little-endian Parquet footer offset and size.
    contents = bytearray(18)
    rows = []
    for metadata, payload in samples:
        rows.append({**metadata, "tortilla:offset": len(contents), "tortilla:length": len(payload)})
        contents.extend(payload)
    footer = pd.DataFrame(rows).to_parquet(index=False)
    contents[:18] = struct.pack("<2sQQ", b"#y", len(contents), len(footer))
    contents.extend(footer)
    return bytes(contents)


def write_caffe_files(root: Path, *, target_class: int | None = None) -> Path:
    """Write a real nested Taco containing disjoint grayscale/mask GeoTIFF samples."""
    directory = root / "data" / "geobenchv2" / "caffe"
    directory.mkdir(parents=True)
    samples = []
    for split_index, (split, count) in enumerate((("train", 6), ("validation", 4), ("test", 4))):
        for index in range(count):
            y, x = np.indices((16, 16))
            mask = ((x // 8 + 2 * (y // 8) + index) % 4).astype(np.uint8)
            if target_class is not None:
                mask.fill(target_class)
            pixels = 20 + mask.astype(np.float32) * 55 + split_index * 0.5 + index * 0.02
            mask[0, 0] = 255
            sample_id = f"{split}-{index}"
            assets = [
                (
                    {
                        "tortilla:id": f"{sample_id}-{kind}",
                        "tortilla:file_format": "GTiff",
                        "stac:centroid": "POINT (10 45)",
                    },
                    geotiff_bytes(array[None]),
                )
                for kind, array in (("image", pixels), ("mask", mask))
            ]
            samples.append(
                (
                    {
                        "tortilla:id": sample_id,
                        "tortilla:data_split": split,
                        "tortilla:file_format": "TORTILLA",
                    },
                    _tortilla_bytes(assets),
                )
            )
    (directory / "geobench_caffe.tortilla").write_bytes(_tortilla_bytes(samples))
    return directory


def write_torchgeo_download_files(root: Path, name: str) -> Path:
    """Create small imagery ZIPs and standard/spatial split files for HTTP downloads."""
    from torchgeo.datasets import RESISC45, EuroSAT, EuroSATSpatial

    dataset_class = EuroSAT if name == "eurosat" else RESISC45
    directory = root / name
    directory.mkdir(parents=True)
    image_root = EuroSAT.base_dir if name == "eurosat" else RESISC45.directory
    classes = ("AnnualCrop", "Forest") if name == "eurosat" else ("airplane", "forest")
    split_ids: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    with zipfile.ZipFile(directory / dataset_class.filename, "w") as archive:
        for split_index, (split, ids) in enumerate(split_ids.items()):
            for class_index, label in enumerate(classes):
                suffix = "tif" if name == "eurosat" else "jpg"
                sample_id = f"{label}_{split}.{suffix}"
                ids.append(sample_id)
                value = 20 + class_index * 100 + split_index
                if name == "eurosat":
                    payload = geotiff_bytes(np.full((13, 16, 16), value, np.uint16))
                else:
                    buffer = io.BytesIO()
                    Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(buffer, "JPEG")
                    payload = buffer.getvalue()
                archive.writestr(f"{image_root}/{label}/{sample_id}", payload)
    for split, ids in split_ids.items():
        filenames = (
            [EuroSAT.split_filenames[split], EuroSATSpatial.split_filenames[split]]
            if name == "eurosat"
            else [f"resisc45-{split}.txt"]
        )
        for filename in filenames:
            (directory / filename).write_text("\n".join(ids) + "\n")
    return directory
