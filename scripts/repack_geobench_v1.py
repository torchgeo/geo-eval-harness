"""Repack custom GeoBench V1 HDF5 files with JSON metadata into tar shards.

Usage::

    python scripts/repack_geobench_v1.py m-eurosat
    python scripts/repack_geobench_v1.py m-bigearthnet --shard-size 1000
    python scripts/repack_geobench_v1.py m-eurosat --validate

Each output sample is::

    <id>.bands.npz   per-band float arrays keyed by their source name
    <id>.meta.json   data-only metadata from the HDF5 ``metadata_json`` attribute

Keep partition files unchanged to preserve splits. Never read or convert pickle metadata.
"""

import argparse
import io
import json
import logging
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np

from geo_eval_harness.datasets._metadata import decode_metadata, read_hdf5_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_sample(hdf5_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return band arrays and data-only metadata for one V1 HDF5 sample."""
    with h5py.File(hdf5_path, "r") as f:
        metadata = read_hdf5_metadata(f.attrs)
        bands = {k: f[k][:] for k in f}
    if any(array.dtype.hasobject for array in bands.values()):
        raise ValueError(f"Band arrays in {hdf5_path} must not contain Python objects.")
    return bands, metadata


def repack(dataset_dir: Path, out_dir: Path, shard_size: int = 1000) -> int:
    """Repack per-sample HDF5 files into WebDataset shards, returning the shard count."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive.")
    sample_paths = sorted(dataset_dir.glob("*.hdf5"))
    if not sample_paths:
        raise FileNotFoundError(f"No id_*.hdf5 files in {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Packing %d samples into shards of %d...", len(sample_paths), shard_size)
    written = 0
    for shard, start in enumerate(range(0, len(sample_paths), shard_size)):
        with tarfile.open(out_dir / f"shard_{shard:05d}.tar", "w") as sink:
            for hp in sample_paths[start : start + shard_size]:
                bands, metadata = _read_sample(hp)
                bands_buf = io.BytesIO()
                np.savez(bands_buf, **bands)
                for suffix, payload in (
                    ("bands.npz", bands_buf.getvalue()),
                    ("meta.json", json.dumps(metadata, allow_nan=False).encode("utf-8")),
                ):
                    member = tarfile.TarInfo(f"{hp.stem}.{suffix}")
                    member.size = len(payload)
                    sink.addfile(member, io.BytesIO(payload))
                written += 1
                if written % 1000 == 0:
                    logger.info("  packed %d / %d", written, len(sample_paths))

    # Preserve dataset splits, statistics, and attribution.
    for sidecar in dataset_dir.iterdir():
        if sidecar.suffix == ".json" or sidecar.name in (
            "LICENSE",
            "README",
            "README.md",
        ):
            shutil.copy2(sidecar, out_dir / sidecar.name)

    logger.info("Wrote %d samples to %s", written, out_dir)
    return written


def index_shards(shard_paths: list[Path]) -> dict[str, dict[str, tuple[Path, int, int]]]:
    """Index tar members by suffix so sample IDs may contain dots."""
    index: dict[str, dict[str, tuple[Path, int, int]]] = {}
    for path in shard_paths:
        with tarfile.open(path, "r") as t:
            for m in t.getmembers():
                for ext in ("bands.npz", "meta.json"):
                    suffix = "." + ext
                    if m.name.endswith(suffix):
                        base = m.name[: -len(suffix)]
                        index.setdefault(base, {})[ext] = (path, m.offset_data, m.size)
                        break
    return index


def validate(dataset_dir: Path, out_dir: Path, n_samples: int = 50) -> None:
    """Compare a reproducible sample of images and metadata in the two layouts."""
    import random

    sample_paths = sorted(dataset_dir.glob("*.hdf5"))
    rng = random.Random(0)
    rng.shuffle(sample_paths)
    targets = {p.stem: p for p in sample_paths[:n_samples]}

    shard_paths = sorted(out_dir.glob("shard_*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {out_dir}")

    logger.info("Validating %d samples against %d shards...", len(targets), len(shard_paths))
    index = index_shards(shard_paths)

    def _read(ref: tuple[Path, int, int]) -> bytes:
        path, offset, size = ref
        with open(path, "rb") as stream:
            stream.seek(offset)
            return stream.read(size)

    found = 0
    for sid in list(targets):
        parts = index.get(sid)
        if not parts:
            continue
        with np.load(io.BytesIO(_read(parts["bands.npz"])), allow_pickle=False) as archive:
            new_bands = {name: archive[name] for name in archive.files}
        if "meta.json" not in parts:
            raise ValueError(f"{sid}: missing '.meta.json' metadata; pickle is not supported.")
        new_meta = decode_metadata(_read(parts["meta.json"]))

        ref_bands, ref_meta = _read_sample(targets[sid])

        assert set(new_bands) == set(ref_bands), f"{sid}: band keys differ"
        for k in ref_bands:
            assert np.array_equal(new_bands[k], ref_bands[k]), f"{sid}: band '{k}' differs"
        assert ref_meta == new_meta, f"{sid}: metadata differs"
        found += 1

    if found != len(targets):
        raise RuntimeError(f"only {found}/{len(targets)} samples checked — missing in shards")
    logger.info("OK — %d samples bit-equal between HDF5 and shards", found)


def main() -> None:
    """Repack one GeoBench V1 dataset into WebDataset shards."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="V1 dataset name (e.g. m-eurosat)")
    parser.add_argument(
        "--root", default="data/classification_v1.0", help="GeoBench V1 collection root"
    )
    parser.add_argument(
        "--out-root",
        default="data/classification_v1.0_wds",
        help="Output root for sharded copies",
    )
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument(
        "--validate", action="store_true", help="run a 50-sample bit-equality check"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.root) / args.dataset
    out_dir = Path(args.out_root) / args.dataset
    repack(dataset_dir, out_dir, shard_size=args.shard_size)
    if args.validate:
        validate(dataset_dir, out_dir)


if __name__ == "__main__":
    main()
