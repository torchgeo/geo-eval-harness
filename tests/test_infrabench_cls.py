"""Infra-Bench CLS split loading, band selection, download, and geography."""

import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from geo_eval_harness import download, geography
from geo_eval_harness.datasets.infrabench_cls import CLASS_NAMES, TILE_SIZE, InfraBenchCLS
from geo_eval_harness.datasets.loading import download_command

TILES = [
    ("africa_energy", "a", "energy.distribution.substation_minor", (9, 60, 60)),
    ("africa_energy", "b", "energy.generation.solar_farm", (9, 61, 62)),
    ("africa_transport", "shared", "transport.train_station", (9, 60, 23)),
    ("asia_transport", "shared", "transport.train_station", (9, 60, 60)),
    ("asia_water", "c", "water.treatment.plant", (9, 60, 60)),
]
SPLITS = {"a": "train", "b": "train", "shared": "val", "c": "test"}


def write_cells(root: Path, tiles: list[tuple[str, str, str, tuple[int, ...]]]) -> None:
    for cell, asset_id, asset_type, shape in tiles:
        cell_dir = root / f"dataset_{cell}_v1_1k"
        (cell_dir / "images").mkdir(parents=True, exist_ok=True)
        image_file = f"{asset_id}_stac_sentinel2_ms+sentinel1.npy"
        bands = np.arange(shape[0], dtype=np.float32).reshape(-1, 1, 1)
        np.save(cell_dir / "images" / image_file, np.broadcast_to(bands, shape).copy())
        manifest_path = cell_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {"records": []}
        )
        manifest["records"].append(
            {
                "asset_id": asset_id,
                "asset_type": asset_type,
                "image_file": image_file,
                "lat": 10.0,
                "lon": 20.0,
            }
        )
        manifest_path.write_text(json.dumps(manifest))
    table = pd.DataFrame({"asset_id": list(SPLITS), "split": list(SPLITS.values())})
    table.to_parquet(root / "asset_id_to_split_v1.parquet")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "infrabench_cls"
    write_cells(root, TILES)
    return root


def test_splits_follow_asset_ids_across_cells(root: Path) -> None:
    bench = InfraBenchCLS()
    train, val, test = (bench.get_dataset(split) for split in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (2, 2, 1)
    labels = [int(train[i]["label"]) for i in range(len(train))]
    assert labels == [
        CLASS_NAMES.index("energy.distribution.other"),
        CLASS_NAMES.index("energy.generation.solar_farm"),
    ]
    assert int(test[0]["label"]) == CLASS_NAMES.index("water.water_works")


def test_tiles_are_raw_float32_at_a_fixed_size(root: Path) -> None:
    sample = InfraBenchCLS().get_dataset("val")[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (9, TILE_SIZE, TILE_SIZE)
    torch.testing.assert_close(sample["image"][:, 0, 0], torch.arange(9, dtype=torch.float32))
    assert sample["label"].dtype == torch.long


@pytest.mark.parametrize(
    ("bands", "expected"),
    [(None, list(range(9))), (("b04", "b03", "b02"), [0, 1, 2]), (("vh", "b08"), [8, 3])],
)
def test_band_selection_runs_before_the_caller_transform(
    root: Path, bands: tuple[str, ...] | None, expected: list[int]
) -> None:
    seen: list[torch.Tensor] = []

    def transform(sample: dict) -> dict:
        seen.append(sample["image"].clone())
        return sample

    InfraBenchCLS().get_dataset("train", bands=bands, transform=transform)[0]
    torch.testing.assert_close(seen[0][:, 0, 0], torch.tensor(expected, dtype=torch.float32))


def test_unknown_asset_type_is_rejected(root: Path) -> None:
    write_cells(root, [("europe_water", "a", "water.fountain", (9, 60, 60))])
    with pytest.raises(ValueError, match="unknown asset_type"):
        InfraBenchCLS().get_dataset("train")


def test_unknown_split_is_rejected(root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        InfraBenchCLS().get_dataset("valid")


def test_missing_data_names_the_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="download infrabench-cls"):
        InfraBenchCLS().get_dataset("train")


def test_extract_geography_uses_the_manifests(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        geography, "_attribute_continents", lambda lon, lat: Counter(Africa=len(lon))
    )
    record = geography.extract_geography("infrabench-cls")
    assert (record.status, record.version, record.n) == ("extracted", "manifest", len(TILES))


def test_download_hint_names_the_dataset() -> None:
    assert download_command("infrabench-cls") == "geo-eval-harness download infrabench-cls"


def test_download_extracts_cells_and_verifies_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = io.BytesIO()
    with zipfile.ZipFile(cell, "w") as archive:
        archive.writestr("dataset_africa_energy_v1_1k/manifest.json", '{"records": []}')
    source = tmp_path / "bundle.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("datasets/dataset_africa_energy_v1_1k.zip", cell.getvalue())
        archive.writestr("datasets/substations.zip", b"not extracted")
    split = tmp_path / "split.parquet"
    split.write_bytes(b"split")
    monkeypatch.setattr(
        download, "INFRABENCH_SPLIT_SHA256", hashlib.sha256(split.read_bytes()).hexdigest()
    )
    out = tmp_path / "data"
    target = out / "infrabench_cls"

    def fetch(
        repo_id: str, revision: str, local_dir: Path, *args: object, **kwargs: object
    ) -> None:
        shutil.copy(source, local_dir / download.INFRABENCH_ZIP)

    def retrieve(url: str, dest: Path) -> None:
        Path(dest).write_bytes(split.read_bytes())

    with (
        mock.patch.object(download, "fetch_pinned", side_effect=fetch) as pinned,
        mock.patch.object(download.urllib.request, "urlretrieve", side_effect=retrieve),
    ):
        download.download_datasets(["infrabench-cls"], out)

    assert pinned.call_args.args[:3] == (
        download.INFRABENCH_REPO,
        download.INFRABENCH_REVISION,
        target,
    )
    assert pinned.call_args.kwargs["allow_patterns"] == [download.INFRABENCH_ZIP]
    assert (target / "dataset_africa_energy_v1_1k" / "manifest.json").is_file()
    assert not list(target.glob("*substation*"))

    monkeypatch.setattr(download, "INFRABENCH_SPLIT_SHA256", "0" * 64)
    with (
        mock.patch.object(download, "fetch_pinned", side_effect=fetch),
        mock.patch.object(download.urllib.request, "urlretrieve", side_effect=retrieve),
        pytest.raises(ValueError, match="SHA-256 checksum mismatch"),
    ):
        download.download_infrabench_cls(out)
