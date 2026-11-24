"""PLANTEUR crop-type segmentation: conversion, loading, band and month selection."""

import csv
import json
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from geo_eval_harness import download, hub
from geo_eval_harness.datasets.planteur import PLANTEUR, PLANTEURLOIOMayotte, convert_planteur

SPLIT_IDS = {"train": [3, 0], "val": [1], "test": [2]}
SIZES = {split: len(ids) for split, ids in SPLIT_IDS.items()}
LOIO_IDS = {"train": [2, 0], "val": [1], "test": [3]}
# Patch 2 stores linear VV 0.2 and VH 1.2, which the conversion turns into dB.
DB_0_2, DB_1_2 = 10 * np.log10(0.2), 10 * np.log10(1.2)


def _write_upstream(root: Path) -> None:
    """Write release-format patches whose pixel values encode patch, month, and channel."""
    for name in ("DATA_S2_MONTHLY", "DATA_S1_MONTHLY", "ANNOTATIONS", "splits"):
        (root / name).mkdir(parents=True)
    features = []
    months = np.arange(12)[:, None, None, None]
    for patch_id in range(4):
        s2 = 1000 * np.arange(12)[None, :, None, None] + 10 * months + patch_id
        s1 = np.arange(2)[None, :, None, None] + 0.01 * months + 0.1 * patch_id
        np.save(
            root / "DATA_S2_MONTHLY" / f"S2_{patch_id}.npy",
            np.broadcast_to(s2, (12, 12, 128, 128)).astype(np.uint16),
        )
        np.save(
            root / "DATA_S1_MONTHLY" / f"S1_{patch_id}.npy",
            np.broadcast_to(s1, (12, 2, 128, 128)).astype(np.float32),
        )
        target = np.full((1, 128, 128), patch_id, dtype=np.uint8)
        target[0, :, 64:] = 255
        np.save(root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy", target)
        lon, lat = -61.0 + patch_id, 16.0
        ring = [[lon, lat], [lon + 0.2, lat], [lon + 0.2, lat + 0.2], [lon, lat + 0.2], [lon, lat]]
        features.append(
            {
                "properties": {"ID_PATCH": patch_id, "Island": f"island-{patch_id}"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    (root / "splits" / "443.json").write_text(json.dumps({"splits": SPLIT_IDS}))
    (root / "splits" / "loio.json").write_text(json.dumps({"folds": {"Mayotte": LOIO_IDS}}))
    (root / "metadata.geojson").write_text(json.dumps({"features": features}))


@pytest.fixture
def converted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _write_upstream(tmp_path / "upstream")
    target = tmp_path / "data" / "planteur"
    convert_planteur(tmp_path / "upstream", target, expected_sizes=SIZES)
    return target


def test_conversion_keeps_split_order_labels_and_centres(converted: Path) -> None:
    train = PLANTEUR().get_dataset("train", bands=("b02",))
    assert [len(PLANTEUR().get_dataset(split)) for split in SIZES] == list(SIZES.values())
    for index, patch_id in enumerate(SPLIT_IDS["train"]):
        sample = train[index]
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (1, 128, 128)
        assert sample["image"][0, 0, 0] == patch_id
        expected = torch.full((128, 128), patch_id, dtype=torch.int64)
        expected[:, 64:] = 255
        torch.testing.assert_close(sample["mask"], expected)
    with (converted / "metadata.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["id_patch"]), row["split"]) for row in rows] == [
        (3, "train"),
        (0, "train"),
        (1, "val"),
        (2, "test"),
    ]
    assert rows[0]["island"] == "island-3"
    assert (float(rows[0]["lon"]), float(rows[0]["lat"])) == pytest.approx((-57.9, 16.1))


def test_conversion_rejects_unexpected_split_size(tmp_path: Path) -> None:
    _write_upstream(tmp_path / "upstream")
    with pytest.raises(ValueError, match="train: expected 3 patches, got 2"):
        convert_planteur(
            tmp_path / "upstream", tmp_path / "out", expected_sizes={**SIZES, "train": 3}
        )


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        (None, [1000.0 * band + 2 for band in range(12)] + [DB_0_2, DB_1_2]),
        (("vh", "b04", "vv"), [DB_1_2, 2002.0, DB_0_2]),
    ],
)
def test_band_selection_spans_sensors_in_request_order(
    converted: Path, bands: tuple[str, ...] | None, expected: list[float]
) -> None:
    del converted
    sample = PLANTEUR().get_dataset("test", bands=bands)[0]
    torch.testing.assert_close(
        sample["image"][:, 5, 5], torch.tensor(expected, dtype=torch.float32)
    )


def test_time_steps_select_evenly_spaced_months(converted: Path) -> None:
    del converted
    sample = PLANTEUR().get_dataset("val", bands=("b03",), time_steps=3)[0]
    assert sample["image"].shape == (3, 1, 128, 128)
    torch.testing.assert_close(sample["image"][:, 0, 0, 0], torch.tensor([1001.0, 1041.0, 1081.0]))
    with pytest.raises(ValueError, match="between 1 and 12"):
        PLANTEUR().get_dataset("val", time_steps=13)


def test_loio_fold_reads_patches_across_default_splits_in_list_order(converted: Path) -> None:
    del converted
    bench = PLANTEURLOIOMayotte()
    assert bench.data_root() == PLANTEUR.data_root()
    for split, ids in LOIO_IDS.items():
        dataset = bench.get_dataset(split, bands=("b02",))
        assert [int(dataset[i]["image"][0, 0, 0]) for i in range(len(dataset))] == ids
        assert [int(dataset[i]["mask"][0, 0]) for i in range(len(dataset))] == ids


def test_pickled_split_reopens_its_arrays(converted: Path) -> None:
    del converted
    dataset = PLANTEUR().get_dataset("val", bands=("vv",))
    expected = dataset[0]["image"]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._arrays == {}
    torch.testing.assert_close(restored[0]["image"], expected)


def test_missing_split_points_to_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="geo-eval-harness download planteur"):
        PLANTEUR().get_dataset("train")


@pytest.mark.parametrize("digest_matches", [True, False])
def test_download_verifies_stacked_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, digest_matches: bool
) -> None:
    monkeypatch.setattr(PLANTEUR, "split_sizes", SIZES)
    checksums = {} if digest_matches else {"val_mask.npy": "0" * 64}
    monkeypatch.setattr(download, "load_checksums", lambda manifest: checksums)

    def fake_snapshot(*, local_dir: Path, **_: object) -> None:
        _write_upstream(local_dir)

    root = tmp_path / "planteur"
    with mock.patch.object(hub, "snapshot_download", side_effect=fake_snapshot) as snapshot:
        if digest_matches:
            download.download_datasets(["planteur"], tmp_path)
        else:
            with pytest.raises(ValueError, match="checksum mismatch"):
                download.download_planteur(tmp_path)

    snapshot.assert_called_once_with(
        repo_id="allenai/planteur-benchmark",
        repo_type="dataset",
        revision=download.PLANTEUR_REVISION,
        local_dir=root / "upstream",
        allow_patterns=download.PLANTEUR_FILES,
        cache_dir=None,
    )
    assert (root / "upstream").exists() != digest_matches
    assert (root / "test_s1.npy").is_file()
