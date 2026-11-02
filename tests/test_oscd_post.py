"""OSCD post-image dataset contracts on tiny GeoTIFFs in torchgeo's layout."""

import random
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from geo_eval_harness.datasets.oscd_post import (
    OFFICIAL_TEST_CITIES,
    OFFICIAL_TRAIN_CITIES,
    VAL_CITIES,
    OSCDPost,
)
from geo_eval_harness.models._band_mapping import canonical_band_name
from geo_eval_harness.models._input_units import InputUnit, detect_input_unit
from geo_eval_harness.models._normalization import build_normalizer
from tests.support.data import write_oscd_post_files


def test_splits_are_official_cities_with_a_seeded_val_draw() -> None:
    assert sorted(random.Random(0).sample(sorted(OFFICIAL_TRAIN_CITIES), 3)) == list(VAL_CITIES)
    splits = OSCDPost.split_cities
    assert set(splits["train"]) == set(OFFICIAL_TRAIN_CITIES) - set(VAL_CITIES)
    assert set(splits["test"]) == set(OFFICIAL_TEST_CITIES)


@pytest.mark.usefixtures("small_oscd_post")
def test_tiles_are_raw_zero_padded_and_remapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_oscd_post_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    dataset = OSCDPost().get_dataset("train", bands=("b04", "b03", "b02"))
    assert dataset.tiles == [("alpha", r, c) for r in range(2) for c in range(3)] + [
        ("bravo", 0, 0)
    ]
    # Bottom-right tile of the 12 x 20 city: only the top-left 4 x 4 pixels are real.
    sample = dataset[5]
    with rasterio.open(OSCDPost.label_path(root, "alpha")) as raster:
        native = raster.read(1)[8:, 16:]
    with rasterio.open(OSCDPost.image_dir(root, "alpha") / "B04.tif") as raster:
        red = raster.read(1)[8:, 16:]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (3, 8, 8)
    assert torch.equal(sample["image"][0, :4, :4], torch.from_numpy(red.astype(np.float32)))
    assert sample["mask"].dtype == torch.int64
    assert torch.equal(sample["mask"][:4, :4], torch.from_numpy(native - 1).long())
    padding = torch.ones(8, 8, dtype=torch.bool)
    padding[:4, :4] = False
    assert (sample["image"][:, padding] == 0).all()
    assert (sample["mask"][padding] == 255).all()


@pytest.mark.usefixtures("small_oscd_post")
def test_missing_files_and_wrong_tile_counts_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_oscd_post_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(OSCDPost.split_sizes, "val", 5)
    with pytest.raises(ValueError, match="val has 4 tiles, expected 5"):
        OSCDPost().get_dataset("val")
    (OSCDPost.image_dir(root, "delta") / "B10.tif").unlink()
    with pytest.raises(FileNotFoundError, match="geo-eval-harness download oscd-post"):
        OSCDPost().get_dataset("test")


def test_l1c_bands_are_s2_dn_and_zscore_to_unit_scale() -> None:
    bands = OSCDPost.bands
    assert detect_input_unit(bands) == InputUnit.S2_DN
    assert detect_input_unit(OSCDPost.select_band_specs(OSCDPost.rgb_bands)) == InputUnit.S2_DN
    assert canonical_band_name("b10") == "cirrus"
    assert len({canonical_band_name(band.name) for band in bands}) == 13
    mean = torch.tensor([band.mean for band in bands]).view(1, -1, 1, 1)
    std = torch.tensor([band.std for band in bands]).view(1, -1, 1, 1)
    images = mean + std * torch.randn(
        4, len(bands), 64, 64, generator=torch.Generator().manual_seed(0)
    )
    output = build_normalizer("bandspec_zscore", bands)(images)
    assert output.mean().abs() < 0.05
    assert (output.std() - 1).abs() < 0.05
