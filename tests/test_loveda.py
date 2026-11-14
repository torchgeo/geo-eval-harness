"""LoveDA dataset contracts on tiny PNG tiles and the checked-in memberships."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from geo_eval_harness.datasets.loveda import LoveDA, carve_official_train
from geo_eval_harness.models._input_units import InputUnit, detect_input_unit
from geo_eval_harness.models._normalization import build_normalizer
from tests.support.data import write_loveda_files

pytestmark = pytest.mark.usefixtures("small_loveda")


def test_raw_rgb_remapped_mask_and_band_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_loveda_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    dataset = LoveDA().get_dataset("train", bands=None)
    assert [len(LoveDA().get_dataset(split)) for split in ("train", "val", "test")] == [2, 1, 2]

    sample = dataset[0]
    image = np.array(Image.open(directory / "Train/Rural/images_png/0.png"))
    native = torch.from_numpy(np.array(Image.open(directory / "Train/Rural/masks_png/0.png")))
    assert sample["image"].dtype == torch.float32
    assert torch.equal(sample["image"], torch.from_numpy(image).permute(2, 0, 1).float())
    assert sample["mask"].dtype == torch.int64
    assert torch.equal(sample["mask"], torch.where(native == 0, 255, native.long() - 1))

    selected = LoveDA().get_dataset("train", bands=("blue", "red"))[0]["image"]
    assert torch.equal(selected, sample["image"][[2, 0]])


def test_missing_tile_names_download_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_loveda_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    (directory / "Train/Urban/images_png/2.png").unlink()
    with pytest.raises(FileNotFoundError, match="geo-eval-harness download loveda"):
        LoveDA().get_dataset("train")


def test_checked_in_memberships_are_the_seeded_carve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LoveDA, "splits_dir", None)
    members = {split: LoveDA.split_members(split) for split in LoveDA.split_sizes}
    assert {split: len(paths) for split, paths in members.items()} == LoveDA.split_sizes
    assert carve_official_train(sorted(members["train"] | members["val"])) == {
        "train": members["train"],
        "val": members["val"],
    }
    assert all(path.startswith("Val/") for path in members["test"])


def test_bands_are_uint8_and_zscore_train_like_pixels() -> None:
    assert detect_input_unit(LoveDA.bands) is InputUnit.UINT8
    mean = torch.tensor([band.mean for band in LoveDA.bands]).view(1, 3, 1, 1)
    std = torch.tensor([band.std for band in LoveDA.bands]).view(1, 3, 1, 1)
    pixels = mean + std * torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    normalized = build_normalizer("bandspec_zscore", LoveDA.bands)(pixels)
    assert normalized.mean().abs() < 0.05
    assert (normalized.std() - 1).abs() < 0.05
