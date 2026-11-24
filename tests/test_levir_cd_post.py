"""LEVIR-CD post-image tiling, labels, membership and normalization."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from geo_eval_harness.datasets.levir_cd_post import LEVIRCDPost
from geo_eval_harness.models._input_units import InputUnit, detect_input_unit
from geo_eval_harness.models._normalization import build_normalizer
from tests.support.data import write_levir_cd_post_files


@pytest.mark.usefixtures("small_levir_cd_post")
def test_tiles_follow_image_row_col_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = write_levir_cd_post_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    dataset = LEVIRCDPost().get_dataset("val")
    assert len(dataset) == 8
    for index in range(8):
        image_index, tile = divmod(index, 4)
        row, col = divmod(tile, 2)
        window = np.s_[row * 8 : row * 8 + 8, col * 8 : col * 8 + 8]
        name = f"val_{image_index + 1}.png"
        image = np.asarray(Image.open(directory / "B" / name))[window].transpose(2, 0, 1)
        native = np.asarray(Image.open(directory / "label" / name))[window]
        sample = dataset[index]
        assert sample["image"].dtype == torch.float32
        assert torch.equal(sample["image"], torch.from_numpy(image.astype(np.float32)))
        # Stray 156/254 values count as change, like torchgeo.
        assert torch.equal(sample["mask"], torch.from_numpy((native > 0).astype(np.int64)))


@pytest.mark.usefixtures("small_levir_cd_post")
def test_band_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_levir_cd_post_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    full = LEVIRCDPost().get_dataset("train")[5]["image"]
    selected = LEVIRCDPost().get_dataset("train", bands=("blue", "red"))[5]["image"]
    assert torch.equal(selected, full[[2, 0]])


@pytest.mark.usefixtures("small_levir_cd_post")
def test_incomplete_split_names_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = write_levir_cd_post_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    (directory / "label" / "test_2.png").unlink()
    with pytest.raises(FileNotFoundError, match="geo-eval-harness download levir-cd-post"):
        LEVIRCDPost().get_dataset("test")


def test_uint8_unit_and_zscore() -> None:
    assert detect_input_unit(LEVIRCDPost.bands) is InputUnit.UINT8
    mean = torch.tensor([b.mean for b in LEVIRCDPost.bands]).view(1, 3, 1, 1)
    std = torch.tensor([b.std for b in LEVIRCDPost.bands]).view(1, 3, 1, 1)
    images = torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(0)) * std + mean
    normalized = build_normalizer("bandspec_zscore", LEVIRCDPost.bands)(images)
    assert torch.allclose(normalized.mean((0, 2, 3)), torch.zeros(3), atol=0.05)
    assert torch.allclose(normalized.std((0, 2, 3)), torch.ones(3), atol=0.05)
