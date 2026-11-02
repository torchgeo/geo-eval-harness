"""BRIGHT post-event SAR dataset contracts on tiny GeoTIFFs."""

from pathlib import Path

import pytest
import rasterio
import torch

from geo_eval_harness.datasets.bright_post_sar import BRIGHTPostSAR
from geo_eval_harness.models._input_units import InputUnit, detect_input_unit
from geo_eval_harness.models._normalization import build_normalizer
from tests.support.data import write_bright_post_sar_files


@pytest.mark.usefixtures("small_bright_post_sar")
def test_loads_raw_sar_and_native_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bright_post_sar_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    dataset = BRIGHTPostSAR().get_dataset("train")
    assert len(dataset) == 3
    sample = dataset[0]
    image_path, mask_path = dataset.paths[0]
    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
        assert torch.equal(sample["image"], torch.from_numpy(image.read().astype("float32")))
        assert torch.equal(sample["mask"], torch.from_numpy(mask.read(1).astype("int64")))
    assert set(sample["mask"].unique().tolist()) == {0, 1, 2, 3}


@pytest.mark.usefixtures("small_bright_post_sar")
def test_bad_labels_and_missing_files_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bright_post_sar_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    dataset = BRIGHTPostSAR().get_dataset("train")
    _, mask_path = dataset.paths[0]
    with rasterio.open(mask_path, "r+") as mask:
        mask.write(mask.read(1) * 0 + 4, 1)
    with pytest.raises(ValueError, match="labels outside"):
        dataset[0]
    mask_path.unlink()
    with pytest.raises(FileNotFoundError, match="geo-eval-harness download bright-post-sar"):
        BRIGHTPostSAR().get_dataset("train")


def test_packaged_splits_match_sizes_and_are_disjoint() -> None:
    splits = {split: set(BRIGHTPostSAR.split_ids(split)) for split in BRIGHTPostSAR.split_sizes}
    assert {split: len(ids) for split, ids in splits.items()} == BRIGHTPostSAR.split_sizes
    assert len(set.union(*splits.values())) == sum(BRIGHTPostSAR.split_sizes.values())


def test_sar_band_is_uint8_and_zscores_to_unit_scale() -> None:
    (band,) = BRIGHTPostSAR.bands
    assert detect_input_unit(BRIGHTPostSAR.bands) == InputUnit.UINT8
    images = band.mean + band.std * torch.randn(
        4, 1, 32, 32, generator=torch.Generator().manual_seed(0)
    )
    normalized = build_normalizer("bandspec_zscore", BRIGHTPostSAR.bands)(images)
    assert abs(float(normalized.mean())) < 0.05
    assert abs(float(normalized.std()) - 1) < 0.05
