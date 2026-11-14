"""BRIGHT post-event SAR dataset contracts on tiny GeoTIFFs."""

from pathlib import Path

import pytest
import rasterio
import torch

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.bright_post_sar import BRIGHTPostSAR, BRIGHTPostSARRGB
from geo_eval_harness.datasets.loading import download_command
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


@pytest.mark.usefixtures("small_bright_post_sar")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_rgb_variant_repeats_sar_into_identical_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    write_bright_post_sar_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert get_bench_dataset_class("bright-post-sar-rgb") is BRIGHTPostSARRGB
    sar_dataset = BRIGHTPostSAR().get_dataset(split)
    rgb_dataset = BRIGHTPostSARRGB().get_dataset(split)
    assert rgb_dataset.paths == sar_dataset.paths
    sar = sar_dataset[0]
    rgb = rgb_dataset[0]
    assert rgb["image"].shape == (3, *sar["image"].shape[-2:])
    assert all(torch.equal(rgb["image"][i], sar["image"][0]) for i in range(3))
    assert torch.equal(rgb["mask"], sar["mask"])
    assert [b.name for b in BRIGHTPostSARRGB.bands] == ["red", "green", "blue"]
    assert all(b.wavelength_um is None and b.sensor == "aerial" for b in BRIGHTPostSARRGB.bands)
    two = BRIGHTPostSARRGB().get_dataset("train", bands=("red", "green"))[0]
    assert two["image"].shape[0] == 2
    assert all(torch.equal(channel, two["image"][0]) for channel in two["image"])


def test_rgb_variant_metadata_and_download_share_sar_source() -> None:
    from geo_eval_harness.datasets import BRIGHTPostSARRGB as public_class

    assert public_class is BRIGHTPostSARRGB
    assert BRIGHTPostSARRGB.data_root() == BRIGHTPostSAR.data_root()
    assert download_command("bright-post-sar-rgb") == download_command("bright-post-sar")
    assert [b.name for b in BRIGHTPostSARRGB().resolve_band_specs("rgb")] == [
        "red",
        "green",
        "blue",
    ]
    (sar,) = BRIGHTPostSAR.bands
    for band in BRIGHTPostSARRGB.bands:
        assert (band.mean, band.std, band.min, band.max) == (
            sar.mean,
            sar.std,
            sar.min,
            sar.max,
        )


def test_rgb_variant_rejects_unsupported_partition_and_bands() -> None:
    with pytest.raises(ValueError, match="default partition"):
        BRIGHTPostSARRGB().get_dataset("train", partition="custom")
    with pytest.raises(ValueError, match="sar"):
        BRIGHTPostSARRGB().get_dataset("train", bands=("sar",))
