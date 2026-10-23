"""Runtime band selection uses dataset metadata without loading samples."""

import pytest

from geo_eval_harness.datasets import get_bench_dataset_class
from geo_eval_harness.datasets.base import BenchDataset


@pytest.fixture(params=["m-eurosat", "so2sat", "caffe", "kuro_siwo", "treesatai", "pastis"])
def bench(request: pytest.FixtureRequest) -> BenchDataset:
    return get_bench_dataset_class(request.param)()


def test_rgb_uses_dataset_specific_names(bench: BenchDataset) -> None:
    selected = bench.resolve_band_specs("rgb")
    assert [band.name for band in selected] == bench.rgb_bands
    by_name = {band.name: band for band in bench.bands}
    assert all(band is by_name[name] for band, name in zip(selected, bench.rgb_bands, strict=True))


def test_all_preserves_original_band_objects_and_order(bench: BenchDataset) -> None:
    selected = bench.resolve_band_specs("all")
    assert len(selected) == bench.num_channels
    assert all(actual is expected for actual, expected in zip(selected, bench.bands, strict=True))


def test_explicit_names_preserve_requested_order(bench: BenchDataset) -> None:
    expected = list(reversed(bench.bands))
    selected = bench.resolve_band_specs([band.name for band in expected])
    assert all(actual is band for actual, band in zip(selected, expected, strict=True))


def test_explicit_names_accept_an_iterator(bench: BenchDataset) -> None:
    selected = bench.resolve_band_specs(band.name for band in bench.bands)
    assert all(actual is expected for actual, expected in zip(selected, bench.bands, strict=True))


def test_unknown_explicit_band_is_rejected(bench: BenchDataset) -> None:
    with pytest.raises(ValueError, match="unknown band"):
        bench.resolve_band_specs(["not-a-band"])


@pytest.mark.parametrize("selection", ["", "RGB", "typo", "red,green,blue"])
def test_unknown_selection_is_not_interpreted_as_all(bench: BenchDataset, selection: str) -> None:
    with pytest.raises(ValueError, match="band selection"):
        bench.resolve_band_specs(selection)


def test_existing_none_selection_keeps_all_bands(bench: BenchDataset) -> None:
    selected = bench.select_band_specs(None)
    assert all(actual is expected for actual, expected in zip(selected, bench.bands, strict=True))


@pytest.mark.parametrize(
    ("dataset", "channels"), [("caffe", 1), ("kuro_siwo", 2), ("m-eurosat", 3)]
)
def test_rgb_does_not_assume_three_channels(dataset: str, channels: int) -> None:
    bench = get_bench_dataset_class(dataset)()
    assert len(bench.resolve_band_specs("rgb")) == channels
