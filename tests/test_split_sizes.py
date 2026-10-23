"""Check reference split sizes against available datasets; skip missing data."""

import pytest

from geo_eval_harness.datasets import get_bench_dataset_class
from tests.support.data import require_dataset_data

# V1 counts come from data/classification_v1.0/<name>/default_partition.json.
# V2 counts come from len(...) on each upstream geobench_v2.datasets.GeoBench<X> split.
# EuroSAT uses torchgeo split-file counts: 27000 images split 60/20/20.
# RESISC45 uses torchgeo split-file counts: 31500 images split 60/20/20.
# UCMerced uses torchgeo split-file counts: 2100 images split 60/20/20.
# HOTOSM buildings uses the official Hugging Face split row counts.
EXPECTED_SIZES: dict[str, dict[str, int]] = {
    "m-eurosat": {"train": 2000, "val": 1000, "test": 1000},
    "m-forestnet": {"train": 6464, "val": 989, "test": 993},
    "m-so2sat": {"train": 19992, "val": 986, "test": 986},
    "m-pv4ger": {"train": 11814, "val": 999, "test": 999},
    "m-brick-kiln": {"train": 15063, "val": 999, "test": 999},
    "m-bigearthnet": {"train": 20000, "val": 1000, "test": 1000},
    "benv2": {"train": 20000, "val": 4000, "test": 4000},
    "treesatai": {"train": 4000, "val": 1000, "test": 2000},
    "so2sat": {"train": 19992, "val": 986, "test": 986},
    "forestnet": {"train": 6464, "val": 989, "test": 993},
    "caffe": {"train": 4000, "val": 1000, "test": 2000},
    "burn_scars": {"train": 524, "val": 160, "test": 120},
    "cloudsen12": {"train": 4000, "val": 535, "test": 975},
    "dynamic_earthnet": {"train": 700, "val": 100, "test": 200},
    "flair2": {"train": 4049, "val": 1022, "test": 3022},
    "fotw": {"train": 4000, "val": 1000, "test": 2000},
    "kuro_siwo": {"train": 4000, "val": 1000, "test": 2000},
    "pastis": {"train": 1455, "val": 482, "test": 496},
    "spacenet2": {"train": 5186, "val": 1461, "test": 2961},
    "spacenet7": {"train": 3500, "val": 652, "test": 1152},
    "eurosat": {"train": 16200, "val": 5400, "test": 5400},
    "resisc45": {"train": 18900, "val": 6300, "test": 6300},
    "ucmerced": {"train": 1260, "val": 420, "test": 420},
    "eurosat-spatial": {"train": 16200, "val": 5400, "test": 5400},
    "aid": {"train": 6000, "val": 2000, "test": 2000},
    "hotosm_buildings": {"train": 57890, "val": 7237, "test": 7236},
}


@pytest.mark.slow
@pytest.mark.parametrize("dataset_name", sorted(EXPECTED_SIZES))
def test_split_sizes(dataset_name: str) -> None:
    require_dataset_data(dataset_name)
    bench_cls = get_bench_dataset_class(dataset_name)
    bench = bench_cls()
    expected = EXPECTED_SIZES[dataset_name]

    actual = {
        split: len(bench.get_dataset(split, bands=tuple(bench.rgb_bands)))
        for split in ("train", "val", "test")
    }

    assert bench.split_sizes == expected
    assert actual == expected, (
        f"{dataset_name}: split sizes diverge from reference. expected={expected}, got={actual}"
    )
