"""Check committed geography records without loading raw imagery."""

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from geo_eval_harness.datasets import list_datasets
from geo_eval_harness.geography import (
    GEO_ALIAS,
    INDEX_NAME,
    NO_GEO,
    STORE_DIR,
    GeoRecord,
    _v1_origin,
    build_index,
    extract_geography,
    list_geography,
    missing_datasets,
    write_record,
)

VALID_STATUSES = {"extracted", "no_geo", "not_downloaded"}


@pytest.fixture(scope="module")
def store() -> dict[str, GeoRecord]:
    if not STORE_DIR.is_dir():
        pytest.skip(f"geography store not generated at {STORE_DIR}")
    return list_geography()


@pytest.mark.parametrize("storage", ["string", "bytes"])
def test_v1_origin_reads_hdf5_metadata(tmp_path: Path, storage: str) -> None:
    metadata = {
        "label": 0,
        "bands_order": ["B04"],
        "B04": {"transform": [10, 0, 456000, 0, -10, 1230000], "crs": "EPSG:32615"},
    }
    payload = json.dumps(metadata)
    path = tmp_path / "sample.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs["metadata_json"] = (
            payload if storage == "string" else np.bytes_(payload.encode("utf-8"))
        )

    assert _v1_origin(str(path)) == (456000.0, 1230000.0, "EPSG:32615")


def test_v1_origin_requires_sample_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _v1_origin(str(tmp_path / "missing.hdf5"))


@pytest.mark.parametrize("directory_exists", [False, True])
def test_extract_geography_requires_imagery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, directory_exists: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    if directory_exists:
        (tmp_path / "data/classification_v1.0/m-eurosat").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="`geo-eval-harness download geobench_v1`"):
        extract_geography("m-eurosat")


def test_extract_geography_reads_hotosm_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "hotosm_buildings"
    root.mkdir(parents=True)
    points = [(-77.0, -12.0, "Peru"), (31.0, -26.0, "Eswatini"), (139.0, 35.0, "Japan")]
    for split, (lon, lat, country) in zip(("train", "val", "test"), points, strict=True):
        pq.write_table(
            pa.table({"lon": [lon], "lat": [lat], "country": [country]}), root / f"{split}.parquet"
        )

    record = extract_geography("hotosm_buildings")

    assert record.status == "extracted"
    assert record.version == "hotosm"
    assert record.n == 3
    assert record.points == [list(point) for point in points]
    assert record.bbox == [-77.0, -26.0, 139.0, 35.0]
    assert sum(count for _, _, count in record.bins) == 3


def test_build_index_weights_continents_by_sample_count(tmp_path: Path) -> None:
    for record in (
        GeoRecord("a", "extracted", n=3, continents={"Europe": 100.0}),
        GeoRecord("b", "extracted", n=1, continents={"Asia": 100.0}),
        GeoRecord("c", "no_geo", reason="No coordinates"),
    ):
        write_record(record, tmp_path)

    index = build_index(tmp_path)
    assert index["totals"] == {
        "datasets": 3,
        "extracted": 2,
        "samples": 4,
        "continents": {"Europe": 75.0, "Asia": 25.0},
    }
    assert build_index(tmp_path) == index
    assert json.loads((tmp_path / INDEX_NAME).read_text()) == index


def test_store_contains_exactly_the_registered_datasets(store: dict[str, GeoRecord]) -> None:
    """Keep every registered dataset represented in the geographic store."""
    assert set(store) == set(list_datasets())
    assert missing_datasets() == set(), (
        f"registered datasets with no geography record: {sorted(missing_datasets())}"
    )


def test_statuses_are_valid(store: dict[str, GeoRecord]) -> None:
    for name, record in store.items():
        assert record.status in VALID_STATUSES, f"{name}: bad status {record.status!r}"


def test_absent_coordinates_are_explained(store: dict[str, GeoRecord]) -> None:
    """A dataset without coordinates must say why, so the map can disclose it."""
    for name, record in store.items():
        if record.status in ("no_geo", "not_downloaded"):
            assert record.reason, f"{name}: status={record.status} but no reason given"


def test_known_no_geo_datasets_are_declared(store: dict[str, GeoRecord]) -> None:
    """Datasets with no source coordinates must retain that status."""
    for name in NO_GEO:
        if name in store:
            assert store[name].status == "no_geo", f"{name} unexpectedly has coordinates"


def test_aliases_point_at_their_source(store: dict[str, GeoRecord]) -> None:
    """Re-split datasets borrow geometry from the dataset holding it."""
    for name, target in GEO_ALIAS.items():
        if name in store and store[name].status == "extracted":
            assert store[name].alias_of == target


def test_extracted_records_are_wellformed(store: dict[str, GeoRecord]) -> None:
    extracted = [r for r in store.values() if r.status == "extracted"]
    if not extracted:
        pytest.skip("no dataset has extracted coordinates in this store")

    for record in extracted:
        assert record.n > 0, f"{record.name}: extracted but n=0"
        assert record.bbox is not None, f"{record.name}: extracted but no bbox"

        min_lon, min_lat, max_lon, max_lat = record.bbox
        assert -180 <= min_lon <= max_lon <= 180, f"{record.name}: bad lon bbox {record.bbox}"
        assert -90 <= min_lat <= max_lat <= 90, f"{record.name}: bad lat bbox {record.bbox}"

        assert record.bins, f"{record.name}: no density bins"
        assert record.points, f"{record.name}: no sampled points"
        assert sum(c for _, _, c in record.bins) == record.n, (
            f"{record.name}: bin counts do not sum to n"
        )

        if record.continents:
            total = sum(record.continents.values())
            assert 99.0 <= total <= 101.0, f"{record.name}: continents sum to {total}"

        # Bounds are rounded to three decimal places, so allow 0.001 at each edge.
        for lon, lat, *_ in record.points:
            assert min_lon - 0.001 <= lon <= max_lon + 0.001, (
                f"{record.name}: lon {lon} outside bbox"
            )
            assert min_lat - 0.001 <= lat <= max_lat + 0.001, (
                f"{record.name}: lat {lat} outside bbox"
            )


def test_index_matches_the_records(store: dict[str, GeoRecord]) -> None:
    index_path = STORE_DIR / INDEX_NAME
    assert index_path.exists(), "index.json missing; run the extractor to rebuild it"

    index = json.loads(index_path.read_text())
    entries = {e["name"]: e for e in index["datasets"]}
    assert set(entries) == set(store), "index.json is stale relative to the record files"

    for name, entry in entries.items():
        assert entry["status"] == store[name].status
        assert entry["n"] == store[name].n

    totals = index["totals"]
    assert totals["datasets"] == len(store)
    assert totals["samples"] == sum(r.n for r in store.values())


@pytest.mark.parametrize(
    "record",
    [
        GeoRecord("unlocated", "no_geo", reason="Coordinates are not published"),
        GeoRecord("absent", "not_downloaded", reason="Imagery is not installed"),
        GeoRecord(
            "located",
            "extracted",
            version="v1",
            n=2,
            alias_of="original",
            bbox=[10.0, 20.0, 11.0, 21.0],
            continents={"Asia": 100.0},
            bins=[[760, 440, 1], [764, 444, 1]],
            points=[[10.0, 20.0, "北"], [11.0, 21.0]],
        ),
    ],
)
def test_records_roundtrip_through_json(tmp_path: Path, record: GeoRecord) -> None:
    path = write_record(record, tmp_path)
    assert list_geography(tmp_path) == {record.name: record}
    payload = json.loads(path.read_text())
    assert GeoRecord.from_json({**payload, "future_field": True}) == record
    assert path.read_text() == json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_stored_files_are_canonical_json(store: dict[str, GeoRecord]) -> None:
    """Canonical JSON keeps regenerated records byte-identical."""
    for path in sorted(STORE_DIR.glob("*.json")):
        raw = path.read_text()
        assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")), (
            f"{path.name} is not canonical; regenerate it with the extractor"
        )
