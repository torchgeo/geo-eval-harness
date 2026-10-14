# Add HOT VHR building segmentation (`hotosm_buildings`)

Source: https://huggingface.co/datasets/hotosm/vhr-building-segmentation, pinned to
revision `8d3e64e5c69aa37209953cce3a48df1092bc7c94` (2026-05-08).

Decisions:

- Use the official train/val/test splits as they are: no subsampling and no de-duplication.
- Feature-cache scaling is a model/pipeline problem and stays out of scope here (see "Out of scope").
- Make loading efficient with a one-time conversion at download time (see 2).

## Facts from inspecting the data (downloaded and fully scanned 2026-10-08)

| Item | Value |
|---|---|
| Files | 15 parquet shards under `data/`: 11 train, 2 validation, 2 test. 5.9 GB total. Parquet codec `UNCOMPRESSED`. |
| Rows | train 57,890 · validation 7,237 · test 7,236 (matches the dataset card) |
| Columns | `image`/`mask` (HF Image structs `{bytes, path}`), `tile_id`, `tile_x/y/z`, `project_id`, `project_name`, `country`, `organisation`, `imagery_url`, `num_buildings`, `label_geojson`, `bbox_west/south/east/north` |
| Images | 256x256, uint8. 71,805 are JPEG RGB and 558 are PNG RGBA. The PNG alpha is {0, 255}, alpha=0 marks nodata, and RGB is 0 under it. `path` says `.tif` even though the bytes are JPEG/PNG. |
| Masks | 256x256 TIFF mode L with values {0, 255}. 25,946 tiles have no building pixels; 7 are all building. They take 3.8 GB of the 4.7 GB train image+mask bytes (64 KB each, uncompressed). |
| Classes | 2 (background, building). Building pixel fraction is 13.5% on train, 10.2% on val, 10.8% on test. |
| GSD | Every tile is zoom 19, so about 0.30 m x cos(lat). Effective resolution varies by OAM source; the val Peru project 16698 is visibly upsampled and blurry. |
| Train RGB stats (raw 0-255, float64, all pixels, RGBA converted to RGB) | mean 109.5668 / 102.0423 / 87.5501, std 52.4298 / 42.6833 / 40.7742. These agree with upstream `norm_stats.json` x 255. |
| Splits | Grouped by project; no `imagery_url` appears in two splits. Train has 81 projects in 20 countries, val 6 projects in 3 countries, test 6 projects in 6 countries. |
| Skew | Train: 3 Myanmar projects hold 76% of tiles. Val: Peru 64%, Eswatini 36%. Test: Eswatini 33%, Peru 19%, Japan 18%, Mozambique 18%, Tajikistan 11%, Philippines 1%. Eswatini appears in val and test but not in train. |
| Leakage / dupes | 5 tile coordinates appear in both train and val. 223 rows share a tile coordinate with another row because overlapping projects tile the same area. Kept as-is per the official split; mention in the docstring. |
| Nodata | 0.8% of train/test tiles and 11% of val tiles (mostly project 16698) have more than 5% black pixels. 51 tiles with more than 50% black still contain building pixels. |
| Label alignment | Visually checked overlays from all three splits. Footprints line up well; there are some missed buildings, which is expected for OSM labels. |
| License | Imagery CC-BY 4.0 (OAM); labels ODbL 1.0 (OSM). Users download from HF; we redistribute nothing. |

## Design

### 1. Name and metadata (`src/torchgeo_bench/datasets/hotosm_buildings.py`)

`class HOTBuildings(BenchDataset)`:

- `name = "hotosm_buildings"`, `task = "segmentation"`, `num_classes = 2`, `multilabel = False`,
  `supports_partitions = False`
- `split_sizes = {"train": 57890, "val": 7237, "test": 7236}`
- `rgb_bands = ["red", "green", "blue"]`; `bands` uses sensor `"aerial"`, source names `R/G/B`,
  wavelengths 0.65/0.55/0.45 µm (the UC Merced convention), min 0 and max 255, and the train stats
  above. Re-measure them with `scripts/compute_band_statistics.py --dataset hotosm_buildings` and
  note in a comment that they came from that script.
- `data_root()` returns `Path("data/hotosm_buildings")`.
- Split name mapping: `val` maps to upstream `validation`.

### 2. Efficient storage: convert once at download, read from RAM

Why the raw shards don't work as-is:

- Each row group holds about 1,100 rows (about 90 MB). Shuffled `__getitem__` access would
  re-read a whole row group for every sample.
- Loading the upstream image and mask columns takes 4.7 GB for train alone (8.2 GB peak RSS),
  almost all of it uncompressed TIFF masks.

The approach is one conversion pass in the download step, writing `data/hotosm_buildings/{train,val,test}.parquet`:

| Column | Content |
|---|---|
| `image` | Upstream JPEG/PNG bytes, unchanged (no re-encode, so no extra loss) |
| `mask` | Lossless PNG (mode L) with 0 = background, 1 = building, and **255 where the PNG image alpha is 0** (nodata gets the pipeline's default `ignore_index`) |
| `tile_id`, `project_id`, `country`, `lon`, `lat` | `lon`/`lat` are bbox centres, used for geography and per-project analysis later |

Measured: a re-encoded mask averages 1.1 KB, about 57x smaller. That makes the train split about
1.0 GB (0.9 GB of images plus 66 MB of masks), val and test about 0.13 GB each. Conversion streams
row group by row group through a `ParquetWriter`, so memory stays low. It takes roughly 1 minute
single-threaded (0.8 ms per mask).

- After a successful conversion, `download` deletes the upstream shards and frees 5.9 GB. Re-running
  the download fetches them again. It also checks that the row counts equal `split_sizes`.
- Loader: in `__init__`, `pq.read_table(path, columns=["image", "mask"])` keeps the two
  `BinaryArray`s.
  - Arrow buffers have no per-item Python refcounts, so forked DataLoader workers share them
    copy-on-write.
  - Spawn and forkserver workers require `--workers 0` to avoid serializing a full split into each worker.
  - `__getitem__` decodes with PIL. The image goes through `.convert("RGB")` to a float32
    `(3, 256, 256)` tensor; the mask becomes an int64 `(256, 256)` tensor.
  - It returns `{"image", "mask"}`. Band selection uses `select_bands` from `_transforms.py`, applied
    before the caller's transform; resizing 256 to 224 is handled by the existing `_ResizeTransform`
    (nearest for masks).
- A missing split file raises `FileNotFoundError`; `get_datasets` turns that into the download hint.

A small `torch.utils.data.Dataset` subclass lives in the same module. No torchgeo base class fits:
`NonGeoDataset` adds nothing here.

### 3. Download (`src/torchgeo_bench/download.py`)

- Constants: `HOTOSM_REPO = "hotosm/vhr-building-segmentation"` and
  `HOTOSM_REVISION = "8d3e64e5..."`.
- `download_hotosm_buildings(output_dir)`:
  1. Run `snapshot_download(repo_type="dataset", revision=HOTOSM_REVISION, allow_patterns=["data/*.parquet"], local_dir=<root>/upstream)`.
     The HF LFS etag verification covers integrity, so no extra sha256 table is needed (unlike the
     AID zip).
  2. Convert each split as in section 2.
  3. Delete `upstream/`.

  The conversion function (`convert_hotosm_split(src_files, dst)`) lives in the dataset module, so
  the tests can build fixtures with it.
- Add `"hotosm_buildings"` to `DIRECT_DATASETS`, route it in `download_datasets`, and update the
  module docstring.
- `datasets/loading.py`: register it in `_REGISTRY_SPEC` (segmentation section) and add it to the
  name set in `download_command`.
- `datasets/__init__.py`: add `"HOTBuildings"` to `__all__` and `_LAZY_CLASSES`, alphabetically.

### 4. Geography (`src/torchgeo_bench/geography.py`)

This dataset has real coordinates, so the status is `extracted`, not `no_geo`. Add a third branch to
`extract_geography`: if `directory/"train.parquet"` exists, read `lon`, `lat` and `country` (as the
place) from all three split files, with `version="hf_parquet"`. Generate it with
`scripts/extract_dataset_geography.py --dataset hotosm_buildings` and commit
`docs/_static/_dataset_geography/hotosm_buildings.json` plus `index.json`.

### 5. Tests

- `tests/test_hotosm_buildings.py` (fast, no real data):
  - Build two tiny upstream-format shards in `tmp_path`: an 8x8 JPEG row and a PNG RGBA row with
    alpha=0 pixels, plus {0, 255} TIFF masks.
  - Run `convert_hotosm_split`, then `HOTBuildings().get_dataset(...)` under `monkeypatch.chdir`.
  - Assert:
    - the image is float32 with shape `(3, 8, 8)`;
    - the mask is int64 with values in {0, 1, 255}, and 255 appears exactly at the alpha=0 pixels;
    - band selection order and subsets work (parametrised like `test_aid.py`), with the
      transform seeing the selected bands;
    - a missing file raises `FileNotFoundError`;
    - `"val"` maps to upstream `validation`.
- `tests/test_split_sizes.py`: add the `EXPECTED_SIZES` entry.
- `tests/support/data.py`: give `require_dataset_data` a `data/hotosm_buildings` path.
- `tests/test_download.py`:
  - test that dispatch calls `download_hotosm_buildings`;
  - mock `snapshot_download` and assert the pinned `revision` and `allow_patterns`.
- `tests/test_geography.py` picks up the new record automatically.

### 6. Docs and display

- `docs/user/datasets.rst`:
  - a row in the segmentation table;
  - a line in the filesystem layout and download command block;
  - notes on the label source, licenses, the Myanmar skew, and the official splits including the
    5 train/val tile-coordinate overlaps.
- `docs/api/datasets.rst`: add `.. autoclass::`.
- `docs/user/changelog.rst`: add an entry under Unreleased.
- `README.md`: update the dataset count or list if it enumerates datasets.
- `scripts/regen_leaderboard.py`: `DATASET_DISPLAY["hotosm_buildings"] = "HOT Buildings"`.

### 7. Verification

```bash
torchgeo-bench download hotosm_buildings
pytest tests/test_hotosm_buildings.py tests/test_download.py tests/test_geography.py -v
pytest tests/test_split_sizes.py -m slow -k hotosm
python scripts/compute_band_statistics.py --dataset hotosm_buildings   # expect the values above
torchgeo-bench datasets hotosm_buildings
cat > /tmp/hotosm-smoke.yaml <<'YAML'
segmentation:
  head: fpn
  cache_features: false
YAML
torchgeo-bench run --config /tmp/hotosm-smoke.yaml --model timm/resnet50 --dataset hotosm_buildings --workers 0  # GPU smoke run
```

Sanity checks:

- An FPN probe should clearly beat the all-background baseline. That baseline gets mIoU of about
  0.43: background IoU of about 0.87 and building IoU of 0.
- Look at a few test predictions from the Eswatini and Japan projects.

## Out of scope (model/pipeline side)

`evaluate_segmentation` caches every split's features in RAM and then moves them to the GPU.
On the full 57,890-tile train split that is roughly 70 GB of ViT-B 4-layer fp16 features, plus
23 GB of int64 masks. Until the pipeline handles this (for example a disk-backed cache or automatic
streaming above a size budget), run this dataset with `segmentation.cache_features=false`; the
streaming `SegmentationSolver.fit` path already exists. Keep the setting the same for every model so
results stay comparable. Note that with 10 epochs over 58k tiles, the head gets about 14x more
gradient steps than on the 4k-tile V2 datasets.
