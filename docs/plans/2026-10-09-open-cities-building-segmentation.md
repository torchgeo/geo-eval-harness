# Add Open Cities AI Challenge building segmentation (`open_cities`)

Source: https://source.coop/open-cities/ai-challenge (GFDRR Labs 2020, DOI 10.34911/rdnt.f94cxb).
Read via the public S3-compatible endpoint `https://data.source.coop/open-cities/ai-challenge/`.

Companion to `docs/plans/2026-10-08-hotosm-building-segmentation.md`. HOT ships ready-made tiles with
official splits. This dataset ships large georeferenced drone scenes and an unlabelled test set, so the
chip grid and the train/val/test assignment have to be defined here and committed to the repo.

## Execution notes (read first)

- **Execute this plan carefully and in order.** The committed index fixes the benchmark's splits for good, so a
  mistake in steps 1-9 of Design §1 cannot be fixed silently later. If the index has to change after merge, that
  means a new dataset version (new `name` or partition), never an in-place edit.
  - After each stage, stop and check its output against the numbers in this plan before going on: chip
    counts, the dedup and Zanzibar drops, the split table, and the split maps.
  - Differences from the trial run (below) must be explained before committing.
- **Environment:** implement and test in the `testpy313` conda env. Call the binaries directly, e.g.
  `/home/nils/miniconda3/envs/testpy313/bin/python` and `.../bin/pytest`, because `conda run` can pick up the
  wrong Python. Checked 2026-10-09: Python 3.13.13, rasterio 1.5.0 / GDAL 3.12.1, geopandas 1.1.3, shapely 2.1.2.
- **Trial run first (done 2026-10-09).** `docs/plans/assets/2026-10-09-open-cities-split-trial/` contains:
  - `open_cities_split_trial.py`: the script, run with testpy313;
  - `open_cities_index_trial.csv`: the full chip index;
  - `blocks.geojson`: for QGIS;
  - `summary.md`: per-city and per-split tables;
  - `split_map.png` plus `split_map_<city>.png`: block maps with chips per block and scene ids;
  - `chip_gallery.png`: native 512 px chips per city and split with label outlines and no-data shading.

  The trial changed two design points (the validity rule in §1.3 and chip-level dedup in §1.6). The user
  inspected it on 2026-10-09 and accepted the splits. `scripts/generate_open_cities_index.py` should start from
  that script. The committed index may differ from the trial only through the open decisions below.

## Implementation status (2026-10-09, branch `open-cities`)

Done (all uncommitted on branch `open-cities`, created from `main`):

- **§1 index:**
  - `scripts/generate_open_cities_index.py` was run. `open_cities_index.csv` is **identical to the trial CSV in
    every row and column** (110,462 chips; 76,818 / 11,236 / 22,408). It has a header comment with the seeds and
    the dropped Zanzibar blocks.
  - `open_cities.sha256` is written.
  - Tier-1 data is downloaded to `data/open_cities/` (32 GiB, about 7 min).
- **§2:**
  - `download_open_cities`, `fetch_open_cities_file` (resume and UA `geo-eval-harness`; Python's default UA gets
    403) and `_check_open_cities_masks` are in `download.py`.
  - `build_scene_mask`, `is_valid`, `load_index`, `load_checksums` and `upstream_url` are in
    `datasets/open_cities.py`.
- **§3:**
  - `OpenCities` and `_OpenCitiesSplit` are written and registered in `loading.py` (`_REGISTRY_SPEC` and
    `download_command`), `datasets/__init__.py` and `DIRECT_DATASETS`.
  - `split_sizes` is set.
  - **The BandSpec mean/std are still placeholders (0/1).**
- **§4:** the `geography.py` branch `_extract_open_cities` is in place. **The record JSON is not generated yet.**
- **§5:**
  - `scripts/plot_open_cities_splits.py` is written.
  - Outputs are in `figures/open_cities/` (local only): `split_map*.png`, `chip_gallery.png` and `blocks.geojson`.
  - Scene footprints come from the 1/32 overview, not the STAC geometry.
- **§6:**
  - `tests/test_open_cities.py` passes for the fixture, loading, resume, checksum and mask check.
  - It also has the `TestCommittedIndex` invariants, which have not been run yet.
  - `test_split_sizes`, `support/data.py` and the `test_download` dedup parametrize are updated.
  - `--ignore-index` was added to `compute_band_statistics.py`, with a test.
  - Per the user, the split-assignment code gets **no unit tests**, but both scripts go into the PR.
- **§7 (partial):** `regen_leaderboard.py` has `DATASET_DISPLAY` and `RGB_ONLY_DATASETS` entries.

Done in the second session (2026-10-09):

- Masks: the first background build was cut off by a machine restart (8 stale `.tmp` files, no masks). The rerun
  wrote **30 masks**, because `znz/bc32f1` has no chips in the index: all of its blocks fall under the Zanzibar rule,
  as in the trial. All 3 splits log "masks match the index" (|delta| <= 0.0002). The 8-thread build emits a
  spurious rasterio `NotGeoreferencedWarning`; masks are byte-identical between 1 and 8 threads (checked on 2,580
  windows).
- Geography: `open_cities.json` and `index.json` written. 19% of chips land in "Ocean / unmatched" (coastal
  Zanzibar and Dar es Salaam against the coarse land polygons). `extract_geography` now delegates file-based
  extraction to `_extract_files` (ruff PLR0911).
- Docs: `datasets.rst` (intro, layout row, download line, own section with the split map), `api/datasets.rst`,
  `changelog.rst`, `README.md`.
- `pyproject.toml`: `geopandas`, `rasterio`, `shapely` added (`pyproj` is only used by the maintainer script).

Done in the third session (2026-10-09):

- Band statistics (2 workers, batch 8, niced; about 35 min) pasted into `OpenCities.bands`: R 126.58/49.13,
  G 122.15/45.81, B 100.84/52.72 (mean/std over non-nodata train pixels).
- `uv lock` adds only the three direct dependencies (they were already transitive via torchgeo). The wheel ships
  `open_cities_index.csv` (11.5 MB) and `open_cities.sha256`.
- Slow `test_split_sizes` and `TestCommittedIndex`: 17 passed. Full fast suite: 2406 passed; the 14 failures and 1
  error are environmental and unrelated (the `geo-eval-harness` binary is not on PATH when pytest is called directly,
  no local `results/all_results.csv`, installed torchgeo lacks `EuroSAT.sha256`).
- §8: 3.2 ms per random train `__getitem__` (warm page cache); all-background mIoU on 2,000 random test chips is
  0.371 (building fraction 0.259).
- **Not done:** the GPU smoke run. The workstation has no GPU, and without partitions a CPU run would cover all
  76,818 train chips.

Remaining before the third session, kept for the record:

1. Band statistics. **Run with at most 2 workers, batch 8, `nice`, `OMP_NUM_THREADS=1`, `GDAL_CACHEMAX=128`**:
   a 16-worker run nearly took down the 15 GB / 12-core machine.
   `python scripts/compute_band_statistics.py --dataset open_cities --ignore-index 255 --num-workers 2 --batch-size 8`,
   then paste the statistics into `OpenCities.bands`.
2. Full `pytest`, `pytest -m slow -k open_cities tests/test_split_sizes.py`, `TestCommittedIndex`, and a wheel build to
   confirm the CSV and sha256 files ship. `uv lock` for the new dependencies.
3. §8 sanity checks: time 1,000 random `__getitem__` calls; GPU smoke run with `segmentation.head=fpn`;
   all-background mIoU baseline of about 0.37.
4. Commit and open the PR, including both scripts and this plan (the class docstring links to it).

## Decisions

Settled with the user on 2026-10-09:

- **Native resolution, no resampling.** Chips are cut on each scene's own pixel grid. The roughly 10x GSD
  spread (0.02-0.20 m) is expected, and any resizing is a model/experiment setting.
- **Geographic block splits, hardcoded.** Train/val/test are assigned per 500 m geographic block and committed
  as a chip index CSV, so the splits never depend on re-running code.
- **Processing scripts live in the repo** (`scripts/`), and their outputs are committed.
- **Nothing resized is ever processed or stored.** Upstream imagery stays byte-identical, and masks are rasterised on
  the same native pixel grid. Resizing happens only at load time, in the datamodule/transform, as each model
  experiment requires.
- **512x512 px native chips** are the default (`chip_size` is a single constant, so 1024 stays possible).
- **Zanzibar block rule:** keep only 500 m blocks with at least 1% building cover.
- **Split maps:** the train/val/test geography must be clearly visible. A committed figure for each city plus
  an overview, and a blocks GeoJSON (see Design §5).
- **Splits as in the trial:** 70/10/20 of chips within each city, balanced jointly on chip count and building
  pixels, using the 1,000-shuffle search. Accepted after inspecting the trial maps.

Confirmed by the user on 2026-10-09 (formerly open D1-D3):

| # | Decision |
|---|---|
| D1 | **Tier 1 only** (31 scenes, 7 cities). Tier 2 is a possible later noisy-train partition (Follow-ups) |
| D2 | **Keep the upstream COGs byte-identical; precompute one native-grid mask GeoTIFF per scene; read chips by window** |
| D3 | **No buffer between splits** (a one-chip buffer would drop 8.3% of chips: 14% of test, 16% of val, 30% of Pointe-Noire) |

All decisions are settled, and the plan is ready for implementation.

## Facts from inspecting the data (2026-10-09)

Everything below was measured over HTTP range reads plus all 72 label GeoJSONs (1.4 GB). No full
imagery download was needed. The analysis scripts are in the session scratchpad; the numbers that
matter are reproduced here.

### Layout

| Item | Value |
|---|---|
| Bucket | 23,322 objects, 87.5 GB. `train_tier_1/` (31 scenes, 33.9 GB), `train_tier_2/` (41 scenes, about 44 GB), `test/` (11,481 chips, 9.3 GB), `documentation.pdf`, `README.md` |
| Catalog | STAC 0.8.1. `train_tier_N/catalog.json` → `<city>/collection.json` → `<scene>/<scene>.json` (image item) and `<scene>-labels/<scene>-labels.json` (label item). The asset sits next to each item: `<scene>/<scene>.tif`, `<scene>-labels/<scene>.geojson` |
| Imagery | COG with 4 bands (RGB + alpha), uint8, JPEG-compressed including alpha (lossy; no-data is alpha < 128, or black RGB in 4 Zanzibar scenes), 512x512 blocks, overviews 2/4/8/16/32. Reprojected to a local UTM zone. Native GSD runs from 0.02 m (Accra `665946`) to 0.20 m (Pointe-Noire); most scenes are 0.04-0.08 m |
| Labels | GeoJSON (CRS84) polygons with OSM tags (`building`, `building:material`, `building:roof`, ...) plus `scene_id` (the OAM image id). Clipped to the scene's valid area |
| Test | **Unusable**: 1024x1024 chips with **no labels** in the bucket, and georeferencing stripped (fake 0-0.045° bbox). Excluded |
| Dates | 2015-04 to 2019-07. Some items carry the placeholder `2019-10-29` |
| License | Labels ODbL-1.0 (OSM; Zanzibar labels come from the Zanzibar Mapping Initiative, and their item has no license field, so the dataset-level ODbL applies). Imagery CC BY 4.0 on 38 items, ODbL on 22, unset on 12. We redistribute nothing: users download from source.coop (D2) |
| Access | The anonymous S3 listing works (`?list-type=2&prefix=ai-challenge/`). **source.coop returns 403 to Python's default `urllib` User-Agent**. curl and GDAL with `GDAL_HTTP_USERAGENT` set work. Check whether GDAL's default UA works, and set it explicitly if not |

### Scenes (tier 1 is used, tier 2 is summarised)

| City (code) | Tier | Scenes | Valid km² | Buildings | Bldg cover | Native GSD (m) |
|---|---|---|---|---|---|---|
| Accra (acc) | 1 | 4 | 7.9 | 33,585 | 0.27-0.46 | 0.02-0.05 |
| Dar es Salaam (dar) | 1 | 6 | 42.8 (39.0 union) | 121,171 | 0.20-0.48 | 0.04-0.07 |
| Kampala (kam) | 1 | 1 | 1.1 | 4,056 | 0.19 | 0.035 |
| Monrovia (mon) | 1 | 4 | 1.9 | 6,947 | 0.25-0.56 | 0.04-0.08 |
| Niamey (nia) | 1 | 1 | 0.7 | 634 | 0.04 | 0.10 |
| Pointe-Noire (ptn) | 1 | 2 | 1.9 | 8,731 | 0.30-0.36 | 0.20 |
| Zanzibar (znz) | 1 | 13 | 102.6 | 13,407 | 0.000-0.10 | 0.06-0.08 |
| Dar es Salaam | 2 | 31 | 223 (161 union) | 571,047 | 0.09-0.39 | 0.04-0.08 |
| Ngaoundéré (gao), Kinshasa (kin), Mahé (mah), Niamey | 2 | 10 | 35.4 | 32,906 | 0.03-0.19 | 0.05-0.17 |

## Label-quality analysis

Three independent checks, each run over all 72 scenes.

### 1. Geometry hygiene (from the GeoJSONs)

- Invalid polygons: `dar/0a4c40` has 74 (tier 1) and `gao/4f38e1` has 9; elsewhere 0. Fix with `shapely.make_valid`.
- Exact duplicate polygons: `ptn/abe1a3` has 457 (7% of the scene, part of 960 overlapping pairs); the 2 tier-2
  Niamey scenes have more than 600 overlapping pairs each. Rasterising the union makes duplicates harmless.
- Labels outside the scene footprint: almost none in tier 1. Tier 2 has 628 in `dar/1d8af6` and 214 in `gao/4f38e1`.
  Rasterising only inside valid pixels makes these harmless.
- Tiny polygons (< 5 m²) make up 1-5% per scene, mostly sheds and kiosks. Keep them.

### 2. Label-to-image alignment (automatic, edge-based)

Method: in 30 random 96 m windows per scene (each window has at least 5 buildings and at most 5% nodata),
read at 0.3 m, take the Sobel gradient magnitude of the image, rasterise the label *boundaries*, and
search for the shift within ±6 m that maximises mean gradient along the boundaries. The output is the
per-scene shift of the summed score surface plus per-window shift statistics. Visual spot checks agree,
for example `dar/e14d1d` is visibly about 5-6 m off with outlines that don't match the roofs.

| Group | Median per-window shift | Scenes with more than 50% of windows shifted > 1 m |
|---|---|---|
| Tier 1, Accra / Kampala / Monrovia / Niamey / Pointe-Noire | 0.0-0.3 m | 0 of 12 |
| Tier 1, Zanzibar | 0.3-0.6 m (at the 0.3 m pixel quantisation) | 0 of 13 |
| Tier 1, Dar es Salaam | 0.0-0.76 m. 10-37% of windows are over 1 m; the drift is local, not a global offset | 0 of 6 |
| **Tier 2, Dar es Salaam** | **1.56 m** (median over scenes) | **22 of 31**. Global offsets of 5-6 m in `e14d1d`, `0ccd08`, `ab32c9`, `9870ba` |
| Tier 2, other cities | 0.9-5.6 m | 5 of 10 (`kin/255028` 3.9 m, `nia/982a1f` 5.6 m, `gao/048ffb` at or beyond the ±6 m search bound) |

At 0.3 m, a 2-6 m offset is 7-20 px, which is larger than many of the buildings. **This is the main
reason for D1.**

### 3. Completeness against an independent reference

Reference: Google Open Buildings v3 (confidence ≥ 0.75) plus Microsoft footprints, from the VIDA combined
GeoParquet on source.coop (`vida/google-microsoft-open-buildings/geoparquet/by_country/`, bbox-filtered
with DuckDB). Metric: the share of reference buildings inside the scene (10 m inner buffer) that have a
challenge label within 3 m. Caveat: the reference comes from 2020+ satellite imagery, while the scenes are
from 2015-2019, so buildings built later count as misses.

| Group | Median recall@3 m | Median label precision@3 m |
|---|---|---|
| Tier 1, Accra / Monrovia / Pointe-Noire | 0.98 | 0.90-0.91 |
| Tier 1, Dar es Salaam / Kampala | 0.95 / 0.96 | 0.91 / 0.89 |
| Tier 1, Niamey | 0.80 | 0.85 |
| Tier 1, Zanzibar | 0.91 (0.50 `bc32f1` to 0.98 `aee7fd`) | 0.73 |
| Tier 2, Dar es Salaam | 0.94 (0.66 `94a004` to 0.99) | 0.94 |
| Tier 2, Ngaoundéré / Kinshasa / Mahé / Niamey | 0.77 / 0.87 / 0.82 / 0.87 | 0.86-0.94 |

Visual check of the unmatched reference buildings:

- **Zanzibar** (`bc32f1`, `425403`): the misses are bare ground, foundation slabs or construction sites in the 2016
  imagery. They are reference false positives or later construction, **not label omissions**, so keep Zanzibar.
- **Dar tier 1** (`0a4c40`): genuine omissions, for example a cluster of green-roofed buildings and an industrial
  shed, plus some reference false positives (boats). Expect about 5% missing buildings in Dar tier 1.
- **`gao/048ffb`** (tier 2): clearly incomplete (recall 0.61; label area is 0.53x the reference) and misaligned.

Takeaway: tier 2's main defect is **misalignment**, not completeness. Completeness is similar to tier 1
outside a few scenes.

### 4. Other defects found

- **Monrovia `207cc7` and `401175` are stored in EPSG:32636 (UTM 36N)**, but Monrovia lies in 29N, about 44°
  of longitude away. Pixels are therefore not square on the ground. The labels still line up because they are
  CRS84 vectors. Chips stay on the native grid, so these two scenes keep anisotropic ground pixels (document it).
  Block assignment uses chip centres transformed to EPSG:32629, so the splits are unaffected.
- **Scene overlap.** All 37 Dar scenes (tiers 1 and 2) form one connected overlap component. Tier-1 Dar scenes
  overlap each other: `0a4c40`/`42f235` share 24%, `42f235`/`f883a0` 10%. Tier-2 `82a1f3` and `ca3445` lie entirely
  inside tier-1 `42f235`, and 24 of the 39 km² of tier-1 Dar are also covered by tier 2. Including tier 2 would
  leak test areas into train under any split that isn't spatial.
- Zanzibar is mostly rural bush (cover 0-10%). On the native grid it is 46% of all chips, and 86% (1024 px) to
  91% (512 px) of its chips contain no building. This is the reason for the Zanzibar block rule.

## Zanzibar subsampling: what was compared

Native grid, 512 px chips, a chip kept if at least 50% of it is valid (from the STAC footprint), 500 m blocks.
The other cities have a building-free chip rate of 24%.

| Strategy | Chips | Zanzibar share | Zanzibar empty | All empty | Building pixels | Zanzibar buildings kept |
|---|---|---|---|---|---|---|
| Keep all | 162,260 | 46% | 91% | 55% | 0.169 | 100% |
| Drop every empty Zanzibar chip | 94,993 | 7% | 0% | 22% | 0.289 | 100% |
| Cap Zanzibar empties at 24% (random) | 97,120 | 9% | 24% | 24% | 0.283 | 100% |
| Blocks with building fraction ≥ 0.5% | 111,679 | 21% | 73% | 34% | 0.246 | 99% |
| **Blocks with building fraction ≥ 1% (chosen)** | **108,031** | **18%** | **70%** | **32%** | **0.254** | **96%** |
| Blocks with building fraction ≥ 2% | 102,159 | 14% | 63% | 29% | 0.267 | 88% |
| Random 25% of blocks | 106,288 | 17% | 93% | 36% | 0.249 | 16% |

Why the block rule:

- It removes the large uninhabited bush areas but keeps whole 500 m neighbourhoods intact, including the
  rural background around settlements. Zanzibar stays more rural (70% empty) than the cities, which is real.
- It loses only 4% of Zanzibar's labelled buildings.
- It needs no random draw, so it is trivially reproducible.

Chip-level filtering instead cherry-picks chips, which changes the background distribution in a way that
no deployment would see.

At 1024 px the picture is the same (Zanzibar falls from 46% to 19% of 27k chips).

## Design

### 1. Chip index (committed): `scripts/generate_open_cities_index.py`

Run once by a maintainer; its output is committed and never regenerated by users.

1. List the tier-1 STAC (31 image items and their label items).
2. For each scene, lay a native pixel grid of `CHIP_SIZE = 512` px chips anchored at pixel (0, 0). Chip offsets are
   multiples of 512, which matches the COG's 512 px blocks, so a chip read decodes exactly one JPEG tile.
   Only full chips are used; the partial strip at the right and bottom edges is dropped.
3. Compute `valid_frac` per chip from overview 16 (32x32 px per chip; about 130 MB of range reads for all of tier 1),
   using `valid = (alpha >= 128) & (max(R, G, B) > 10)`. Keep chips with at least 50% valid pixels. The trial
   showed that both parts of this rule are needed:
   - The alpha band is **JPEG-compressed (lossy)**, so no-data edges carry alpha 1-127 on 0.1-2% of pixels.
     `alpha > 0` is wrong.
   - Zanzibar `3f8360` (9.3% of pixels), `c7415c` (8.2%), `e52478` (7.7%) and `06f252` (0.8%) store no-data as
     **black RGB under opaque alpha**. In every other scene, pixels with max(RGB) ≤ 10 and opaque alpha make up
     ≤ 0.14%, so the black rule costs almost nothing.
   The mask builder (§2) uses the same `is_valid` function to write 255.
4. Compute `building_frac` per chip from the cleaned labels (`make_valid`, union, clipped to the chip).
5. Blocks: transform the chip centre to the **city's** UTM (acc 32630, dar/znz 32737, kam 32636, mon 32629,
   nia 32631, ptn 32732). Block id = `<city>_<floor(x/500)>_<floor(y/500)>`.
6. Deduplicate overlapping scenes **at chip level**. Within a city, process scenes in order of valid-chip count
   (descending). Drop a chip when its ground footprint (corners transformed to the city UTM) overlaps a chip
   already kept from an earlier scene by more than 1% of its area. A block-level "one scene per block" rule
   was tried and rejected: it also dropped 7.8k chips of *adjacent*, non-overlapping scenes that merely share a
   block. The trial drops 3,592 chips, 3,531 of them in Dar es Salaam (`0a4c40` 1,614, `42f235` 870, `a017f9` 575,
   `b15fce` 472), which matches the roughly 4 km² of real scene overlap.
7. Zanzibar rule: drop Zanzibar blocks whose mean `building_frac` is below 0.01.
8. Splits, per city:
   - try 1,000 seeded block shuffles and fill test, then val, then train;
   - keep the shuffle with the smallest summed deviation from 70/10/20 in chip count **and** building-pixel
     mass, with at least one block per split.
   - Record the chosen seed in the CSV header comment.
9. Write `src/geo_eval_harness/datasets/open_cities_index.csv` (package data, about 110k rows, about 11 MB) with columns
   `chip_id, city, scene, gsd_m, col_off, row_off, block, split, lon, lat, valid_frac, building_frac, n_buildings`.
   Make sure the packaging config includes the CSV, and check that it ships in the built wheel.
10. Also write `src/geo_eval_harness/checksums/open_cities.sha256` with the sha256 of the 31 tier-1 `.tif` and
    31 `.geojson` files, so that a download can be verified.

Trial outcome (2026-10-09; `docs/plans/assets/2026-10-09-open-cities-split-trial/summary.md`):

- 172,628 valid chips;
- minus 3,592 overlap duplicates;
- minus 58,574 chips in 355 dropped Zanzibar blocks;
- leaving **110,462 chips** in 437 blocks.

| | train | val | test | total |
|---|---|---|---|---|
| acc | 25,055 | 3,691 | 7,371 | 36,117 |
| dar | 32,271 | 4,632 | 9,296 | 46,199 |
| kam | 2,333 | 405 | 735 | 3,473 |
| mon | 2,214 | 325 | 638 | 3,177 |
| nia | 173 | 29 | 59 | 261 |
| ptn | 125 | 20 | 37 | 182 |
| znz | 14,647 | 2,134 | 4,272 | 21,053 |
| **all** | **76,818 (69.5%)** | **11,236 (10.2%)** | **22,408 (20.3%)** | **110,462** |

Building-pixel fraction is 0.248-0.252 and the empty-chip rate 32-34% in every split.

Caveats:

- **Small cities are lumpy.** Kampala (10 blocks) has a single val block, and its test blocks are the vegetated
  western edge. Niamey (8 blocks) has 1 test block with 2.4% building cover against 5.0% in train. Their
  per-city scores rest on 1-2 blocks, so the pooled metric is the headline.
- **Chip counts per block vary with GSD.** For example, in Dar es Salaam a 500 m block holds about 361 chips
  from `f883a0` (0.051 m GSD) but about 182 from `42f235` (0.073 m). Balancing on chip count therefore weights
  finer-GSD areas more. This is acceptable because the building-pixel balance is enforced jointly.
- **No buffer between splits.** Adjacent blocks in different splits share an edge, so chips on either side
  are spatially autocorrelated. If that matters, drop chips within one chip width of a split boundary
  (an open question for the user; not applied in the trial).

### 2. Download: `download_open_cities(output_dir)` → `data/open_cities/`

1. Download the 31 tier-1 `.tif` files (33.9 GB) and 31 `.geojson` files (0.3 GB) over HTTPS from
   `data.source.coop` into `data/open_cities/<city>/<scene>.tif|.geojson`.
   - Resume partial files.
   - Verify against `open_cities.sha256`.
   - Set the User-Agent explicitly: source.coop returns 403 to Python's default `urllib` UA.
2. Build masks: for each scene, write `data/open_cities/<city>/<scene>_mask.tif`, a uint8 tiled GeoTIFF on the
   scene's **native** grid (same transform, same size, 512 px tiles; nothing is resampled) with DEFLATE and
   `SPARSE_OK=TRUE`. Only the windows in the index are written. For each indexed chip:
   - read RGBA;
   - rasterise the cleaned polygons (`all_touched=False`): 0 background, 1 building;
   - set 255 where `is_valid` is false (alpha < 128 or max(RGB) ≤ 10); 255 is the pipeline's default `ignore_index`.

   Process this per window, never the whole scene (Accra `665946` alone is 12.7 Gpx). Use a thread pool over
   scenes.
3. Check that each split's mean `building_frac` measured from the written masks matches the index within 0.002.
4. The imagery is never re-encoded. Disk: about 34 GB of imagery plus a few hundred MB of masks.

`rasterio`, `geopandas`, `shapely` and `pyproj` currently arrive only through `torchgeo`. Add the ones used
directly to `[project.dependencies]`.

### 3. Dataset class: `src/geo_eval_harness/datasets/open_cities.py`

- `class OpenCities(BenchDataset)`:
  - `name = "open_cities"`, `task = "segmentation"`, `num_classes = 2`, `multilabel = False`,
    `supports_partitions = False`.
  - `split_sizes` hardcoded from the committed index (asserted by a test).
  - The band statistics must **exclude no-data pixels**. They are black, and they make up most of every chip's
    allowed 0-50% invalid area. Add an optional `--ignore-index 255` to `scripts/compute_band_statistics.py` that
    drops pixels where `mask == ignore_index`. This is a small, general change; HOT also has nodata pixels.
  - `rgb_bands = ["red", "green", "blue"]`. `bands` uses sensor `"aerial"`, wavelengths 0.65/0.55/0.45 µm,
    min 0 / max 255, and train mean/std from `scripts/compute_band_statistics.py --dataset open_cities`.
  - `data_root() = Path("data/open_cities")`.
  - The docstring covers: tier 1, native GSD 0.02-0.20 m (a 512 px chip spans 10-100 m), 512 px native chips, 7 cities, 500 m block splits, the
    Zanzibar rule, mask encoding, licenses, and the caveats (Dar tier 1 has about 5% omissions; Monrovia
    `207cc7`/`401175` are stored in UTM 36N).
- `_OpenCitiesSplit(Dataset)`:
  - Takes the split's index rows as a DataFrame argument, so tests can inject a tiny index. `OpenCities.get_dataset`
    reads the packaged CSV and filters it by split.
  - Opens the rasterio handles lazily per worker (cached by path, and reset when the PID changes after fork).
  - `__getitem__` reads RGB bands 1-3 and the mask through the same `Window(col_off, row_off, CHIP_SIZE, CHIP_SIZE)`.
    No-data image pixels are returned as they are stored (black); the mask's 255 makes the loss and metrics
    ignore them.
  - Returns `{"image": float32 (3, 512, 512), "mask": int64 (512, 512)}` at native resolution.
  - Band selection goes through `select_bands`. **Resizing happens only here, at load time**, through the existing
    transforms (`_ResizeTransform`, nearest for masks) driven by the model/experiment config. No resized data is
    written anywhere.
- A missing scene file raises `FileNotFoundError` with the download hint.
- Registration: `_REGISTRY_SPEC`, the `download_command` name set, `datasets/__init__.py`, `DIRECT_DATASETS`
  and dispatch in `download.py`, and the module docstring. This is independent of the `hotosm-buildings`
  branch, since no parquet is involved.

### 4. Geography

Lon/lat already sit in the committed index. Add a branch to `geography.extract_geography` that reads them
(`place` = city name) from the packaged CSV, so the record doesn't need the 34 GB download. Commit
`docs/_static/_dataset_geography/open_cities.json` and `index.json`.

### 5. Split maps (committed visual record of the geography)

Script: `scripts/plot_open_cities_splits.py`. It reads only the committed index plus low-resolution COG overviews,
so it runs without the full download.

- Outputs, written to the git-ignored `figures/open_cities/` (user decision at PR time: plots stay local and go into
  the PR description instead):
  - `split_map.png`: one panel per city. Each 500 m block is filled train / val / test in the validated
    categorical colours (`#2a78d6` / `#eb6834` / `#1baf7a`) over a desaturated imagery thumbnail. Dropped Zanzibar
    blocks are dotted outlines, and each panel has a scale bar. The panel title gives per-split chip counts, and
    the legend gives the totals. The aqua swatch is below 3:1 contrast, so the legend and counts carry identity,
    not colour alone.
  - `split_map_<city>.png`: one larger map per city.
  - `blocks.geojson`: block polygons (EPSG:4326) with `city`, `split`, `n_chips`, `building_frac`, `scene`, plus the
    dropped Zanzibar blocks with `split = "dropped"`. This makes it possible to inspect the split in QGIS or
    geojson.io.
- `docs/user/datasets.rst` embeds `split_map.png` in the Open Cities section.
- The generator script calls the plotting function at the end, so the figures always match the committed CSV.
  A test checks that `blocks.geojson` agrees with the index (same blocks, same splits).

The trial outputs in `docs/plans/assets/2026-10-09-open-cities-split-trial/` are the reference for these figures.
The plotting code there (`draw_city`, `chip_gallery`) moves into this script largely unchanged.

### 6. Tests

- `tests/test_open_cities.py` (fast, no network):
  - Write a tiny local tiled RGBA GeoTIFF (UTM, with alpha=0 in one corner) and a GeoJSON with one normal,
    one duplicate and one invalid bow-tie polygon.
  - Run the mask builder for a two-chip index, using a small chip size passed as a parameter.
  - Load through `OpenCities.get_dataset` under `monkeypatch.chdir`.
  - Assert:
    - the image is float32 with the expected shape;
    - mask values are within {0, 1, 255}, and 255 appears exactly where alpha < 128 or the RGB is black. Cover
      both cases, plus alpha noise of 1-127 like JPEG produces.
    - the building pixels match the polygon to within its boundary pixels;
    - band subsets and order work;
    - a checksum mismatch raises.
- Index invariants test, reading the committed CSV:
  - `chip_id` is unique;
  - there are no overlapping windows within a scene, and offsets are multiples of `CHIP_SIZE`;
  - each block maps to exactly one split (a block may hold chips from several scenes after chip-level dedup);
  - `valid_frac >= 0.5` for every row;
  - no Zanzibar block is below 1% building fraction;
  - every city appears in every split;
  - the counts equal `OpenCities.split_sizes`.
- `tests/test_split_sizes.py` (`EXPECTED_SIZES`), `tests/support/data.py` (data path), and
  `tests/test_download.py` (dispatch, mocked HTTP and checksum).

### 7. Docs and display

- `docs/user/datasets.rst`: segmentation table row, layout and download line, the embedded split map, and a
  label-quality note linking here.
- `docs/api/datasets.rst`: add the class.
- `docs/user/changelog.rst`, and `README.md` if it enumerates datasets.
- `scripts/regen_leaderboard.py`: `DATASET_DISPLAY["open_cities"] = "Open Cities"`.

### 8. Verification

```bash
python scripts/generate_open_cities_index.py      # maintainer, once; commits CSV, sha256, split maps, blocks.geojson
geo-eval-harness download open_cities
pytest tests/test_open_cities.py tests/test_download.py tests/test_geography.py -v
pytest tests/test_split_sizes.py -m slow -k open_cities
python scripts/compute_band_statistics.py --dataset open_cities --ignore-index 255
geo-eval-harness run -m timm/resnet50 -d open_cities   # GPU smoke run (the CLI is argparse, not Hydra)
```

Sanity checks:

- The all-background baseline gets an mIoU of about 0.37 (background IoU about 0.75 at 25% building pixels).
  FPN should clearly beat it.
- Plot 2 random test chips per city with mask overlays, including Monrovia `207cc7`/`401175` and one
  Zanzibar chip.
- Data loading: time 1,000 random `__getitem__` calls (one JPEG tile per chip). At about 75k train chips, feature
  caching is a model-side question, as for HOT.

## Out of scope

- The official `test/` chips (no labels, no georeference).
- Tier 2 (D1).
- Building attributes (roof material etc.): sparse and inconsistent across cities.

## Follow-ups (not part of this change)

1. **Label-noise partition.** Use the same tier-1 val/test and train on tier 1 + tier 2, dropping tier-2 chips
   that fall into tier-1 val/test blocks. The per-scene alignment table gives a measured noise level, which
   turns the dataset's main defect into a controlled experiment.
2. **City-held-out partition** (OOD, as the original challenge did): for example test on Monrovia + Pointe-Noire
   + Kampala.
3. Run the `cleanlab_seg` label-quality pipeline on the train split. The Dar tier-1 omissions should rank high.
4. Commit the alignment and completeness audit as `scripts/audit_open_cities_labels.py`.
