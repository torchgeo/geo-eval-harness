# OpenEarthMap-SAR integration into geo-eval-harness

## Problem and agreed approach

Add the official OpenEarthMap-SAR land-cover dataset as a standalone SAR semantic-segmentation benchmark named `openearthmap-sar`. The integration will use the published IEEE GRSS Data Fusion Contest 2025 Track I train, validation, and test splits; download all four required Zenodo archives from their immutable record-specific URLs; expose only the single-channel SAR modality; and integrate loading, registration, download, offline tests, geography, user/API documentation, and a runnable example.

The benchmark protocol is fixed at 4,333 training images, 210 validation images, and 490 test images. Native labels `1..8` become model classes `0..7`; native label `0` becomes the repository-standard ignore value `255`. This preserves the official eight foreground-class mIoU while intentionally ignoring rare native-zero pixels during probe training, as selected by the user. RGB imagery bundled in the training archive is outside scope and will not be extracted or exposed.

The implementation plan itself must be published as `docs/plans/openearthmap-sar-integration.md` before implementation begins. This session plan is the working source of truth until plan-mode approval permits that repository change.

## Verified sources and release inventory

Primary sources:

- Dataset concept/current record: https://zenodo.org/records/14950559
- Development/train-validation record: https://zenodo.org/records/14622048
- Test-image record: https://zenodo.org/records/14948851
- Official DFC baseline: https://github.com/cliffbb/DFC2025-OEM-SAR-Baseline
- Dataset reference repository: https://github.com/cliffbb/OpenEarthMap-SAR
- Paper: https://arxiv.org/abs/2501.10891v2

The release is versioned across three Zenodo records. Do not use a mutable `latest` URL or assume the current record contains prior-version files.

| Purpose | Record | Archive | Bytes | Published checksum |
|---|---:|---|---:|---|
| Train images, train labels, validation SAR images | 14622048 | `dfc25_track1_trainval.zip` | 8,864,311,171 | `md5:100772c6daf81fd09bb869ccd6f9f5d7` |
| Test SAR images | 14948851 | `dfc25_track1_test.zip` | 403,551,447 | `md5:bc52f9c4c3b5e3db775cde84371fbc14` |
| Validation labels | 14950559 | `dfc25_track1_val_labels.zip` | 8,615,257 | `md5:7bbb7f90bc6d448fc7fa779d62e5cc73` |
| Test labels | 14950559 | `dfc25_track1_test_labels.zip` | 17,842,327 | `md5:29558d6198855e9e8ff8db31d3b6fa8b` |

The train-validation ZIP has 13,209 TIFF files:

- `train/labels/`: 4,333 label TIFFs
- `train/sar_images/`: 4,333 SAR TIFFs
- `train/rgb_images/`: 4,333 optical RGB TIFFs
- `val/sar_images/`: 210 SAR TIFFs

The test ZIP has 490 files under `test/sar_images/`. The validation-label ZIP has 210 real label TIFFs under `val/labels/` plus 214 irrelevant macOS metadata/dotfile entries under `__MACOSX` and `val/.DS_Store`; these must never be extracted. The test-label ZIP has 490 TIFFs under `test/labels/`.

All official split memberships are exact, paired, and contiguous:

- Train: `TrainArea_001.tif` through `TrainArea_4333.tif`
- Validation: `ValArea_001.tif` through `ValArea_210.tif`
- Test: `TestArea_001.tif` through `TestArea_490.tif`

SHA-256 digests of lexicographically sorted basenames serialized with LF separators and a trailing LF:

- Train: `e57f0e7cd83220fee58017bdf5ddede7879aedd1e01dc0bcdb80192c052b3830`
- Validation: `b3f5bd821fbcafba1f044a8de9cdb0f6f43abb5cfe0805a3f055190bdab64187`
- Test: `4ac13caefa337f0e1cc35bb8a483d3888d4609d2e5fa91bfbaef7376b8eeb22c`

These digests are planning cross-checks, not a request for redundant runtime hash files. Runtime membership can be generated from the fixed numeric ranges and validated against actual paired files.

## Verified raster and label semantics

- The publication and all inspected files use 1024 x 1024 pixels at 0.15-0.5 m GSD.
- SAR images are single-band grayscale GeoTIFFs stored as `uint8` in raw `0..255` units. A sampled training SAR tile used the full range and had mean 43.5147/std 24.5934; a sampled validation tile also used the full range. These sample values are not suitable as final normalization statistics.
- Training RGB files are three-band `uint8`, but are not part of this SAR-only benchmark.
- Labels are single-band `uint8` GeoTIFFs with native IDs:
  1. bareland
  2. grass/rangeland
  3. pavement/developed space
  4. road
  5. tree
  6. water
  7. cropland/agricultural land
  8. building
- Native ID `0` occurs but is rare. A full scan found it in 159/4,333 train masks, 24/210 validation masks, and 34/490 test masks. Pixel counts were 6,659,368 train, 2,038,247 validation, and 2,828,931 test. Every ID `1..8` occurs in every split.
- All 4,333 train masks and all released validation/test masks were decoded during planning to establish the complete label domain.
- The official baseline creates nine logits, trains native zero as background, and computes mIoU over classes `1..8`. Because geo-eval-harness cannot train a class while excluding only that class from metrics, the user selected `0 -> 255` and `1..8 -> 0..7`. Document this deliberate loss-level deviation from the official baseline.
- The official DFC baseline scales image bytes to `[0, 1]`, randomly crops training images to 512, and resizes validation to 1024. The geo-eval-harness adapter must instead emit raw float32 `0..255` tensors, perform no augmentation, and leave resizing/normalization to the existing runner/model configuration.
- GeoTIFF georeferencing is inconsistent. An inspected train SAR/RGB/label triplet had no GeoTIFF CRS/affine tags, while an inspected validation SAR tile carried NAD83 / UTM zone 18N tags and a 0.5 m pixel size. The geography implementation must scan real headers across all selected SAR files, retain valid locations, and report actual coverage rather than inventing coordinates or assuming every tile is georeferenced.
- Zenodo describes SAR imagery from the Umbra Open Data Program under CC BY 4.0. Optical-source and label licenses vary by source; do not describe the entire installed benchmark as uniformly CC BY 4.0 without preserving the source-specific caveat.

Planning inspection used HTTP byte ranges instead of retaining the full 8.86 GB archive. The complete ZIP central directories, the first 96 MB containing all compressed training masks, every released mask, representative SAR/RGB images, and representative TIFF geotags were inspected. The full archive still must be downloaded during implementation to verify end-to-end integrity, compute train statistics, validate every image header, and exercise real extraction.

## Fixed benchmark protocol

1. Dataset CLI name: `openearthmap-sar`.
2. Task: semantic segmentation only.
3. Input modality: the single SAR byte band only. Do not expose training RGB as an optional band because validation/test are SAR-only and modality-dependent split schemas would violate the predictable `BenchDataset` contract.
4. Data root: `data/openearthmap-sar`.
5. Preserve the official extracted subtrees directly beneath that root: `train/`, `val/`, and `test/`, each with `sar_images/` and `labels/`.
6. Train split: the 4,333 official `TrainArea_*` image/label pairs.
7. Validation split: the 210 official `ValArea_*` image/label pairs, using the labels published in record 14950559.
8. Test split: the 490 official `TestArea_*` image/label pairs, using the labels published in record 14950559.
9. Sort samples by numeric ID, not ZIP member order and not lexicographic filename order.
10. Require exact expected membership and one-to-one image/label pairing. Missing pairs, duplicate IDs, unexpected TIFF names, or malformed rasters are errors.
11. Emit image tensors as owned float32 `(1, 1024, 1024)` arrays in raw `0..255` units.
12. Emit masks as long `(1024, 1024)` arrays after mapping `0 -> 255` and `1..8 -> 0..7`. Reject any other native value before mapping.
13. Declare eight classes and use the existing segmentation `ignore_index=255`.
14. Use one `BandSpec` with sensor/name/source name `sar`, `wavelength_um=None`, and measured train-only raw-unit statistics. Set `rgb_bands = ["sar"]`, following the existing grayscale/SAR convention where the default `rgb` selector means the primary model-ready bands.
15. No runtime download, implicit split generation, tiling, augmentation, label fabrication, or fallback to RGB.

## Repository integration findings

- `src/geo_eval_harness/datasets/base.py` defines the raw float32 `BenchDataset` contract.
- `src/geo_eval_harness/datasets/loading.py::_REGISTRY_SPEC` is the authoritative lightweight catalog. It must get a standalone segmentation entry and an explicit standalone `download_command` route; otherwise it will incorrectly suggest GeoBench V2.
- `src/geo_eval_harness/datasets/__init__.py` owns sorted lazy public exports.
- A direct `BenchDataset` adapter is appropriate. OpenEarthMap-SAR is neither GeoBench V1 nor V2, and TorchGeo 0.10 has no adapter for it.
- Rasterio, NumPy, PyTorch, and torchvision are already available through hard project dependencies; no dependency-manifest change is required.
- `src/geo_eval_harness/datasets/_transforms.py::select_bands` should be reused, although the only valid selection is the single SAR band.
- `get_datasets` already owns resize behavior, categorical nearest-neighbor mask resizing, loader construction, missing-data error wrapping, and partition warnings.
- The existing segmentation solver can consume eight classes with ignored `255` pixels without changes.
- `scripts/compute_band_statistics.py` computes train-only raw-unit statistics and should be used after the real adapter/download are operational.
- `src/geo_eval_harness/download.py` explicitly registers and dispatches standalone datasets. A large ZIP64 release needs record-pinned URLs, published checksum validation, exact-member extraction, and repeat-run repair semantics.
- Python `zipfile` supports these ZIP64 archives. Extraction must whitelist expected members and reject path traversal instead of calling unrestricted `extractall`.
- Geography coverage is mandatory: `tests/test_geography.py` requires every registry entry in the committed store. Add a standalone TIFF-header extraction path that tolerates genuinely ungeoreferenced samples but fails malformed georeferencing.
- `tests/support/data.py::require_dataset_data` currently routes unknown standalone datasets to `data/geobenchv2`; it needs an explicit `data/openearthmap-sar` case.
- `docs/user/contribute_dataset.rst` requires measured BandSpecs, split-size coverage, user/API docs, changelog, and geography/index updates.
- The existing optical plan at `docs/plans/openearthmap-integration.md` contains reusable design precedent for a direct rasterio adapter, exact-member Zenodo extraction, test fixtures, geography, and documentation. Share helpers only where doing so clearly removes duplicate behavior; do not make either dataset depend on the other being implemented first.

## Implementation todos

### 1. Publishing the approved plan

- Add this fully derived plan to `docs/plans/openearthmap-sar-integration.md`.
- Keep it separate from `docs/plans/openearthmap-integration.md`; the optical and SAR releases have different archives, split protocols, modalities, and label-zero decisions.
- Do not start dataset code before the plan publication is reviewed as part of the approved implementation.

### 2. Implementing the standalone dataset adapter and registration

- Add `src/geo_eval_harness/datasets/openearthmap_sar.py` with public `OpenEarthMapSAR(BenchDataset)` and a private map-style raster dataset.
- Declare `name="openearthmap-sar"`, `task="segmentation"`, `num_classes=8`, `multilabel=False`, `supports_partitions=False`, `split_sizes={"train": 4333, "val": 210, "test": 490}`, one SAR `BandSpec`, and `rgb_bands=["sar"]`.
- Implement one split descriptor mapping each split to its prefix, numeric range, image directory, and label directory. Generate IDs with minimum three-digit zero padding and sort numerically.
- Validate `split`, `partition`, requested bands, the exact expected image/label filenames, and one-to-one spatial dimensions. Ignore no extra TIFF silently; dotfiles/non-TIFF metadata are irrelevant outside expected directories, but unexpected TIFFs inside benchmark directories should fail installation validation.
- Read with context-managed rasterio handles. Require one image band and one label band, `uint8` storage, 1024 x 1024 dimensions, and matching image/mask shapes. Copy arrays before closing file handles.
- Return raw float32 images and long masks. Convert masks to a signed integer type before mapping; map zero to 255 and subtract one from `1..8`; reject values outside `0..8`.
- Compose `select_bands` before the caller transform. Do not normalize, divide by 255, crop, resize, augment, or emit one-hot masks in the adapter.
- Register the dataset in `_REGISTRY_SPEC`, add `OpenEarthMapSAR` to lazy exports and `__all__`, and make `download_command("openearthmap-sar")` return `geo-eval-harness download openearthmap-sar`.
- Keep config, segmentation solver, result schema, resume keys, and public CLI parsing unchanged.

### 3. Implementing the verified multi-record downloader

- Add immutable archive metadata constants containing record-specific URLs, filenames, byte sizes, published MD5 values, and allowed member prefixes.
- Add `download_openearthmap_sar(output_dir)` and include `openearthmap-sar` in standalone/direct downloadable datasets and dispatch.
- Download to `data/openearthmap-sar` (or the corresponding explicit output root), verify every published checksum, and let network/checksum failures surface with the archive path and expected/actual digest.
- Inspect each ZIP index before writing files. Require the exact train/validation/test basename sets and pair relationships established above.
- Extract only:
  - `train/sar_images/*.tif`
  - `train/labels/*.tif`
  - `val/sar_images/*.tif`
  - `val/labels/*.tif`
  - `test/sar_images/*.tif`
  - `test/labels/*.tif`
- Never extract `train/rgb_images`, `__MACOSX`, `.DS_Store`, arbitrary extra members, or paths escaping the target root.
- Use a shared safe exact-member extraction helper if the optical OpenEarthMap implementation has landed; otherwise add the smallest reusable helper justified by both ZIP download paths rather than duplicating path-safety logic.
- Preserve verified archives so interrupted extraction can be repaired without another 9.29 GB network transfer. A repeat run must revalidate cached checksums and repair missing extracted files; directory existence alone is not success.
- Validate the completed installation through the same split/path authority used by the runtime loader.
- Add progress logging per archive and extraction phase; do not emit per-file logs for 10,000+ members.

### 4. Measuring and freezing SAR normalization statistics

- After the real download and adapter work, run `uv run python scripts\compute_band_statistics.py --dataset openearthmap-sar --batch-size 4 --num-workers 0` (adjust the batch downward only for memory).
- Compute over all 4,333 training SAR tiles only, at native 1024 x 1024 resolution and raw `uint8` magnitude. Do not include validation/test, RGB, remapped masks, or sample estimates.
- Replace temporary BandSpec placeholders with the measured mean/std/min/max. Retain raw min/max (expected but not assumed to be 0/255), require positive std, and add the concise provenance comment required by the contributor guide.
- Cross-check the script result with an independent streaming aggregate or a second worker configuration before freezing constants.
- Inspect every selected image/mask header while data is local and fail the implementation if dimensions, channel counts, dtypes, or label domain contradict the verified protocol.

### 5. Integrating honest dataset geography

- Add an explicit OpenEarthMap-SAR extraction route in `src/geo_eval_harness/geography.py`.
- Iterate the benchmark SAR images only, once per sample, across the fixed splits. Read CRS/affine headers without decoding pixel arrays.
- Reuse the existing CRS-grouping and reprojection-to-EPSG:4326 mechanism, extracting a shared raster-origin helper if the optical OpenEarthMap work has not already done so.
- Treat absent CRS/affine tags as unavailable coordinates, not malformed samples. Treat partially specified or invalid georeferencing as explicit read failures.
- Report `GeoRecord.n` as the number of genuinely geolocated SAR samples and use a truthful standalone version/source string such as `openearthmap-sar`.
- After the full-header scan, generate and commit `docs/_static/_dataset_geography/openearthmap-sar.json` and regenerate `index.json`. If valid tags unexpectedly exist in zero files, add a precise `NO_GEO` reason instead; representative planning inspection indicates at least some validation imagery is georeferenced.
- Add synthetic geography coverage containing one ungeoreferenced training TIFF and one projected/georeferenced validation TIFF, proving missing tags are skipped and valid tags are reprojected.

### 6. Adding focused offline and real-data tests

- Keep tests minimal and centered on logistics, public contracts, and logical flow rather than numerical precision or exhaustive input permutations.
- Add `tests/test_openearthmap_sar.py` using tiny real GeoTIFF fixtures under the official directory names. Use a small number of representative tests covering deterministic pairing/order, raw single-channel image and remapped mask output, band selection/transform flow, and one representative malformed-installation failure.
- Extend `tests/support/data.py` with a small writer for the official SAR directory layout and explicit standalone data-root detection.
- Extend `tests/test_download.py` with tiny in-memory ZIPs and mocked transport/digests. Prefer a compact end-to-end downloader-flow test across all four records plus focused tests for checksum failure, safe selective extraction, and incomplete-install repair; avoid separately testing every archive or filename permutation.
- Extend `tests/integration/test_download.py` with a local-HTTP CLI download followed by real adapter loading, without external network access.
- Add exact split counts to `tests/test_split_sizes.py::EXPECTED_SIZES`.
- Extend existing catalog, lazy-import, missing-data-command, band-selection, and dry-run parametrizations only where adding the dataset name exercises established shared tests. Do not create dataset-specific duplicates of generic behavior.
- Add at most one bounded offline segmentation integration case with tiny train/validation/test fixtures if existing generic segmentation coverage cannot exercise the adapter through the canonical runner. Check successful flow, eight-class metadata, ignored native-zero pixels, and finite outputs without asserting precise metric values or duplicating resume/caching grids.
- Add one focused geography-flow test proving that absent tags are skipped and valid projected tags are reprojected, then assert the committed geography store/index contains the new name.
- Build a wheel and confirm the new Python module/docs are packaged; no new split manifest is needed because membership is generated from fixed official numeric ranges.

### 7. Documenting the public dataset and protocol

- Update `docs/user/datasets.rst` with a standalone segmentation entry, download command, exact data layout, one SAR band, 4,333/210/490 counts, eight classes, native-zero ignore mapping, 1024-pixel/GSD facts, source records, and source-specific licensing caveat.
- State explicitly that training RGB is intentionally not installed/exposed and that the benchmark evaluates SAR only.
- State explicitly that geo-eval-harness ignores native-zero pixels during loss and metrics, unlike the official baseline which trains a ninth background output but excludes it from mIoU.
- Add `OpenEarthMapSAR` to `docs/api/datasets.rst` under a standalone segmentation section.
- Add a concise Unreleased changelog entry and update README download examples/supported-dataset wording where appropriate.
- Add `docs/examples/openearthmap-sar-run.yaml` using the strict existing schema, a segmentation-capable model/head, explicit `ignore_index: 255`, the `sar` band, and an explicit image size. Do not add a dataset-specific configuration engine.
- Keep the plan and public docs focused on current behavior; do not narrate implementation history.

### 8. Validating the complete integration

- Run focused adapter/downloader/catalog/geography tests first.
- Run the local-HTTP download integration and bounded offline segmentation integration.
- Run the real-data split-size test and a direct sample load for each split after the full verified download.
- Run band-statistics and geography extraction/check commands against the real installation.
- Run Ruff format/check and Pyrefly over changed code, then the relevant broader test suite if targeted failures expose shared behavior.
- Build and inspect the wheel.
- Do not require a full foundation-model sweep over all 5,033 native-resolution images as a completion criterion; require a bounded no-pretrained smoke run through the canonical CLI instead.

## Dependency order

1. Publish the approved plan.
2. Implement the adapter/registration and downloader in parallel against the same fixed path/split descriptor.
3. Add focused tests alongside each implementation surface.
4. Use the completed adapter plus verified downloader to compute final BandSpecs and full-population raster/header validation.
5. Geography can proceed from the completed adapter/downloader independently of BandSpec calculation.
6. Final public documentation depends on the finalized statistics and geography findings.
7. Complete validation depends on every prior surface.

## Suggested validation commands

Adapt test filenames only if the final organization differs:

```powershell
uv run pytest tests\test_openearthmap_sar.py tests\test_download.py tests\test_dataset_catalog.py tests\test_geography.py --no-cov
uv run pytest tests\integration\test_download.py tests\integration\test_segmentation.py -k openearthmap_sar --no-cov
uv run pytest tests\test_split_sizes.py -m slow -k openearthmap-sar --no-cov
uv run python scripts\compute_band_statistics.py --dataset openearthmap-sar --batch-size 4 --num-workers 0
uv run python scripts\extract_dataset_geography.py --dataset openearthmap-sar
uv run python scripts\extract_dataset_geography.py --check
uv run geo-eval-harness datasets openearthmap-sar
uv run geo-eval-harness run --config docs\examples\openearthmap-sar-run.yaml --dry-run
uv run ruff check <changed Python files>
uv run ruff format --check <changed Python files>
uv run pyrefly check
uv build --wheel
```

## Acceptance criteria

- `geo-eval-harness download openearthmap-sar` retrieves the four immutable Zenodo archives, verifies integrity, safely installs only SAR images and labels, and repairs incomplete installs.
- The dataset catalog and strict run configuration resolve `openearthmap-sar` lazily as an eight-class segmentation dataset.
- Train/validation/test expose exactly 4,333/210/490 deterministic paired samples in raw SAR units.
- Native labels are validated and mapped exactly as agreed: `0 -> 255`, `1..8 -> 0..7`.
- BandSpec values are measured from the full raw training split only.
- Geography honestly represents every sample with usable published georeferencing and fabricates none.
- A minimal offline test set covers the adapter, downloader, CLI integration, and geography workflow without network access or numerical-precision assertions.
- Real-data checks validate all downloaded files, split sizes, headers, and class domains.
- User/API docs, changelog, runnable example, geography store, and this plan are committed with the code.
- No new dependency, alternate config engine, RGB evaluation mode, runtime downloader, or unrelated benchmark sweep is introduced.

## Notes and boundaries

- Semantic segmentation only; UDA and image translation from the dataset reference repository are out of scope.
- SAR only; training RGB is out of scope.
- Official released validation/test labels are used. No hidden-label reconstruction or prediction-as-label fallback.
- No new generic metric exclusion feature: the chosen ignore/remap policy fits the existing solver.
- No changes to global result/resume identity logic are required because this is a new dataset name with a fixed protocol.
- Source-specific attribution and label licensing must remain visible.
- Do not bundle optical OpenEarthMap implementation into this change. Reuse already-landed helpers where appropriate, but keep both datasets independently reviewable.
- No unrequested commit or pull request creation.
