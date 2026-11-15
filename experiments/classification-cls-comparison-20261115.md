# CLS pooling comparisons

Both batches are complete: 486 configurations, 972 result rows, 114 explicit exclusions, and no partial results or failed jobs. No OOM retries occurred; every effective hash and batch size matches its planned value. All existing mean and segmentation rows are preserved.

## Coverage

Planned counts include the 60 dataset/band/normalization combinations per model before compatibility exclusions. Each complete configuration contributes one KNN and one linear result.

| Model | Planned | Complete | Partial | Excluded | Exclusion reason |
|---|---:|---:|---:|---:|---|
| terratorch/clay_v1_cls | 60 | 56 | 0 | 4 | 4: Native normalization rejects the selected mixture of optical and SAR input scales. |
| terratorch/clay_v1_5_cls | 60 | 56 | 0 | 4 | 4: Native normalization rejects the selected mixture of optical and SAR input scales. |
| torchgeo/scalemae_large_fmow_cls | 60 | 36 | 0 | 24 | 24: Scale-MAE FMOW RGB weights require exactly three ordered RGB channels; the all-band input contains additional channels. |
| smarties/base | 60 | 50 | 0 | 10 | 8: SMARTIES requires a single sensor; the all-band input combines Sentinel-1 and Sentinel-2.; 2: SMARTIES native normalization only declares pretrained statistics for 8-bit RGB; this sensor/band selection requires dataset normalization. |
| smarties/large | 60 | 50 | 0 | 10 | 8: SMARTIES requires a single sensor; the all-band input combines Sentinel-1 and Sentinel-2.; 2: SMARTIES native normalization only declares pretrained statistics for 8-bit RGB; this sensor/band selection requires dataset normalization. |
| terrafm/base_s2 | 60 | 14 | 0 | 46 | 46: The selected inputs lack a required pretrained model band; zero-fill ablations are outside this sweep. |
| terratorch/prithvi_eo_v2_100_tl_cls | 60 | 56 | 0 | 4 | 4: Native normalization rejects the selected mixture of optical and SAR input scales. |
| terratorch/prithvi_eo_v2_300_cls | 60 | 56 | 0 | 4 | 4: Native normalization rejects the selected mixture of optical and SAR input scales. |
| terratorch/prithvi_eo_v2_300_tl_cls | 60 | 56 | 0 | 4 | 4: Native normalization rejects the selected mixture of optical and SAR input scales. |
| terratorch/prithvi_eo_v2_600_cls | 60 | 56 | 0 | 4 | 4: Native normalization rejects the selected mixture of optical and SAR input scales. |

## Matched scores

Scores below are percentages of the benchmark primary metric, averaged equally over matched configurations. CI wins count configurations where the winning pooling’s lower 95% confidence bound exceeds the other pooling’s upper bound. They are descriptive comparisons, not adjusted significance tests.

| Model | Method | Matched | Mean pooling | CLS pooling | CLS CI wins | Mean CI wins |
|---|---|---:|---:|---:|---:|---:|
| terratorch/clay_v1_cls | knn5 | 56 | 58.62 | 67.72 | 41 | 0 |
| terratorch/clay_v1_cls | linear | 56 | 79.42 | 79.64 | 0 | 0 |
| terratorch/clay_v1_5_cls | knn5 | 56 | 65.88 | 71.68 | 32 | 0 |
| terratorch/clay_v1_5_cls | linear | 56 | 81.84 | 82.00 | 0 | 0 |
| torchgeo/scalemae_large_fmow_cls | knn5 | 36 | 66.76 | 71.68 | 12 | 0 |
| torchgeo/scalemae_large_fmow_cls | linear | 36 | 81.78 | 81.97 | 0 | 0 |
| smarties/base | knn5 | 50 | 63.23 | 64.30 | 8 | 2 |
| smarties/base | linear | 50 | 79.84 | 78.84 | 0 | 7 |
| smarties/large | knn5 | 50 | 63.65 | 66.58 | 14 | 2 |
| smarties/large | linear | 50 | 80.06 | 79.60 | 0 | 8 |
| terrafm/base_s2 | knn5 | 14 | 77.72 | 79.74 | 6 | 0 |
| terrafm/base_s2 | linear | 14 | 86.50 | 87.19 | 1 | 2 |
| terratorch/prithvi_eo_v2_100_tl_cls | knn5 | 56 | 59.16 | 63.38 | 31 | 0 |
| terratorch/prithvi_eo_v2_100_tl_cls | linear | 56 | 78.00 | 77.00 | 0 | 6 |
| terratorch/prithvi_eo_v2_300_cls | knn5 | 56 | 59.06 | 64.77 | 33 | 0 |
| terratorch/prithvi_eo_v2_300_cls | linear | 56 | 79.01 | 78.83 | 0 | 0 |
| terratorch/prithvi_eo_v2_300_tl_cls | knn5 | 56 | 59.47 | 64.42 | 30 | 0 |
| terratorch/prithvi_eo_v2_300_tl_cls | linear | 56 | 78.54 | 77.94 | 1 | 3 |
| terratorch/prithvi_eo_v2_600_cls | knn5 | 56 | 59.11 | 65.27 | 33 | 0 |
| terratorch/prithvi_eo_v2_600_cls | linear | 56 | 79.65 | 79.56 | 0 | 1 |

## Collection integrity

The model CSVs contain 12,210 classification rows (11,238 mean-refresh rows plus 972 CLS rows) and 11,102 segmentation rows. All 148 pre-companion CSV prefixes retain their exact bytes. The repository’s canonical deduplication key, including pooling and segmentation setup, has no duplicate keys in any file.

The narrower key `(model, dataset, method, bands, normalization, metric_name, seed)` does not distinguish pooling or segmentation setups: it reports 6,977 excess segmentation rows and 228 intentional mean/CLS collisions across 64 files. These are distinct measurements under the canonical key; no segmentation rows are changed. ImageStats has 60 rows and no duplicate keys under either check.
