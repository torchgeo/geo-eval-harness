# CLS pooling comparisons

Batch 1 is complete: 262 configurations, 524 result rows, 98 explicit exclusions, and no partial results or OOM retries. Batch 2 is running; its comparison will be added after collection.

Scores below are percentages of the benchmark primary metric, averaged equally over matched configurations. CI wins count configurations where the winning pooling’s lower 95% confidence bound exceeds the other pooling’s upper bound. They are descriptive comparisons, not adjusted significance tests.

| Model | Method | Matched | Mean pooling | CLS pooling | CLS CI wins | Mean CI wins |
|---|---|---:|---:|---:|---:|---:|
| smarties/base | knn5 | 50 | 63.23 | 64.30 | 8 | 2 |
| smarties/base | linear | 50 | 79.84 | 78.84 | 0 | 7 |
| smarties/large | knn5 | 50 | 63.65 | 66.58 | 14 | 2 |
| smarties/large | linear | 50 | 80.06 | 79.60 | 0 | 8 |
| terrafm/base_s2 | knn5 | 14 | 77.72 | 79.74 | 6 | 0 |
| terrafm/base_s2 | linear | 14 | 86.50 | 87.19 | 1 | 2 |
| terratorch/clay_v1_5_cls | knn5 | 56 | 65.88 | 71.68 | 32 | 0 |
| terratorch/clay_v1_5_cls | linear | 56 | 81.84 | 82.00 | 0 | 0 |
| terratorch/clay_v1_cls | knn5 | 56 | 58.62 | 67.72 | 41 | 0 |
| terratorch/clay_v1_cls | linear | 56 | 79.42 | 79.64 | 0 | 0 |
| torchgeo/scalemae_large_fmow_cls | knn5 | 36 | 66.76 | 71.68 | 12 | 0 |
| torchgeo/scalemae_large_fmow_cls | linear | 36 | 81.78 | 81.97 | 0 | 0 |
