# Benchmark Results

This directory holds the checked-in benchmark outputs for **geo-eval-harness**. Files are organized into two categories: canonical benchmark results consumed by the leaderboard and results explorer, and historical one-off study outputs.

For full `EvaluationResult` column schemas, normalization strategy names, method values, and resume-key semantics, see [docs/user/results-format.rst](../docs/user/results-format.rst). For CoordBench evaluation details, see [docs/user/coordbench.rst](../docs/user/coordbench.rst).

## Canonical Benchmark Results

These files are the up-to-date benchmark data read by `scripts/regen_leaderboard.py`, `scripts/regen_results_explorer.py`, and CI checks.

| Path | Description |
| --- | --- |
| `models/<model>.csv` | Per-model classification (`knn5`, `linear`, and `-std` variants) and segmentation (`seg-*`) benchmark metrics. Each model writes to its own CSV so re-running one model only touches that file. |
| `compute_cost.csv` | Per-sample compute cost measurements (GFLOPs, parameter counts, throughput, and memory) produced by `geo-eval-harness flops` and `experiments/run_compute_cost_sweep.py`. |
| `compute_cost.md`, `compute_cost_coverage.csv`, `compute_cost_provenance.json` | Companion protocol notes, requested-cell coverage matrix, and hardware/software provenance metadata for `compute_cost.csv`. |
| `intrinsic_dim/<model>.csv` | Per-model intrinsic-dimension estimates and centered feature-spectrum diagnostics (`method=intrinsic_dim`). |
| `profiles/<model>.csv` | Per-model runtime profile rows (`method=profile`). |
| `coordbench_results.csv` | Canonical CoordBench location-encoder benchmark results produced by `geo-eval-harness coord`. |

## Historical One-Off Studies

The following top-level files and directories are outputs from standalone experiments in `experiments/`. They are retained for reference and are not consumed by the leaderboard, results explorer, or CI.

| Path | Produced by | Purpose |
| --- | --- | --- |
| `c_sweep_experiment.csv` | `experiments/run_c_sweep_experiment.py` | Linear-probe regularization (`C`) sensitivity study. |
| `cls_token_experiment.csv` | `experiments/run_cls_token_experiment.py` | Comparison of transformer CLS token vs. mean-pooled patch features. |
| `effect_of_lbfgs_vs_adam.csv` | `experiments/run_effect_of_lbfgs_vs_adam.py` | Comparison of L-BFGS vs. Adam fit time and accuracy for linear probing. |
| `segmentation_protocol/metadata.json`, `segmentation_protocol_study.csv`, `segmentation_protocol_study_selected.csv`, `segmentation_protocol_study_summary.csv` | `experiments/run_segmentation_protocol_study.py` and `experiments/analyze_segmentation_protocol_study.py` | Segmentation training-protocol comparison study. |
| `segmentation_representative_rgb_all_224_10ep.csv`, `segmentation_representative_rgb_all_224_10ep.csv.sweep.json` | `experiments/run_segmentation_representative_sweep.py` | 10-epoch representative segmentation sweep across models, heads, and band selections. |
