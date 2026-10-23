# Cleanlab dataset analysis

Standalone label-quality analysis using frozen-model linear-probe probabilities from `geo-eval-harness`. Run all commands below from the repository root with Python 3.12 or newer.

## Installation

Install the core library separately, then the analysis dependencies:

```bash
python -m pip install -e .
python -m pip install -r projects/cleanlab/requirements.txt
```

Models that need optional dependencies require the corresponding extra, such as `geo-eval-harness[olmoearth]`, or TerraTorch's separate [source installation](../../docs/user/installation.rst). Datasets use the usual `data/` locations; see the [repository README](../../README.md) for downloads.

## Workflow

Start with existing linear-probe benchmark results in `results/models/`. Extract probabilities once per dataset, selecting the best compatible linear-probe row:

```bash
python projects/cleanlab/cleanlab_extract_probs.py --dataset m-eurosat --verbose
```

The extractor accepts `--results results/all_results.csv` for a legacy combined CSV, or another results directory. It fits the probe on train+val at the recorded `best_c` and saves `results/cleanlab/probs/<dataset>__<model>_{train,test}.npz`. Existing train/test artifact pairs are skipped unless `--force` is supplied. Use `--device cpu`, `--batch-size`, and `--num-workers` as needed.

Reuse saved probabilities to produce per-sample issue scores, aggregate summaries, and per-class reports:

```bash
python projects/cleanlab/run_cleanlab_audit.py --verbose
python projects/cleanlab/cleanlab_per_class_singlelabel.py --splits test
python projects/cleanlab/cleanlab_per_class_multilabel.py --splits test
```

These commands write to `results/cleanlab/`. The per-class scripts accept `--datasets` to select datasets; all three accept `--probs-dir` and `--out-dir`. No extraction is needed if compatible probability artifacts already exist.

For manual review and cross-dataset overlap checks:

```bash
python projects/cleanlab/render_flagged_gallery.py --splits test --top-k 50
python projects/cleanlab/eurosat_family_dedup.py --verbose
```

Galleries are saved under `results/cleanlab/galleries/`. EuroSAT-family hashes and collision groups are saved to `results/cleanlab/dedup_eurosat_family.csv` and `results/cleanlab/dedup_eurosat_family_collisions.csv`. These image-based steps need the datasets on disk; they do not modify them.

## Artifacts and interpretation

NPZ artifacts contain numeric `indices`, `labels`, `probs`, and `classes`, plus Unicode `meta` strings. Any additional textual fields, such as `sample_ids`, must also be string arrays, not object arrays. Consumers always load with `allow_pickle=False` and reject object-backed required arrays. Unused legacy metadata is not read, so numeric artifacts with old object metadata remain usable without unpickling it; model names come from filenames.

Train probabilities are in-sample, so their flag rates underestimate train noise. Test probabilities are out-of-sample; flags still require manual review and are not confirmed label errors. Multi-label aggregate flag rates can be inflated by class imbalance and the union of per-class flags. See the [historical GeoBench audit report](cleanlab_audit_geobench.md) for the original findings and limitations.
