# torchgeo-bench

[![CI](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml/badge.svg)](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/torchgeo/torchgeo-bench/branch/main/graph/badge.svg)](https://codecov.io/gh/torchgeo/torchgeo-bench)
[![PyPI version](https://img.shields.io/pypi/v/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A lightweight benchmarking framework for evaluating **frozen** geospatial
foundation models on GeoBench V1/V2 and location encoders on CoordBench. Plug
in a backbone or coordinate encoder and run consistent downstream probes through
explicit CLI flags and strictly validated Pydantic YAML configuration.

- **Frozen-backbone evaluation** — KNN-5, L-BFGS logistic regression, and
  linear / conv / FPN / DPT segmentation probes.
- **GeoBench V1 + V2 built in** — classification and segmentation, RGB or
  full multispectral / multi-modal stacks.
- **CoordBench built in** — coordinate-only regression and classification with
  random, spatial-block, and official holdouts.
- **Config-driven** — sweep models, datasets, partitions, image sizes, and
  bands without code changes.
- **Resumable** — `--resume` skips already-computed `(dataset, method, model, …)` rows. Atomic CSV appends are safe across parallel jobs.
- **Bring your own model** — copy
  [`contrib_template.py`](src/torchgeo_bench/models/contrib_template.py),
  implement `_forward_patch_features`, and add a one-file model config.
  See the [Stage 1 guide](https://torchgeo.org/torchgeo-bench/user/eval_own_model.html)
  for a full walkthrough, or the
  [Stage 2 guide](https://torchgeo.org/torchgeo-bench/user/contribute_model.html)
  to contribute the model back upstream.

## Installation

```bash
pip install torchgeo-bench
```

For development:

```bash
git clone https://github.com/torchgeo/torchgeo-bench
cd torchgeo-bench
uv sync --extra dev
```

Requires Python 3.12+ and runs on **Linux**, **macOS**, and **Windows**. On
Linux x86_64 the install automatically includes GPU-accelerated FAISS KNN
(CUDA 12, driver R525+, also works on GPU-less machines); other platforms get
CPU FAISS.

## Download a dataset

The runner expects datasets under `./data/`. To grab GeoBench V1:

```bash
torchgeo-bench download geobench_v1
```

To download only the datasets needed for a run:

```bash
torchgeo-bench download geobench_v1 --datasets m-eurosat
```

V1 uses the pickle-free [JSON-metadata mirror](https://huggingface.co/datasets/calebrob6/geobenchv1-webdataset) under `data/classification_v1.0_wds/`, with a pinned revision and archive SHA-256 verification. Re-download existing pickle-based V1 caches; the readers no longer unpickle metadata.

V2 (classification + segmentation) and torchgeo's EuroSAT, RESISC45, and UC Merced downloaders work the same way, as do the AID and HOTOSM buildings Hugging Face rehosts (`torchgeo-bench download geobench_v2`, `torchgeo-bench download eurosat`, `torchgeo-bench download resisc45`, `torchgeo-bench download ucmerced`, `torchgeo-bench download hotosm_buildings`). See the [documentation](https://torchgeo.org/torchgeo-bench/user/datasets.html) for all options.

Hugging Face downloads (GeoBench V1 and V2, AID and HOTOSM buildings) use fixed commits and check every downloaded archive against a SHA-256 manifest in [`src/torchgeo_bench/checksums/`](src/torchgeo_bench/checksums). Missing, modified or, for GeoBench V2, unexpected archives are errors. Adding a pinned source takes a commit hash and one manifest file.

## Run a basic experiment

```bash
# Inspect available presets and datasets without loading ML dependencies
torchgeo-bench models
torchgeo-bench datasets

# A single dataset with a pretrained ImageNet ResNet-50
torchgeo-bench run --model timm/resnet50 --dataset m-eurosat
```

The default device is `cuda:0`. On a machine without a working CUDA GPU (or if
a GPU run crashes — see [troubleshooting](https://torchgeo.org/torchgeo-bench/user/troubleshooting.html)),
fall back to CPU:

```bash
torchgeo-bench run --model rcf --dataset m-eurosat --device cpu
```

Results are appended to `results/models/<model name>.csv`, which **ship pre-populated with reference results**. To start from a clean slate, pass `--output results/my_run.csv` or set `output.file` in a YAML file passed with `--config`. Re-run the same command with `--resume` to skip completed rows. Each evaluation is saved as soon as it finishes, so a later failure does not discard completed metrics.

```bash
# Linear-only probing, with explicit flags overriding the YAML
torchgeo-bench run --config docs/examples/image-run.yaml --methods linear --device cpu

# Validate selections without loading a model or dataset
torchgeo-bench run --model rcf --dataset m-eurosat --dry-run
```

See [`docs/examples/image-run.yaml`](docs/examples/image-run.yaml) for the image configuration
fields and `run --config-help` for the JSON schema. Omitted settings inherit
model and dataset defaults; explicit YAML values override them, and supplied
flags override YAML. This includes explicit `false`, `null`, and `[]`.

`torchgeo-bench`, `python -m torchgeo_bench`, and
`python -m torchgeo_bench.cli` expose the same commands. The old `key=value`
and `+key=value` overrides are **rejected**; migrate scripts to flags or YAML.
There is no separate legacy configuration entry point.

## Measure encoder cost

The standalone `profile` command measures one fixed **real dataset batch** and writes JSON to stdout by default. The `flops` command uses **synthetic inputs**, does not load dataset samples, and appends compute measurements to a CSV:

```bash
torchgeo-bench profile --model rcf --dataset m-eurosat --device cpu \
  --batch-size 8 --warmup 1 --measurements 5 > profile.json

torchgeo-bench flops --model rcf --device cpu --band-configs rgb \
  --seg-heads --output results/my_compute_cost.csv
```

`run`, `flops`, `coord`, and `profile` share `--output-dir` for a common output root and `--output` for an exact file path that takes precedence. Image runs retain their `models/`, `profiles/`, and `intrinsic_dim/` subdirectories. Standalone profiling writes `profile.json` beneath an explicit output root.

Both accept `--config` and `--dry-run`. See
[`docs/examples/profile.yaml`](docs/examples/profile.yaml),
[`docs/examples/flops.yaml`](docs/examples/flops.yaml), and the
[configuration reference](https://torchgeo.org/torchgeo-bench/user/configuration.html).
Optional `profile` and `intrinsic_dim` passes within an image run retain their
separate per-model CSVs unless `output.file` explicitly combines them.

## Handcrafted classification baseline

The handcrafted model adds deterministic spectral and spatial measurements to the ImageStats baseline. The `handcrafted_level1`, `handcrafted_level2`, and `handcrafted_level3` presets select cumulative feature sets; `handcrafted` defaults to level 2. Feature width depends on the input bands and available spectral indices.

Download the classification datasets and run the level sweep from this checkout:

```bash
python -m torchgeo_bench download geobench_v1
python -m torchgeo_bench download geobench_v2 --datasets benv2,treesatai,so2sat,forestnet
python -m torchgeo_bench download eurosat
python -m torchgeo_bench download resisc45
python -m torchgeo_bench download aid
python experiments/run_handcrafted.py
```

Future sweeps include ImageStats as level 0 and every classification protocol in the current dataset catalog, including AID, multilabel datasets and EuroSAT's spatial split. They use all bands with identity input normalization, otherwise retaining the normal 224px resize, KNN-5, validation-selected C, train-plus-validation final refit, and 200 bootstrap draws.

Results go to separate `results/models/handcrafted_level*.csv` files and `imagestats_handcrafted_control.csv`. Feature lists and completion status are saved under `outputs/handcrafted/`. Resume skips matching completed rows; a missing linear or KNN result is still reported as a failure. Summaries, including `--report-only`, select only rows matching the current configuration and requested devices, so historical hashes are not mixed into new comparisons.

The checked-in CSVs are unchanged historical measurements on 13 protocols from the original handcrafted study, not measurements of the current runner; they contain no AID results. Their recorded numbers and configuration hashes are preserved. Current runs use the current configuration hash and do not treat these historical rows as completed work. Use `--output-dir results/handcrafted-current` to keep a new sweep separate.

Use `--levels 1 2`, `--datasets eurosat resisc45`, or `--dry-run` for a smaller run. The extractor also works through the normal CLI:

```bash
python -m torchgeo_bench run --model handcrafted_level2 \
  --dataset eurosat --bands all --normalization none
```

All four presets default to all bands and identity normalization. Custom YAML can set `model: {name: handcrafted, kwargs: {level: 3}}`; preprocessing belongs under `input`, not constructor kwargs.

## CoordBench — location encoders

`torchgeo-bench coord` runs the **coordinate-only** track: point
`(lon, lat)` in, a downstream label out. Benchmarks are streamed directly from
the unified [`taylor-geospatial/coordbench`](https://huggingface.co/datasets/taylor-geospatial/coordbench)
HuggingFace dataset (PDFM, SatCLIP, SustainBench, CDC PLACES, MOSAIKS/USAVars,
the DeepMind/AlphaEarth suite, and more — no local download). A frozen encoder
is probed with **KNN** and a **ridge linear** head under **random** or
**spatial-block** cross-validation (regression → R², classification → accuracy).

```bash
# MIND on the full suite, using random and spatial cross-validation
torchgeo-bench coord --model mind --dataset all --split both

# One dataset family with the sine/cosine baseline and a linear probe
torchgeo-bench coord --model sincos --dataset pdfm --methods linear
```

`--model mind` and `--model mind-small` ([MIND](https://huggingface.co/isaaccorley/MIND),
distilled from AlphaEarth/Climplicit/GeoCLIP/SINR) and `--model sincos` work with
the base install. The other pretrained encoders (SatCLIP / GeoCLIP / Climplicit /
SINR, via `rshf`) need the `coordbench` extra:
`pip install "torchgeo-bench[coordbench]"`,
then `--model climplicit` (etc.). Results land in
`results/coordbench_results.csv`. Add your own encoder by subclassing
`LocationEncoder` (implement `_encode`) and setting `model.target` to its
importable Python name, with constructor options under `model.kwargs`. See the
[CoordBench guide](https://torchgeo.org/torchgeo-bench/user/coordbench.html)
and the runnable
[`FourierLocationEncoder` example](https://github.com/torchgeo/torchgeo-bench/blob/main/docs/examples/coordbench_location_encoder.py).

<!-- skip-on-docs-landing-start -->

## Learn more

- **[Documentation](https://torchgeo.org/torchgeo-bench/)** — full
  configuration reference, available models, dataset tables, multi-band
  experiments, evaluation methodology, output schema, dev / release
  workflow, and troubleshooting.
- **[AGENTS.md](https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md)**
  — contributor guide and house style.
- **[Cleanlab analysis](projects/cleanlab/README.md)** — standalone dataset auditing, probability extraction, and review galleries, with separate dependencies.

<!-- skip-on-docs-landing-end -->

## Citation

If you use this framework, please cite it (once the `torchgeo-bench` paper is
available):

```bibtex
@misc{torchgeobench,
  title  = {torchgeo-bench: A lightweight benchmarking framework for geospatial foundation models},
  author = {torchgeo-bench Contributors},
  year   = {TBD},
  note   = {Software},
  url    = {https://github.com/torchgeo/torchgeo-bench}
}
```

## License

[MIT](LICENSE).
