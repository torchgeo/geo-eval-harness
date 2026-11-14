# geo-eval-harness

[![CI](https://github.com/torchgeo/geo-eval-harness/actions/workflows/ci.yaml/badge.svg)](https://github.com/torchgeo/geo-eval-harness/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/torchgeo/geo-eval-harness/branch/main/graph/badge.svg)](https://codecov.io/gh/torchgeo/geo-eval-harness)
[![PyPI version](https://img.shields.io/pypi/v/geo-eval-harness.svg)](https://pypi.org/project/geo-eval-harness/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/geo-eval-harness.svg)](https://pypi.org/project/geo-eval-harness/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Evaluate geospatial image models and location encoders with classification, segmentation, and coordinate prediction tasks. Configure runs with CLI flags or validated YAML.

- Image classification uses frozen embeddings with KNN-5 or logistic regression.
- Segmentation supports frozen backbones, LoRA, and full fine-tuning with linear, convolutional, FPN, DPT, UPerNet, and U-Net heads.
- CoordBench evaluates location encoders with random, spatial-block, and official holdouts.
- Runs append results to CSV files. `--resume` skips completed evaluations with matching configurations.
- Custom image models implement `BenchModel._forward_patch_features`; custom location encoders implement `LocationEncoder._encode`. See the [custom model guide](https://torchgeo.org/geo-eval-harness/user/eval_own_model.html) and [contribution guide](https://torchgeo.org/geo-eval-harness/user/contribute_model.html).

## Installation

```bash
pip install geo-eval-harness
```

For development:

```bash
git clone https://github.com/torchgeo/geo-eval-harness
cd geo-eval-harness
uv sync --extra dev
```

Requires Python 3.12+ and runs on **Linux**, **macOS**, and **Windows**. On
Linux x86_64 the install automatically includes GPU-accelerated FAISS KNN
(CUDA 12, driver R525+, also works on GPU-less machines); other platforms get
CPU FAISS.

The Python package is `geo_eval_harness`; use `import geo_eval_harness` and `python -m geo_eval_harness`. TerraTorch models require the separate [source installation](https://torchgeo.org/geo-eval-harness/user/installation.html#terratorch-models).

## Download a dataset

The runner expects datasets under `./data/`. To grab GeoBench V1:

```bash
geo-eval-harness download geobench_v1
```

To download only the datasets needed for a run:

```bash
geo-eval-harness download geobench_v1 --datasets m-eurosat
```

V1 uses the [JSON-metadata mirror](https://huggingface.co/datasets/calebrob6/geobenchv1-webdataset) under `data/classification_v1.0_wds/`, with a pinned revision and archive SHA-256 verification.

Download GeoBench V2 or an individual dataset by its catalog name:

```bash
geo-eval-harness download geobench_v2
geo-eval-harness download eurosat
geo-eval-harness download openearthmap-sar
```

Hugging Face archive downloads use pinned commits and SHA-256 manifests in [`src/geo_eval_harness/checksums/`](src/geo_eval_harness/checksums). See the [dataset guide](https://torchgeo.org/geo-eval-harness/user/datasets.html) for download sources, splits, bands, and licenses.

## Run a basic experiment

```bash
# Inspect available presets and datasets without loading ML dependencies
geo-eval-harness models
geo-eval-harness datasets

# A single dataset with a pretrained ImageNet ResNet-50
geo-eval-harness run --model timm/resnet50 --dataset m-eurosat
```

The default device is `cuda:0`. Select CPU with `--device cpu`:

```bash
geo-eval-harness run --model rcf --dataset m-eurosat --device cpu
```

Results are appended to `results/models/<model name>.csv`. The repository includes reference results; installing the package does not download them. Use `--output results/my_run.csv` or `output.file` in YAML to select a separate file. Re-run with `--resume` to skip completed rows whose configuration matches. Each evaluation is saved as it finishes.

```bash
# Linear-only probing, with explicit flags overriding the YAML
geo-eval-harness run --config docs/examples/image-run.yaml --methods linear --device cpu

# Validate selections without loading a model or dataset
geo-eval-harness run --model rcf --dataset m-eurosat --dry-run
```

The `docs/examples/` paths refer to a repository checkout. For an installed package, save a linked example locally and pass its path to `--config`. See [`image-run.yaml`](docs/examples/image-run.yaml) for an image configuration and `run --config-help` for the JSON schema. Omitted settings inherit
model and dataset defaults; explicit YAML values override them, and supplied
flags override YAML. This includes explicit `false`, `null`, and `[]`.

`geo-eval-harness`, `python -m geo_eval_harness`, and `python -m geo_eval_harness.cli` expose the same commands.

## Measure encoder cost

The standalone `profile` command measures one fixed **real dataset batch** and writes JSON to stdout by default. The `flops` command uses **synthetic inputs**, does not load dataset samples, and appends compute measurements to a CSV:

```bash
geo-eval-harness profile --model rcf --dataset m-eurosat --device cpu \
  --batch-size 8 --warmup 1 --measurements 5 > profile.json

geo-eval-harness flops --model rcf --device cpu --band-configs rgb \
  --seg-heads --output results/my_compute_cost.csv
```

`run`, `flops`, `coord`, and `profile` share `--output-dir` for a common output root and `--output` for an exact file path that takes precedence. Image runs retain their `models/`, `profiles/`, and `intrinsic_dim/` subdirectories. Standalone profiling writes `profile.json` beneath an explicit output root.

Both accept `--config` and `--dry-run`. See
[`docs/examples/profile.yaml`](docs/examples/profile.yaml),
[`docs/examples/flops.yaml`](docs/examples/flops.yaml), and the
[configuration reference](https://torchgeo.org/geo-eval-harness/user/configuration.html).
Optional `profile` and `intrinsic_dim` passes within an image run retain their
separate per-model CSVs unless `output.file` explicitly combines them.

## Handcrafted classification baseline

The handcrafted model adds deterministic spectral and spatial measurements to the ImageStats baseline. The `handcrafted_level1`, `handcrafted_level2`, and `handcrafted_level3` presets select cumulative feature sets; `handcrafted` defaults to level 2. Feature width depends on the input bands and available spectral indices.

Download the classification datasets and run the level sweep from this checkout:

```bash
python -m geo_eval_harness download geobench_v1
python -m geo_eval_harness download geobench_v2 --datasets benv2,treesatai,so2sat,forestnet
python -m geo_eval_harness download eurosat
python -m geo_eval_harness download resisc45
python -m geo_eval_harness download aid
python experiments/run_handcrafted.py
```

The sweep includes ImageStats as level 0 and every classification protocol in the current dataset catalog, including AID, multilabel datasets and EuroSAT's spatial split. It uses all bands with identity input normalization, with 224-pixel resizing, KNN-5, validation-selected C, train-plus-validation final refit, and 200 bootstrap draws.

Results go to `results/models/handcrafted_level*.csv` and `imagestats_handcrafted_control.csv`. Feature lists and completion status are saved under `outputs/handcrafted/`. Resume skips matching completed rows; a missing linear or KNN result is reported as a failure. Summaries, including `--report-only`, select rows matching the requested configuration and devices. Use `--output-dir results/handcrafted-current` to keep a sweep separate.

Use `--levels 1 2`, `--datasets eurosat resisc45`, or `--dry-run` for a smaller run. The extractor also works through the normal CLI:

```bash
python -m geo_eval_harness run --model handcrafted_level2 \
  --dataset eurosat --bands all --normalization none
```

All four presets default to all bands and identity normalization. Custom YAML can set `model: {name: handcrafted, kwargs: {level: 3}}`; preprocessing belongs under `input`, not constructor kwargs.

## CoordBench — location encoders

`geo-eval-harness coord` runs the **coordinate-only** track: point
`(lon, lat)` in, a downstream label out. Benchmarks are streamed directly from
the unified [`taylor-geospatial/coordbench`](https://huggingface.co/datasets/taylor-geospatial/coordbench)
HuggingFace dataset (PDFM, SatCLIP, SustainBench, CDC PLACES, MOSAIKS/USAVars,
the DeepMind/AlphaEarth suite, and more — no local download). A frozen encoder
is probed with **KNN** and a **ridge linear** head under **random** or
**spatial-block** cross-validation (regression → R², classification → accuracy).

```bash
# MIND on the full suite, using random and spatial cross-validation
geo-eval-harness coord --model mind --dataset all --split both

# One dataset family with the sine/cosine baseline and a linear probe
geo-eval-harness coord --model sincos --dataset pdfm --methods linear
```

`--model mind` and `--model mind-small` ([MIND](https://huggingface.co/isaaccorley/MIND),
distilled from AlphaEarth/Climplicit/GeoCLIP/SINR) and `--model sincos` work with
the base install. The other pretrained encoders (SatCLIP / GeoCLIP / Climplicit /
SINR, via `rshf`) need the `coordbench` extra:
`pip install "geo-eval-harness[coordbench]"`,
then `--model climplicit` (etc.). Results land in
`results/coordbench_results.csv`. Add your own encoder by subclassing
`LocationEncoder` (implement `_encode`) and setting `model.target` to its
importable Python name, with constructor options under `model.kwargs`. See the
[CoordBench guide](https://torchgeo.org/geo-eval-harness/user/coordbench.html)
and the runnable
[`FourierLocationEncoder` example](https://github.com/torchgeo/geo-eval-harness/blob/main/docs/examples/coordbench_location_encoder.py).

<!-- skip-on-docs-landing-start -->

## Learn more

- **[Documentation](https://torchgeo.org/geo-eval-harness/)** — full
  configuration reference, available models, dataset tables, multi-band
  experiments, evaluation methodology, output schema, dev / release
  workflow, and troubleshooting.
- **[AGENTS.md](https://github.com/torchgeo/geo-eval-harness/blob/main/AGENTS.md)**
  — contributor guide and house style.
- **[Cleanlab analysis](projects/cleanlab/README.md)** — standalone dataset auditing, probability extraction, and review galleries, with separate dependencies.

<!-- skip-on-docs-landing-end -->

## Citation

To cite the software:

```bibtex
@misc{geoevalharness,
  title  = {geo-eval-harness: A lightweight benchmarking framework for geospatial foundation models},
  author = {geo-eval-harness Contributors},
  year   = {2026},
  note   = {Software},
  url    = {https://github.com/torchgeo/geo-eval-harness}
}
```

## License

[MIT](LICENSE).
