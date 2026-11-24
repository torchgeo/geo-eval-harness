## Add model: <!-- model name here -->

Use this template for PRs that add a new model and contribute its benchmark
rows to `results/models/<model name>.csv`.

Docs:

- [Evaluate your own model](https://torchgeo.org/geo-eval-harness/user/eval_own_model.html)
- [Contribute a model](https://torchgeo.org/geo-eval-harness/user/contribute_model.html)
- [Datasets](https://torchgeo.org/geo-eval-harness/user/datasets.html)
- [Results format](https://torchgeo.org/geo-eval-harness/user/results-format.html)

### 1. Model summary

| Field | Value |
|-------|-------|
| **Model name** | <!-- e.g. `new_model` as it appears in the results CSV --> |
| **Class** | <!-- e.g. `geo_eval_harness.models.NewModel` --> |
| **Model preset** | <!-- e.g. `src/geo_eval_harness/conf/model/new_model.yaml` --> |
| **Pretraining data** | <!-- e.g. Sentinel-2 global, fMoW, ImageNet, etc. --> |
| **Sensor coverage** | <!-- e.g. Sentinel-2 RGB, Sentinel-2 multispectral, aerial RGB --> |
| **Weights URL** | <!-- Hugging Face Hub repo, release asset, or equivalent public URL --> |
| **Paper / project page** | <!-- URL if available, otherwise N/A --> |
| **Required extra** | <!-- e.g. `newmodel`, or N/A if no new extra is needed --> |

### 2. Add the model

Follow the Stage 2 guide: [Contribute a model](https://torchgeo.org/geo-eval-harness/user/contribute_model.html).

- [ ] Class inherits `BenchModel` and implements `_forward_patch_features(images) -> (B, K)`.
- [ ] Class is exported from `src/geo_eval_harness/models/__init__.py` and listed in `__all__`.
- [ ] Model preset exists at `src/geo_eval_harness/conf/model/<name>.yaml` with the correct `target`, result `name`, and constructor `kwargs`.
- [ ] The wrapper honors the requested normalization and input BandSpecs.
- [ ] Model weights are publicly accessible without authentication.
- [ ] Optional dependencies are declared under `[project.optional-dependencies]` in `pyproject.toml`, if needed.
- [ ] Tests cover all added code in `tests/test_<model>.py`.
- [ ] Fast tests use random tensors and no network I/O.
- [ ] Weight-download tests are marked `@pytest.mark.slow`.

### 3. Run the model on every dataset

Set up the environment. If the model needs an optional extra, include it in the
same sync command as `dev`.

```bash
uv sync --extra dev
# uv sync --extra dev --extra <required_extra>
```

Download the benchmark data before running the model. These commands fetch
the full V1 and V2 sets; use `--datasets m-eurosat` on the V1 command for a
quick trial.

```bash
uv run geo-eval-harness download geobench_v1
uv run geo-eval-harness download geobench_v2
uv run geo-eval-harness download eurosat
```

Create a run YAML listing the applicable datasets and the exact bands, normalization, and device used. Use `--resume` so an interrupted run can continue without duplicating completed rows:

```yaml
model:
  name: <model_config_name>
datasets: [m-eurosat, m-pv4ger]  # Replace with every applicable dataset.
input:
  bands: rgb
  normalization: dataset
runtime:
  device: cuda:0
```

Save this as `new_model_run.yaml`, including any additional settings needed for the model:

```bash
uv run geo-eval-harness run --config new_model_run.yaml --dry-run
uv run geo-eval-harness run --config new_model_run.yaml --resume
```

If a dataset cannot run because the model does not support that sensor or
modality, list it here with the exact error or reason. Otherwise write `None`.

| Skipped dataset | Reason |
|-----------------|--------|
| <!-- e.g. kuro_siwo --> | <!-- e.g. Model needs color images; this dataset provides radar and elevation. --> |

### 4. Commit results

The goal is for maintainers to review and merge the model code and its
benchmark rows together. Each model writes to its own
`results/models/<model name>.csv`, so your PR should add exactly one new
results file and leave every other model's file untouched.

- [ ] New rows are committed to `results/models/<model name>.csv`.
- [ ] Added result rows are only for this model and the command above.
- [ ] No other model's results file was added, edited, or removed.
- [ ] Result rows match the documented [CSV schema](https://torchgeo.org/geo-eval-harness/user/results-format.html).
- [ ] Added row count: <!-- e.g. 42 rows -->

### 5. Reproduction details

Maintainers should be able to check out this PR and rerun the exact benchmark.

```bash
git checkout <this-branch>
uv sync --extra dev
uv run geo-eval-harness run --config new_model_run.yaml --resume
```

Hardware and software used for submitted results:

| Field | Value |
|-------|-------|
| **Commit SHA** | <!-- `git rev-parse HEAD` --> |
| **OS** | <!-- e.g. Ubuntu 24.04 --> |
| **CPU/GPU** | <!-- e.g. A100 80GB, M3 Max, CPU-only --> |
| **Python** | <!-- `python --version` --> |
| **geo-eval-harness command** | <!-- paste the exact command used --> |

### 6. Local checks

- [ ] `uv run pytest --no-cov tests/test_<model>.py` passes locally.
- [ ] `uv run pytest --no-cov -m slow tests/test_<model>.py` passes locally, if pretrained weights were added.
- [ ] `uv run pytest --no-cov` passes locally.
- [ ] `uv run ruff check . && uv run ruff format --check .` passes locally.
