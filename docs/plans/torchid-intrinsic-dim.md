# Plan: Intrinsic Dimension (ID) metrics via `torchid`

Branch: `feat/torchid-intrinsic-dim`

## Goal

During embedding-based evaluation (KNN, linear probe), optionally compute
intrinsic dimension (ID) metrics on the extracted train/val/test features so we
can correlate ID across pretrained models × datasets with downstream
performance (accuracy / micro-mAP). Hypothesis: higher ID ⇒ better downstream.

## Dependency

- pkg: `torchid` (https://github.com/isaaccorley/torchid)
- install: `pip install torchid`
- API: sklearn-style `Estimator().fit(X).dimension_` (global) and
  `.dimension_pw_` (pointwise). GPU-accelerated, batched, `scikit-dimension`
  parity.
- Estimators:
  - global: `lPCA`, `TwoNN`, `MLE`, `CorrInt`, `MiND_ML`, `KNN`, `DANCo`,
    `FisherS`
  - local: `MOM`, `MADA`, `TLE`, `ESS`

## Integration points

Embeddings already live as `(x_train, y_train)`, `(x_val, ...)`, `(x_test, ...)`
in `src/geo_eval_harness/main.py:559-561` (classification path only — segmentation
has no global feature matrix and is **out of scope** for v1).

Touched files:
- `pyproject.toml` — add `torchid>=…` (pin after install + health check).
- `src/geo_eval_harness/conf/config.yaml` — new `eval.intrinsic_dim` block.
- `src/geo_eval_harness/intrinsic_dim.py` — new thin wrapper module.
- `src/geo_eval_harness/main.py` — call wrapper after `embed_split`, emit rows.
- `tests/test_intrinsic_dim.py` — unit tests on synthetic manifolds.

## Config (Hydra)

```yaml
eval:
  intrinsic_dim:
    enabled: false                  # off by default (extra compute)
    estimators: [TwoNN, MLE, lPCA]  # subset of torchid global estimators
    splits: [train]                 # which splits to compute on (train|val|test|all)
    max_samples: 10000              # subsample per split for speed; null = all
    device: ${device}               # reuse top-level device
    seed: ${seed}
```

Defaults aim cheap: 3 fast global estimators on train only, ≤10k pts.

## Wrapper module sketch

`src/geo_eval_harness/intrinsic_dim.py`:

- `ESTIMATORS: dict[str, type]` lazy import map → torchid classes.
- `compute_id(X, estimators, device, max_samples, seed) -> dict[str, float]`
  - subsample with seeded RNG if `len(X) > max_samples`
  - move to device tensor (float32)
  - for each name: `est = ESTIMATORS[name]().fit(X); out[name] = float(est.dimension_)`
  - on per-estimator failure: log warning, record `nan` (do not abort run).
- Return `{est_name: dim_value}`.

Keep wrapper <100 LOC. No pointwise / local estimators in v1 — scalar-per-split
keeps the schema flat.

## Result schema

Two options — pick A:

**A. New rows** (recommended; minimal schema churn):
- One row per (dataset, model, split, estimator) with
  `method = "intrinsic_dim"`, `metric_name = f"id_{estimator}_{split}"`,
  `metric_value = dim`, CI = 0, feature_dim populated, `best_c = None`.
- Pros: works with existing `EvaluationResult` + CSV append.
- Cons: many rows; downstream join needed for ID-vs-accuracy plots.

**B. Sidecar columns on existing knn5/linear rows.**
- Cons: changes `EvaluationResult` dataclass + breaks resume keys / older CSVs.

Going with **A**. Add `split` to `common_meta` only when emitting ID rows
(reuse a separate row-emit helper to avoid touching the dataclass — store
split inside `name` suffix or extend dataclass with optional `split: str | None
= None`). Decide during impl; lean toward adding `split` field defaulting None.

## Main loop change (sketch)

After line 562 (`feature_dim = x_train.shape[1]`):

```python
if cfg.eval.intrinsic_dim.enabled:
    from geo_eval_harness.intrinsic_dim import compute_id_for_splits
    id_rows = compute_id_for_splits(
        splits={"train": x_train, "val": x_val, "test": x_test},
        cfg=cfg.eval.intrinsic_dim,
        common_meta=common_meta,
        feature_dim=feature_dim,
        n_train=len(x_train), n_val=len(x_val), n_test=len(x_test),
    )
    all_rows.extend(id_rows)
```

ID computation is independent of knn/linear — runs once per (dataset, model)
even if both are skipped via resume? → only run when at least one of knn/linear
runs OR add separate `id_key` resume tracking. Simpler: gate ID on
`not (skip_knn and skip_linear)` for v1.

## Tests

`tests/test_intrinsic_dim.py`:
- Synthetic Swiss roll (true ID=2 in 3D) → `TwoNN`/`MLE` ≈ 2 ± 0.5.
- Uniform cube (true ID=d) → `lPCA` ≈ d.
- Subsampling determinism w/ fixed seed.
- Estimator failure path → nan + logged warning, run continues.

## Open questions

1. **Normalization before ID?** torchid expects raw float features; current
   embeddings are not L2-normalized. KNN uses L2 distance on raw, so match
   that — pass raw. Note this in module docstring.
2. **Pointwise/local estimators** (MADA, TLE) for per-class ID would be
   interesting (does ID per class predict per-class accuracy?). Defer to v2.
3. **Segmentation features.** Could compute ID on patch-token embeddings from
   the segmentation probe — separate follow-up; skip in v1.
4. **CUDA OOM** for large N×D. Cap via `max_samples`; document.

## Rollout

1. Add dep, wrapper, config, tests. Commit: `feat: add torchid intrinsic
   dimension metrics`.
2. Run on a small dataset/model pair, eyeball numbers vs. paper expectations.
3. Open PR. After merge, schedule a sweep across all (model, dataset) and
   produce a notebook plotting ID vs downstream accuracy.

## Non-goals (v1)

- Local/pointwise ID columns.
- ID for segmentation backbones.
- ID-aware model selection / training-loop integration.
