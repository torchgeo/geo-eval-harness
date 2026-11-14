# Segmentation compute (`seg-flops`)

`geo-eval-harness seg-flops` records the per-sample FLOPs and parameters of segmentation probes in
`results/compute_cost_segmentation.csv`. It covers the backbone, neck, decoder and upsampler.
Inputs are synthetic, and no dataset samples are loaded.

The classification table `results/compute_cost.csv` (`geo-eval-harness flops`) is unchanged. Its
old segmentation sweep counted the head only, for FPN/DPT heads at CloudSen12 shapes. That sweep is
deprecated, off by default, and its rows have been removed.

## Counting convention

Counts use `torch.utils.flop_counter.FlopCounterMode`, the same convention as classification:

- One multiply-add counts as two FLOPs.
- Only matrix products, convolutions and attention are counted.
- Element-wise operations, normalization, pooling, softmax, interpolation (including the bilinear
  upsampling to full resolution inside decoders) and optimizer updates count as zero.

`seg-flops` adds formulas for operators that torch's registry lacks:

- CPU flash attention (`_scaled_dot_product_flash_attention_for_cpu`). The stock counter silently
  counts zero for it, so CPU rows of attention models undercount without this formula.
- `nn.MultiheadAttention`'s inference fast path.
- `nn.TransformerEncoderLayer`'s fused inference path.
- Fused `addmm` + activation.

Tests check that each formula equals the decomposed (training-mode) count.

**Coverage audit.** Every operator without a formula is recorded. If one looks like matrix work,
the row gets `flop_coverage = lower_bound` and the operator names go to `uncounted_ops`. Such ops
are matmul/convolution/attention names, or any operator from a non-`aten` namespace such as custom
CUDA, Triton or xformers kernels. Otherwise the row gets `flop_coverage = ok`.

Counts are per sample at batch size 1, with AMP autocast on CUDA as in the harness. Prefer CUDA:
`device` records where each row was measured. Throughput, latency and memory are not measured. The
`throughput_*`, `latency_*`, `peak_gpu_mem_gb` and `timing_*` columns are reserved for a separate
run on reference hardware. FLOPs and parameters do not depend on hardware.

## Cells and keys

A cell is one dataset plus a partial `run` configuration (`model`, `input`, `segmentation`). It is
resolved exactly as `geo-eval-harness run` resolves it: model preset, then dataset overrides, then
explicit settings. As a result, the following match the run:

- default segmentation layers;
- dataset band list;
- class count;
- constructor kwargs.

`compute_key` hashes everything that determines the evaluated network:

- preset target and name, and constructor kwargs (checkpoint paths reduced to the file name);
- dataset and its resolved band list;
- image size and time steps;
- class count;
- decoder, neck and `neck_norm`, upsampler;
- tapped layers, sweep heads, feature standardization;
- `compute_schema`.

Learning rates, epochs, seeds, dropout, normalization and the adaptation mode are excluded.
LoRA adapters are merged for validation and test (`merged_lora`), so a LoRA run evaluates the same
network as a frozen one.

`train_key` adds the adaptation (`frozen` / `lora` / `full`) and the LoRA rank, target modules and
last-block limit.

## Row kinds

| `kind` | one row per | content |
|---|---|---|
| `inference` | `compute_key` | evaluation forward, split into `gflops_backbone`, `gflops_neck`, `gflops_decoder`, `gflops_upsampler` (AnyUp) and `gflops_other`, which sum to `gflops_total`. Also `gflops_sdpa` (attention kernels), `params_*_m`, and the measured taps (`tap_channels`, `tap_hw`, `n_tokens` = deepest tap's grid). |
| `train_unit` | `train_key` | one training step as the solver runs it, per sample: `gflops_train_forward`, `gflops_train_forward_backbone`, `gflops_train_backward`, `gflops_train_step`, `params_trainable_m` |
| `canonical` | model × band config × arm | leaderboard summary at 224 px on CloudSen12 shapes (`rgb`, `s2` = all twelve bands). The arms are `upernet`, `unet`, `upernet+vit`, `unet+vit` and `upernet+anyup`. Combinations a model does not support are skipped. |

**Backbone attribution.** The backbone runs as executed: the full wrapper forward, including blocks
after the deepest tapped layer. The neck is the ViT neck; the upsampler is frozen AnyUp; the decoder
is the rest of the head, including the classifier. Operators are attributed by forward hooks to the
innermost running component.

**Training step.** The solver's training step comprises:

- train mode (frozen and LoRA backbones stay in eval mode, except the adapters);
- AMP;
- cross-entropy loss;
- backward.

A frozen backbone runs without gradients, so its backward is head-only. Unmerged LoRA adds its
low-rank products to the forward.

## Training budget per run

The total compute of one result follows `evaluate_segmentation`. It is computed when a registry
joins the measured rows, with `segmentation_compute.training_budget`:

```
E = epochs actually run (early stopping included)
frozen, cache_features=true:
    encode   = backbone × (n_train + n_val + n_test)        # one pass over each split
    fit      = (train_step − train_forward_backbone) × n_train × E
    evaluate = (total − backbone) × (n_val × E + n_test)
otherwise (uncached frozen, LoRA, full):
    encode   = backbone × n_train  if frozen with feature_standardize, else 0
    fit      = train_step × n_train × E
    evaluate = total × (n_val × E + n_test)
```

Not counted:

- optimizer updates;
- skipped undersized batches;
- recomputation from gradient checkpointing;
- data loading.

## Usage

```bash
# A manifest of cells (JSON Lines or a YAML list of {dataset, run, label})
geo-eval-harness seg-flops --manifest cells.jsonl --device cuda

# Keys only, for joining results without a GPU
geo-eval-harness seg-flops --manifest cells.jsonl --keys-only > keys.jsonl

# Canonical leaderboard grid
geo-eval-harness seg-flops --canonical-models timm/resnet50 timm/vit/vit_base_patch16_224 \
  --kinds canonical --device cuda
```

See [`docs/examples/seg-flops.yaml`](../examples/seg-flops.yaml). The output resumes by key:
rerunning skips rows already present. A failing cell is logged, and the command continues and exits
with status 1. Use `--fail-fast` to stop at the first failure instead.

## Caveats

- **Code drift.** Rows describe the network at the measured commit (`source_commit`). Results from
  older commits are comparable only where the model wrapper and preset did not change in a
  FLOP-relevant way.
- **Data-dependent control flow.** Compute is input-independent for every benchmark model except
  DOFA, whose wavelength-conditioned patch embedding varies by under 0.5% with batch size.
- **Native-size presets** (`image_size: null`) need an explicit size.
