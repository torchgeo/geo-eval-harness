---
orphan: true
---

# Interesting findings

Running notes on things we've noticed during geo-eval-harness sweeps that
might be worth a paragraph in a paper or a future ablation. One section
per finding; nothing on this page is a benchmark result — those live in
`results/` and the explorer.

## Prithvi / Clay CLS-token embeddings collapse on visually redundant patches (2026-05-16)

While running intrinsic-dimension probes on `eurosat-spatial`, TwoNN returned
`nan` for 12 of 34 (model, bands) combos. All 12 were Prithvi-EO (v1/v2 100,
v2 300, v2 300_tl, v2 600) or Clay-v1.5; the other backbones (Terramind,
CROMA, DOFA, ScaleMAE, the timm baselines) were fine.

Diagnostic dump from a representative failing run:

```
[intrinsic-dim] TwoNN nan on X(10000, 768) —
  d1[min=0 median=4.6e-1 zeros=348]
  d2[min=0 median=5.2e-1 zeros=156]
  mu[min=0 max=2.9e+37 zeros=156]
  X[norm=7.8–9.2 std=0.31]
```

348 of 10,000 rows had a nearest-neighbour distance of exactly zero in
fp32. The rows weren't bit-equal — `np.unique(axis=0)` left them in —
only the *squared distance* `‖a‖² + ‖b‖² − 2·a·b` underflowed.

Why these specific backbones:

- Both Prithvi (MAE-style pretraining) and Clay (self-supervised MAE
  variant) end the feature extractor with `LayerNorm + [CLS]` pooling.
  LayerNorm forces every output to zero mean / unit variance per-sample,
  collapsing the 768-D space onto a hypersphere — embeddings vary in
  direction, not magnitude (norms 7.8–9.2 across 10k samples).
- EuroSAT has lots of visually redundant patches (forest, water,
  cropland at the same Sentinel-2 capture time). A frozen MAE encoder
  pretrained on Earth observation has been *taught* to abstract these
  into the same compact cluster.
- fp32 precision wall: at `‖a‖² ≈ 7.7`, the squared-distance dot product
  loses ~7 digits; embeddings with true L2 distance below ~1e-3 round to
  squared-distance zero.
- Terramind doesn't show the failure because its multi-modal generative
  pretraining rewards distinguishing instances cross-modally rather than
  abstracting them.

The mathematical degeneracy aside, this is a substantive observation
about Prithvi/Clay's feature geometry on satellite imagery: a non-trivial
fraction of EuroSAT patches are *literally indistinguishable* under their
CLS-token representation, in fp32 and likely in semantic content too.

Code workaround in `src/geo_eval_harness/intrinsic_dim.py`: filter rows
where `d1 == 0` before TwoNN. Fixes all 12 nan cases on rerun.

## Open ablations / TODOs

### CLS-token vs mean-pool ablation
The finding above is specifically about CLS-token pooling. Different
pretraining objectives may make CLS more or less informative:

- **MAE-style (Prithvi, Clay):** trained to reconstruct masked patches.
  The CLS token isn't directly supervised; it picks up whatever signal
  the encoder happens to route through it. Mean-pooling the patch
  tokens might preserve more information.
- **DINOv3-style:** explicit CLS supervision (self-distillation
  objective); CLS is the canonical feature.
- **CLIP-style:** pooled patch features are the canonical output.

Today the wrappers in `src/geo_eval_harness/models/` mostly hard-code
their pooling choice. Worth adding a `pooling: cls|mean|both` config
knob and re-running the GeoFM sweep with each — could meaningfully
change Prithvi / Clay numbers and the resulting ID estimates.

## CLS pool beats mean across full GeoBench (Prithvi, all v1+v2) (2026-05-17)

The 2026-05-16 finding above noted CLS-token collapse on EuroSAT.  The
full GeoBench v1+v2 sweep (10 datasets, 28 model variants) tells a
different story: **CLS-token linear-probe accuracy beats mean-pool by
+2 to +6 points across all Prithvi families**, averaged over every
(dataset, bands, normalization) cell:

| Model family | Δ (cls − mean), avg over 40 cells |
|---|---|
| `tt_prithvi_eo_v2_600` | **+0.060** |
| `tt_prithvi_eo_v2_300` | **+0.060** |
| `tt_prithvi_eo_v2_300_tl` | **+0.051** |
| `tt_prithvi_eo_v1_100` | **+0.050** |
| `tt_prithvi_eo_v2_100_tl` | **+0.021** |
| `tt_clay_v1_5_base` | -0.002 |
| `tgeo_scalemae_large_fmow` | -0.006 |

Implication: the previous EuroSAT-only observation was real but
specific — EuroSAT's visually homogeneous classes plus MAE pretraining
land enough patches in the same CLS cluster to crater the linear
probe.  On harder datasets (forestnet, treesatai, the so2sat family)
the CLS token actually carries more linearly-separable signal than
average-pooled patches.  Clay and ScaleMAE are roughly neutral.

The pool=cls vs mean ablation should stay; the safe default for
Prithvi linear probing is CLS, not mean.

## `model_native` is catastrophic for Prithvi (2026-05-17)

The sweep enabled a per-(model, dataset) comparison of
`bandspec_zscore` vs `model_native` normalization.  Surprising blowup
on Prithvi:

| Model | bandspec_zscore | model_native | Δ |
|---|---|---|---|
| `tt_prithvi_eo_v2_300` | 0.719 | 0.472 | **−0.247** |
| `tt_prithvi_eo_v2_600` | 0.725 | 0.481 | **−0.244** |
| `tt_prithvi_eo_v2_100_tl` | 0.712 | 0.474 | **−0.238** |
| `tt_prithvi_eo_v2_300_tl` | 0.712 | 0.478 | **−0.234** |
| `tt_prithvi_eo_v1_100` | 0.689 | 0.484 | **−0.205** |

All `prithvi_*_cls` variants drop ~−0.08 to −0.11 (smaller but still
big).  Clay and Terramind drop ~−0.04, the timm and most torchgeo
backbones are roughly neutral.

The Prithvi terratorch wrappers declare
`expected_input_unit = InputUnit.S2_DN`, which under `model_native`
triggers a `/10000` reflectance conversion before pretrain-stat
centring.  The conversion is correct on raw S2 DN input, but
`bandspec_zscore` had already z-scored the input distribution in a
way the encoder evidently expects — the round-trip through the
reflectance pipeline takes ~20 points of linear-probe accuracy.

Defensible default: `bandspec_zscore` for all linear probing.  Treat
`model_native` as a diagnostic, not a baseline.

## DINOv3 (web + sat493m) is competitive with GeoFMs across the board (2026-05-17)

DINOv3 ViT-L (timm checkpoint, no satellite-specific pretraining
beyond `sat493m`) tops the linear-probe leaderboard on **5 of 10**
GeoBench classification datasets:

| Dataset | Winner | Score |
|---|---|---|
| `m-brick-kiln` | `vit_large_patch16_dinov3sat` | **0.976** |
| `forestnet` | `vit_large_patch16_dinov3sat` | **0.580** |
| `m-forestnet` | `vit_large_patch16_dinov3sat` | **0.582** |
| `treesatai` | `vit_large_patch16_dinov3sat` | **0.682** |
| `m-pv4ger` | `vit_large_patch16_dinov3` | **0.974** |
| `m-eurosat` | `tgeo_dofa_large` | 0.973 |
| `m-so2sat` | `tt_terramind_v1_base` | 0.739 |
| `so2sat` | `tt_terramind_v1_base` | 0.743 |
| `benv2` | `tt_terramind_v1_large` | 0.846 |
| `m-bigearthnet` | `tt_terramind_v1_large` | 0.750 |

`sat493m` was pretrained on Maxar WorldView imagery (heads-up that the
pretrain mean/std `(0.43, 0.41, 0.30)` come from that domain), yet the
features transfer cleanly to Sentinel-2 RGB datasets that DINOv3 has
never seen during pretraining.  `vit_large_patch16_dinov3` (web
pretrain, ImageNet stats) wins on m-pv4ger (NAIP RGB) where the
deployment domain is closest to natural imagery.

`model_native` (timm's pretrained stats) is consistently slightly
worse than `bandspec_zscore` for both DINOv3 variants — single-digit
negative deltas, max −8% on m-so2sat / so2sat.  Confirms the
Maxar-derived stats don't transfer perfectly across sensors, even
when they are *a* reasonable normalization.

## Cross-dataset intrinsic dimensions are small and stable (2026-05-17)

TwoNN median intrinsic dimension across 28 model variants, per
dataset:

| Dataset | Median TwoNN ID |
|---|---|
| `m-bigearthnet` | 12.1 |
| `benv2` | 12.0 |
| `m-pv4ger` | 11.3 |
| `treesatai` | 10.4 |
| `m-brick-kiln` | 10.1 |
| `m-eurosat` | 8.7 |
| `so2sat` | 7.9 |
| `m-so2sat` | 7.8 |
| `forestnet` | 6.7 |
| `m-forestnet` | 6.4 |

Embedded feature manifolds live in 6–12 effective dimensions for
every backbone we tested, on 768/1024-D embeddings.  The forestnet
pair sits at the bottom — those are also the hardest-to-classify
datasets (peak linear acc ~58%), consistent with low intrinsic
information content.

One outlier worth flagging: `tgeo_resnet50_s2rgb_moco` reports TwoNN
ID ≈ 365 on so2sat and m-so2sat, vs ≤ 100 for every other
(model, dataset) cell.  Likely either a normalization mismatch
inflating the manifold or genuine high-dim noise in that backbone's
final features — follow-up diagnostic.

## Adapting RGB-pretrained models to multispectral is a regression (2026-05-17)

Wired `timm.adapt_input_conv` into the Swin / ScaleMAE / EarthLoc
torchgeo wrappers (ResNet already had it), and matched the
weights-bound `Normalize` by tiling its RGB mean/std cyclically to
the target channel count — same tiling pattern the conv weights use.
For `in_chans = 7` and 3-channel RGB pretrain stats, both layers
end up treating channels as `[R, G, B, R, G, B, R]`.

Result across 17 (model × dataset) cells where both RGB and adapted
multispectral linear probes ran:

| | Count |
|---|---|
| Improved with adapted multispec | **1** (scalemae × m-eurosat, +2.9pp) |
| Regressed (>0.5pp) | **16** |
| Mean Δ (all − rgb) | **−3.9pp** |
| Worst regression | `scalemae_large_fmow × treesatai` at **−21.1pp** |

Why it fails:

- ImageNet / fMoW / NAIP RGB mean/std are calibrated for natural-image
  reflectance distributions — applying the *red* stat to a Sentinel-2
  SWIR band yields a normalized value that's wildly off-distribution.
- The first-conv tiling has the same problem: each adapted "R-slot"
  channel mostly produces activations as if it were the red band, so
  any per-channel meaning of the extra bands is wiped out.
- Frozen-feature linear probing can't recover from either; for
  fine-tuning the adapter at least provides a working init.

The wrapper still supports the adaptation path — it's gated by
`SINGLE_BAND_MODE_MODELS` in the sweep generator.  These models'
multispectral rows should be marked **"adapted\*"** in the explorer
and `RGB-only` should remain their canonical leaderboard entry.

Verdict-style summary of which models actually want multispectral:

| Model class | Multispec helps? |
|---|---|
| Native multimodal (CROMA, Terramind, Prithvi, Clay, OlmoEarth, Panopticon) | ✓ uniformly |
| Band-agnostic with wavelength embeddings (DOFA) | ✗ empirically RGB-better |
| RGB-pretrained (ResNet/Swin/ScaleMAE/DINOv3 family) | ✗ adapter path hurts |

## Input-scale mismatch silently broke 105 (model, dataset) cells (2026-05-18)

Several torchgeo wrappers ship a weights-bound ``Normalize`` calibrated
for a specific input scale (S2 DN, uint8/255, reflectance), but the
wrapper applied that Normalize regardless of what scale the dataset
actually delivered.  The two extremes:

| Pretrained scale | Dataset scale | Effect of mismatched Normalize |
|---|---|---|
| ``s2_dn_div10000`` | reflectance [0, 2.8] (so2sat) | inputs become ~0.0003; features collapse, linear probe = chance |
| ``uint8_div255`` | S2 DN [0, 10000] | inputs become ~40; first conv saturates; features near-constant |

Worst observed: ``resnet50_s2rgb_moco × m-so2sat`` linear = **0.059** =
1/17 = exact chance level.  An audit (helper: ``convert_unit`` in
``models/_input_units.py``) found **105 affected cells** across:

* ResNet S2-MoCo / SeCo / Satlas variants on uint8 datasets
  (m-forestnet Landsat, m-pv4ger NAIP, forestnet V2 rgb)
* ScaleMAE / EarthLoc / Swin-Satlas on S2 DN datasets
  (m-eurosat, m-bigearthnet, eurosat-spatial, benv2, treesatai)
* CROMA / Panopticon on so2sat reflectance datasets

Fix (PR #85): ``_TorchGeoBackboneBench.normalize_inputs`` now calls
``convert_unit(images, detected_dataset_unit, weights_target_unit)``
before the Normalize layer.  All ResNet/ScaleMAE/EarthLoc/Swin
variants on previously-mismatched datasets jumped 30-53 percentage
points after the rerun.

Example fix-deltas:

| Cell | Before | After | Δ |
|---|---|---|---|
| ``resnet50_s2rgb_moco × m-so2sat`` | 0.059 | **0.595** | +0.536 |
| ``resnet50_s2rgb_moco × so2sat`` | 0.059 | **0.595** | +0.536 |
| ``resnet50_s2_all_moco × so2sat`` | 0.059 | **0.357** | +0.298 |

The previous "intrinsic dim ≈ 365" outlier on
``resnet50_s2rgb_moco × so2sat`` (vs typical 6-12) also resolved — it
was symptomatic of the same collapsed-feature pathology.

## DOFA wavelength fallback is now an error, not a silent default (2026-05-18)

``_resolve_dofa_wavelengths`` used to fall back to 0.6 µm (~green band)
for any ``BandSpec.wavelength_um is None``.  DOFA's wavelength
embedding is the only signal the model gets about each spectral
channel — silently routing a SAR backscatter channel through the
green-band embedding would produce garbage features without any
error.

Audit confirmed the published 14-cell DOFA sweep (all S2 RGB-only)
was unaffected because every S2 optical band has wavelength_um set.
But m-forestnet's Landsat NIR/SWIR_1/SWIR_2 bands lack wavelengths,
and so2sat / benv2 / treesatai SAR bands also have ``None``.  Any
future DOFA probe on those would have silently mapped to ~green.

Now raises a ``ValueError`` listing the offending band names.  No
behavioural change on existing data; future-proofing only.
