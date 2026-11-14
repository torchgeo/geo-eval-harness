Models
======

Select a model preset with ``--model``. This page describes its inputs, outputs, and evaluation settings; :doc:`/api/models` lists the public classes.

Commands that use ``docs/examples/`` paths run from the repository root. With an installed package, save the example YAML locally and pass that path to ``--config``.

.. _model-presets:

Available presets
-----------------

Every preset under :file:`src/geo_eval_harness/conf/model/` becomes a
``--model`` selector. A preset's ``target`` is an importable class; constructor
options live in ``kwargs``, separate from input and evaluation defaults.
A preset may declare ``supported_datasets`` to reject unverified model/dataset combinations during resolution, before samples or weights are loaded. This metadata is not passed to the constructor.

Tessera V2
^^^^^^^^^^

``tessera_v2/nano``, ``tessera_v2/small``, ``tessera_v2/medium``, and ``tessera_v2/large`` select the released pixel students (approximately 1.07M, 7.11M, 21.03M, and 43.83M parameters). Medium is the upstream balanced default. The wrapper downloads immutable Hugging Face revisions, verifies packaged SHA-256 digests, constructs the architecture from checkpoint metadata, and loads the ``model`` state dictionary strictly. Architecture code is MIT-licensed; weights are CC0-1.0. The teacher and V1 checkpoints are not supported.

.. code-block:: console

   $ geo-eval-harness run --config docs/examples/tessera-v2-classification.yaml
   $ geo-eval-harness run --config docs/examples/tessera-v2-segmentation.yaml
   $ geo-eval-harness run --config docs/examples/tessera-v2-lora.yaml

Download the :download:`classification </examples/tessera-v2-classification.yaml>`, :download:`segmentation </examples/tessera-v2-segmentation.yaml>`, or :download:`LoRA </examples/tessera-v2-lora.yaml>` configuration.

**Input contract.** The wrapper requires an explicit S2-only channel selection and maps it to ``B04 B02 B03 B08 B8A B05 B06 B07 B11 B12``, retaining the distinction between B08 and B8A. Mixed sensors are rejected rather than silently dropped. It converts reflectance to DN with a factor of 10000 and applies the released per-band mean and standard deviation exactly once. The authoritative constants are ``S2_MEAN`` and ``S2_STD`` in ``geo_eval_harness.models.tessera`` and the `pinned upstream student source <https://github.com/ucam-eo/tessera/blob/e368539f7c4711e5f16fdd09801573830e34baf9/tessera_infer_v2/student/model.py>`_. Other normalization strategies are rejected. Missing S2 bands are filled with their pretraining means, yielding zero after standardization; spectral-neighbor substitution is disabled. Neither dataset z-scores nor an arbitrary uint8-to-reflectance conversion reproduces this input scale.

The named presets enforce a dataset allowlist: the verified L2A datasets ``benv2``, ``m-bigearthnet``, ``pastis``, ``treesatai``, and ``fotw``, plus two Level-1C datasets accepted as an approximation (see below). They select only S2 channels from mixed-sensor datasets. ``fotw`` uses four-band seasonal composites and therefore mean-fills unavailable bands; its original imagery is `Level-2A bottom-of-atmosphere data <https://arxiv.org/html/2409.16252#S2>`_. Seasonal composites, snapshot timing, cloud content, and spatial resolution still differ from Tessera's annual training distribution. Direct constructor users are responsible for establishing the same L2A provenance.

The allowlist excludes ``m-eurosat``, ``m-brick-kiln``, and ``dynamic_earthnet`` until their band identities/radiometry are resolved in separate dataset work. It also excludes ``burn_scars``, whose upstream HLS imagery includes Landsat and is outside this S2-only protocol. ``eurosat`` and ``sen1floods11-s2`` are Level-1C top-of-atmosphere products, accepted as an explicit approximation. Their reflectance still includes atmospheric path radiance, which raises mainly the shorter-wavelength bands, and it is not converted to the bottom-of-atmosphere reflectance Tessera was trained on. Both datasets were published before processing baseline 04.00 added a +1000 offset to L1C values in January 2022, so no offset is subtracted. Report their rows as L1C approximations, not as L2A results. The other L1C products, ``eurosat-spatial`` and ``oscd-post``, remain excluded. ``so2sat``, ``m-so2sat``, and ``cloudsen12`` remain excluded pending product verification, as does ``forestnet`` with its unverified 8-bit radiometry and conflicting sensor labels. Other sensors and SAR-only inputs are unsupported.

**Snapshot dates.** Tessera normally consumes dated annual time series. This wrapper instead treats one snapshot as the released eight-observation bucket, retaining the original missing-S1 token and both pretrained branches. It computes that bucket as a single token: eight identical observations with one date attend uniformly to each other and pool back to that observation, and the missing-S1 token is the same for every pixel, so its branch runs once per chunk. Features and LoRA gradients match the eight-copy input to floating-point rounding (about 5e-6 on the released weights), at roughly a sixth to a tenth of the cost. PASTIS uses the date of the first S2 frame actually selected by its loader. Other inputs use the configurable ``model.kwargs.fallback_doy`` (183 by default), unless aligned per-sample ``doy`` metadata is supplied. Real calendar ordinals, including 366, are accepted. DOY 183 is a reproducibility convention with no claim of seasonal neutrality: the released model has no unknown-date token. For undated datasets, report validation-only sensitivity at DOY 1, 92, 183, and 274 while keeping samples fixed; do not select a date from test metrics. Multi-date inputs are rejected rather than silently pooled.

Because the canonical samples do not expose a verified input cloud-clear mask, this protocol does not filter clouds. Segmentation ``mask`` values are ground truth and never enter preprocessing.

**Feature endpoint.** Classification averages the final pixel embeddings spatially. Segmentation hooks the complete NCHW ``spatial_features`` map after the MLP reducer and final non-affine LayerNorm, before spatial pooling. ``embedding_dim`` defaults to 128; 16, 32, and 64 select direct prefixes of that final output without a second normalization. These prefixes share the same checkpoint. Snapshot inference and mean-filled bands are harness adaptations, so these results do not reproduce the published annual-embedding benchmark.

**Segmentation and LoRA.** The presets use a single full-resolution map and the existing ``linear`` head, with no ViT neck. For UPerNet, set ``head: upernet`` and ``layers: [pyramid.s32, pyramid.s16, pyramid.s8, pyramid.s4]``: average-pooled copies of that map, coarse-to-fine, at the strides that pyramid backbones such as ConvNeXt give. The pixel embeddings carry no spatial context, so UPerNet's convolutions and pyramid pooling add it. UPerNet on the full-resolution map alone, or behind the ViT neck, which upsamples it up to 4x, needs roughly 6x and 130x the training memory and does not fit Sen1Floods11's 512 px grid at batch 16. LoRA is opt-in and adapts ``s2_backbone.transformer_encoder.layers.<i>.linear1/linear2`` through weight parametrizations so torch's fused Transformer kernels consume the updates. The base encoder, missing S1 branch, and attention-pooling weights stay frozen. Rank, alpha, last-block selection, and backbone learning rate use the existing segmentation fields. This is a snapshot-specific harness protocol rather than an upstream fine-tuning recipe; task-specific LoRA does not guarantee that all other Matryoshka prefixes retain their pretrained properties.

The model processes independent pixels, so ``batch_pixels`` (default 32768) bounds each encoder call. On GPU, small calls are limited by per-call overhead rather than arithmetic, so fewer, larger calls are much faster. A LoRA training call keeps roughly 17, 119, and 178 KiB of activations per pixel for Nano, Medium, and Large (FP32 upper bound), so the default needs at most about 0.5, 3.7, and 5.6 GiB per call; lower it on small GPUs. ``batch_pixels`` is part of ``config_hash``. Gradient-bearing chunks use activation checkpointing, but chunking does not bound the full dense output map. Configured resizing is rejected: keep ``input.image_size: null`` to retain native grids. For uncached segmentation, the loader batch size is ``runtime.batch_size`` (CLI ``--batch-size``), not ``segmentation.batch_size``. Both segmentation examples explicitly set the actual loader batch to 2; choose a measured batch size for each grid rather than relying on the CLI default of 64.

Feature caching defaults to false because a full-resolution cache requires roughly ``N * D * H * W * bytes_per_value`` bytes per split. Enable caching only after checking that estimate. Fleet jobs must use the preset dataset allowlist, native grids, S2-only bands, either the linear head on ``spatial_features`` or UPerNet on the pooled pyramid, no neck, and an explicit loader batch size; generic all-dataset, resized, necked, and full-fine-tuning sweep recipes are not this protocol. These models expose no verified spatial-projection boundary for FLOPs resolution sweeps; use the existing real-batch ``profile`` command to measure execution.

For PyTorch builds that dispatch eager attention through Triton, the fleet image also needs a working C compiler, assembler, development headers, and CUDA driver linker configuration. Check these prerequisites with an AMP pilot for every student rather than assuming that a working FP32 run establishes mixed-precision readiness.

Clay v1 Base
^^^^^^^^^^^^

``terratorch/clay_v1`` and ``terratorch/clay_v1_cls`` load TerraTorch's ``timm_clay_v1_base`` backbone, the 92,102,144-parameter Clay v1 Base encoder from ``made-with-clay/Clay-legacy`` (``v1/clay-v1-base.ckpt``). Mean-patch and CLS-token pooling use the same weights. The Clay v1.5 checkpoint is a separate, larger encoder.

Saved Clay v1 measurements use the canonical v1 identifiers. Their numerical values, source fingerprints, and original configuration hashes are retained. New runs hash the canonical preset name for resume matching. Use ``python scripts/migrations/migrate_clay_v1.py`` to relabel a separate results checkout.

Clay v1.5 Large
^^^^^^^^^^^^^^^

``terratorch/clay_v1_5`` and ``terratorch/clay_v1_5_cls`` use the 311M-parameter Clay v1.5 Large encoder with mean-patch and CLS-token pooling respectively. The official checkpoint is pinned and SHA-256 verified. Segmentation reads blocks 5, 11, 17, and 23.

Inputs default to 256 pixels and must be square with sides divisible by eight. Set ``--image-size`` to change the input size or ``model.kwargs.checkpoint_path`` to load a local checkpoint containing the complete ``model.encoder.*`` weights.

TerraFM
^^^^^^^

``terrafm/base_s2``, ``terrafm/base_s2_l1c``, and ``terrafm/base_s1`` select the Sentinel-2 L2A, Sentinel-2 L1C, and Sentinel-1 VV/VH stems. Download ``TerraFM-B.pth`` from `MBZUAI/TerraFM <https://huggingface.co/MBZUAI/TerraFM>`_ and provide its local path in a run config:

.. code-block:: yaml

   model:
     name: terrafm/base_s2
     kwargs:
       checkpoint_path: checkpoints/TerraFM-B.pth
   datasets: [m-eurosat]

The presets default to dataset z-scoring. With ``--normalization model``, the wrapper converts Sentinel-2 inputs to DN and linear-power Sentinel-1 inputs to dB, then applies TerraFM's ``(x - (mean - 2 * std)) / (4 * std)`` scaling. ``model.kwargs.native_stats`` selects ``pretrain`` statistics (default) or ``dataset`` statistics for that scaling. With ``pretrain`` statistics, Sentinel-1 bands must declare unit ``db`` or ``linear_power``; other SAR inputs raise an error. The 12-band Sentinel-2 order follows Major-TOM L2A, excluding B10; this ordering is assumed rather than documented upstream. Each preset processes one modality, and checkpoint loading rejects missing or mismatched weights.

SMARTIES
^^^^^^^^

``smarties/base`` (ViT-B/16, 768-d) and ``smarties/large`` (ViT-L/16, 1024-d) wrap `SMARTIES <https://github.com/gsumbul/SMARTIES>`_ (`arXiv:2506.19585 <https://arxiv.org/abs/2506.19585>`_). Weights download from `gsumbul/SMARTIES-v1-ViT-B <https://huggingface.co/gsumbul/SMARTIES-v1-ViT-B>`_ and `gsumbul/SMARTIES-v1-ViT-L <https://huggingface.co/gsumbul/SMARTIES-v1-ViT-L>`_ at pinned revisions with SHA-256 checks; the MAE decoder is ignored and every encoder weight must load.

SMARTIES projects each band with a layer chosen by its spectral range, so one input can use any subset of a sensor's bands:

* Sentinel-2 bands use their own projections; B10 cirrus is dropped, as upstream does.
* Sentinel-1 VV and VH (sensors ``s1``, ``sar``, ``s1_asc``, ``s1_desc``) use the Sentinel-1 projections.
* Red, green, and blue from any other optical sensor use the Maxar RGB projections, and its NIR and SWIR bands use the Sentinel-2 projection of the same name.

Other bands, such as WorldView yellow, complex or unpolarised SAR, and elevation, are rejected. Each input must come from one sensor: upstream encodes multi-sensor pairs separately and concatenates the features, which the single-pass interface does not support. Inputs are resized to 224 px and the CLS token is the default embedding, as in upstream kNN and linear probing; segmentation reads blocks 2/5/8/11 (base) or 5/11/17/23 (large).

The presets default to ``--normalization model``, SMARTIES' pretraining preprocessing:

1. Convert each band to its pretraining unit: Sentinel-2 DN, Sentinel-1 dB, or 8-bit RGB.
2. Min-max scale it with the pretraining 1st and 99th percentiles, and clip to [0, 1].
3. Standardize with the post-clip mean and standard deviation.

Sentinel-2 uses the L2A statistics (BigEarthNet) unless ``s2_level: l1c`` selects the L1C statistics (fMoW-Sentinel). The presets select L1C for EuroSAT, BrickKiln, So2Sat, Sen1Floods11, and OSCD.

Upstream downstream evaluations compute percentiles per dataset instead, which ``BandSpec`` does not record. Bands without pretraining statistics fail under ``model``; the presets use ``dataset`` normalization for Dynamic EarthNet, SpaceNet 2, both ForestNets, Infra-Bench CLS, and OpenEarthMap-SAR (``--bands vv,vh``). Aerial NIR and SAR bands without a declared ``BandSpec.unit`` elsewhere need ``--normalization dataset``.

Random Convolutional Features (RCF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~geo_eval_harness.models.RCFBench`. Gaussian or empirical random
features in the spirit of MOSAIKS.

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat

For empirical features, pass this YAML through ``--config``:

.. code-block:: yaml

   model: {name: rcf, kwargs: {mode: empirical, features: 1024}}
   datasets: [m-eurosat]

Image statistics baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~geo_eval_harness.models.ImageStatsBench`. A baseline that returns each channel's mean, population standard deviation, maximum, and minimum as the feature vector.

.. code-block:: console

   $ geo-eval-harness run --model imagestats --dataset m-eurosat

Handcrafted baseline
^^^^^^^^^^^^^^^^^^^^

:class:`~geo_eval_harness.models.HandcraftedBench` is a deterministic,
zero-learned-parameter baseline for arbitrary input channels and positive
spatial dimensions. It includes every raw channel (including SAR), plus
available sensor-local optical index maps. It has no weights to download,
feature selection, random initialization, image gradients or training state.

.. code-block:: console

   $ geo-eval-harness run --model handcrafted_level2 --dataset eurosat --bands all --normalization none
   $ geo-eval-harness run --model handcrafted_level3 --dataset resisc45 --bands all --normalization none
   $ python experiments/run_handcrafted.py --levels 1 2 3

The sweep script selects classification datasets from the current catalog, including AID. ``--dataset all`` also includes segmentation tasks, which this patch-feature baseline does not support.

Use ``--output-dir results/handcrafted-current`` to keep a sweep separate from the repository's reference results.

Levels are cumulative and their feature vectors are strict prefixes:

.. list-table::
   :header-rows: 1
   :widths: 10 15 75

   * - Level
     - Width
     - Features
   * - 1
     - ``9M``
     - Mean, population std, min, max, and linearly interpolated p10, p25,
       p50, p75, p90, for every raw band and available index map.
   * - 2 (default)
     - ``21M``
     - Adds gradient magnitude mean/std, global structure-tensor coherence
       and magnitude-weighted, eight-bin unsigned orientation entropy at
       pooling factors 1, 2 and 4 for every map.
   * - 3
     - ``37M``
     - Adds LBP entropy/uniform share, Harris corner density/strength,
       four quadrant means/stds, and low/high-tail anisotropy/spread for
       every map, including raw SAR channels.

Here ``M`` is the number of input channels plus available index maps, not a
fixed model width. An RGB input has widths 27/63/111; 13 Sentinel-2 bands
with all four indices have widths 153/357/629. Names are ordered first by
level, then by map, then by statistic. Raw maps follow input order; indices
follow sensor first-occurrence order and the fixed NDVI, NDWI, NDBI, NBR order.
``feature_names`` is a tuple and ``num_features`` is fixed at construction,
regardless of image size. Names start with ``sensor.band.actual_band`` or
``sensor.index.index_name``; identifiers are URL-escaped and repeated identical
sensor/band labels receive numbered suffixes. ``feature_metadata`` is a
JSON-serializable list of map records with output columns, actual source band
names/channel indices, canonical roles, and explicitly skipped indices.

**Spectral indices and normalization.** Index roles are resolved by canonical
band name independently within each sensor: NDVI = (NIR - red)/(NIR + red),
NDWI = (green - NIR)/(green + NIR), NDBI = (SWIR1 - NIR)/(SWIR1 + NIR),
and NBR = (NIR - SWIR2)/(NIR + SWIR2). Broad NIR is preferred; narrow
NIR/B8A is used only when broad NIR is absent and its actual source is
recorded. Semantic Landsat NIR/SWIR names work even without wavelengths.
There is no nearest-band fallback, zero-filled source, or cross-sensor
mixing: aerial RGB never borrows Sentinel-2 NIR. SAR produces raw
statistics/texture but no optical indices; missing roles simply omit maps.

The constructor and all four presets default to identity normalization; the presets also select all bands. Explicit ``--normalization`` and ``--bands`` flags override these input defaults. The ``handcrafted`` preset defaults to level 2; custom YAML can set ``model: {name: handcrafted, kwargs: {level: 3}}``. Preprocessing and evaluation settings belong under ``input`` and ``classification``, not constructor kwargs.
Other base-class strategies are applied and recorded;
z-scored bands do not produce physically meaningful normalized-difference
indices. No fitted input or downstream scaler is added. The ratio calculation
rescales both operands by their shared absolute maximum to avoid overflow,
then returns zero where ``abs(a+b) <= 1e-6*(abs(a)+abs(b))``. It does not
divide by an epsilon or clip ratios from signed inputs.

**Texture and structure definitions.** Pooling uses non-overlapping mean
cells, including partial edge cells without zero padding. Gradients are
central differences with one-sided edges and zero on singleton axes.
Coherence is ``sqrt((Jxx-Jyy)^2 + 4*Jxy^2)/(Jxx+Jyy)`` for the global
mean gradient tensor. Orientation entropy is Shannon entropy divided by
``log(8)`` over angles modulo pi, weighted by gradient magnitude.
Constant maps have zero gradient, coherence and orientation entropy.

LBP uses eight clockwise neighbors starting at top-left, with a one bit
only when the neighbor is strictly above the center. It reports 256-bin
entropy divided by ``log(256)`` and the share with at most two circular
bit transitions. Only valid 3-by-3 neighborhoods count; smaller images
return zeros. A constant map with valid neighborhoods has uniform share one.
Harris uses deterministic per-map min/max contrast scaling, a 3-by-3
box-averaged gradient tensor and ``R = det(J) - 0.04*trace(J)^2``. Density
is the fraction of non-strict 3-by-3 local maxima above 1% of the maximum
positive response; strength is the mean positive response. Flat maps have
no corners.

Quadrants are TL, TR, BL, BR, with the extra row/column assigned to
top/left for odd dimensions; empty quadrants return zeros. Tail weights
are positive ``p25 - x`` and ``x - p75`` on min/max contrast-scaled maps.
Weighted coordinate covariance gives anisotropy (eigenvalue difference
over sum) and spread (square root of trace). Coordinates span [-1, 1]
per axis, with singleton axes at zero; empty or point-like tails yield
zeros. Structural divisions use a ``1e-6`` minimum denominator.

Computation uses existing PyTorch only, float32 with autocast disabled, and
two-map chunks to bound intermediate memory. Outputs stay on the input
device and can train an attached head through ordinary ``no_grad`` feature
extraction. Nonfinite inputs are rejected, and float32 overflow raises a
descriptive error. The model never resizes inputs: normal benchmark runs
inherit ``input.image_size: 224`` and bilinear interpolation.

timm — ImageNet-pretrained CNNs and ViTs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~geo_eval_harness.models.TimmPatchBenchModel`.  Configs under
:file:`src/geo_eval_harness/conf/model/timm/` cover ResNet, ConvNeXt,
EfficientNet, DenseNet, RegNet, MobileNetV3, VGG, MaxViT, and more.
ViT / DeiT / Swin variants live under :file:`timm/vit/`.

.. code-block:: console

   $ geo-eval-harness run --model timm/resnet50 --dataset m-eurosat
   $ geo-eval-harness run --model timm/convnext_base --dataset m-eurosat
   $ geo-eval-harness run --model timm/vit/vit_base_patch16_224 --dataset m-eurosat --image-size 224
   $ geo-eval-harness run --model timm/vit/swin_base_patch4_window7_224 --dataset m-eurosat --methods knn

ViT-style backbones expect a fixed spatial resolution.  Set
``--image-size 224`` (``bilinear`` by default; switch to
``bicubic`` / ``nearest`` via ``--interpolation``) to resize the
dataset tiles for any model.

For supported timm ViTs, set ``model.kwargs.dynamic_img_size: true`` to interpolate positional embeddings across input sizes. Add ``model.kwargs.dynamic_img_pad: true`` to zero-pad partial patches when an input dimension is not divisible by the patch size.

timm models rebuild their input convolution for any number of channels —
they work with ``--bands all`` out of the box (pretrained
3-channel weights are averaged / replicated as needed).

.. warning::

   ``timm/efficientnet_b1``'s ``linear`` probe is unreliable across most
   classification datasets (``knn5`` is fine). The C sweep in
   ``evaluate_logistic`` doesn't standardize features before fitting by
   default (see ``classification.feature_standardize``), and
   b1's feature geometry makes the fit unstable across the whole
   ``c_range`` grid rather than diverging outright — so it lands on a poor
   but finite ``best_c`` instead of raising
   ``LinearProbeDivergedError``. Treat
   ``efficientnet_b1`` ``linear`` numbers as noise; other EfficientNet
   variants (b0, b2-b4) are unaffected.

torchgeo foundation models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Configs under :file:`src/geo_eval_harness/conf/model/torchgeo/`.  Most are
RGB-only self-supervised checkpoints from torchgeo's model hub.

.. code-block:: console

   $ # Sentinel-2 RGB SSL
   $ geo-eval-harness run --model torchgeo/resnet50_s2rgb_moco --dataset m-eurosat
   $ geo-eval-harness run --model torchgeo/resnet18_s2rgb_seco --dataset m-eurosat
   $ geo-eval-harness run --model torchgeo/resnet50_fmow_gassl --dataset m-eurosat

   $ # ScaleMAE on fMoW RGB
   $ geo-eval-harness run --model torchgeo/scalemae_large_fmow --dataset m-eurosat

   $ # DOFA uses the selected bands' wavelengths
   $ geo-eval-harness run --model torchgeo/dofa_base --dataset m-eurosat

   $ # Satlas Swin-V2 (NAIP / Sentinel-2 RGB)
   $ geo-eval-harness run --model torchgeo/swinv2b_naip_satlas_mi --dataset m-eurosat
   $ geo-eval-harness run --model torchgeo/swinv2b_s2rgb_satlas_mi --dataset m-eurosat

   $ # EarthLoc place-recognition descriptor
   $ geo-eval-harness run --model torchgeo/earthloc_s2_resnet50 --dataset m-eurosat

Scale-MAE constructs its positional grid at the resolved input size. With ``--image-size none``, image runs and profiling use the dataset's native spatial size.

OlmoEarth (AI2)
^^^^^^^^^^^^^^^

:class:`~geo_eval_harness.models.OlmoEarthBenchModel`.  Requires the
optional ``olmoearth`` extra:

.. code-block:: console

   $ pip install 'geo-eval-harness[olmoearth]'

   $ # OlmoEarth v1 (Nano / Tiny / Base / Large)
   $ geo-eval-harness run --model olmoearth_nano --dataset m-eurosat
   $ geo-eval-harness run --model olmoearth_base --dataset m-eurosat
   $ geo-eval-harness run --model olmoearth_large --dataset m-eurosat --bands all

   $ # OlmoEarth v1.1 (Nano / Tiny / Base)
   $ geo-eval-harness run --model olmoearth_v1_1_nano --dataset m-eurosat
   $ geo-eval-harness run --model olmoearth_v1_1_tiny --dataset m-eurosat
   $ geo-eval-harness run --model olmoearth_v1_1_base --dataset m-eurosat

   $ # OlmoEarth v1.2 (Nano / Tiny / Small / Base)
   $ geo-eval-harness run --model olmoearth_v1_2_nano --dataset m-eurosat
   $ geo-eval-harness run --model olmoearth_v1_2_small --dataset m-eurosat
   $ geo-eval-harness run --model olmoearth_v1_2_base --dataset m-eurosat

   $ # OlmoEarth v1.3 (Base)
   $ geo-eval-harness run --model olmoearth_v1_3_base --dataset m-eurosat

OlmoEarth v1.1 uses a **linear patch embedding** (vs. convolutional in v1),
a single bandset per modality, and updated masking/loss functions, yielding a
≈ 3× reduction in MACs with comparable accuracy.  OlmoEarth v1.2 adds **RoPE
3D position encoding** and a new **Small** size (384-d) between Tiny and Base.
OlmoEarth v1.3 (Base only) adds an **AttentionPooler** after the ViT: it reads
the patch tokens into a 768-d location-token grid, with a detached linear
student projecting it to the 128-d vectors behind AI2's published embeddings.
The ``embedding`` kwarg selects which output is mean-pooled:
``location_tokens`` (default for v1.3, as in AI2's evals), ``patch_tokens``
(the v1.x readout) or ``student``.  v1.3 also sets channels missing from the
input to 0 after normalization (``missing_bands: zero``), the value band
dropout used during pretraining, instead of copying a spectral neighbour.
The ``version`` parameter selects the weight family:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Config
     - Version
     - Size
     - Notes
   * - ``olmoearth_nano``
     - v1
     - Nano
     - multi-bandset, conv patch embed
   * - ``olmoearth_tiny``
     - v1
     - Tiny
     -
   * - ``olmoearth_base``
     - v1
     - Base
     -
   * - ``olmoearth_large``
     - v1
     - Large
     -
   * - ``olmoearth_v1_1_nano``
     - v1.1
     - Nano
     - single-bandset, linear patch embed
   * - ``olmoearth_v1_1_tiny``
     - v1.1
     - Tiny
     -
   * - ``olmoearth_v1_1_base``
     - v1.1
     - Base
     -
   * - ``olmoearth_v1_2_nano``
     - v1.2
     - Nano
     - RoPE position encoding
   * - ``olmoearth_v1_2_tiny``
     - v1.2
     - Tiny
     -
   * - ``olmoearth_v1_2_small``
     - v1.2
     - Small
     - new size (384-d, ≈ 35.6M params)
   * - ``olmoearth_v1_2_base``
     - v1.2
     - Base
     -
   * - ``olmoearth_v1_3_base``
     - v1.3
     - Base
     - AttentionPooler; pools location tokens
   * - ``olmoearth_v1_3_base_patch_tokens``
     - v1.3
     - Base
     - pools ViT patch tokens (v1.x readout)
   * - ``olmoearth_v1_3_base_student128``
     - v1.3
     - Base
     - pools the 128-d student

.. note::

   Input normalization is selected with ``--normalization``
   (default ``dataset``). Each model receives the corresponding internal strategy through
   :class:`~geo_eval_harness.models.BenchModel`; use ``model`` for
   wrappers that declare pretrained input units / statistics, or ``none``
   when a backbone owns all normalization internally.

   GeoBench delivers Landsat imagery (e.g. ``m-forestnet``) as uint8
   [0, 255], a scale OlmoEarth's pretrained Landsat statistics (fit on real
   DN) can't match.  OlmoEarth therefore selects normalization per sensor
   (``norm_from_pretrained="auto"``, the default): Landsat is normalized with
   dataset-specific ``BandSpec`` stats while Sentinel-2 / SAR use the
   pretrained normalizer. Supply constructor-specific settings in YAML
   ``model.kwargs``, not as positional overrides.

.. note::

   **Per-model input resolution.** A preset may set ``input.image_size``
   instead of the built-in default ``224``. OlmoEarth is
   resolution-flexible, so its presets set ``image_size: null`` to evaluate at
   each dataset's **native** resolution rather than upsampling to 224×224
   (matching the reference OlmoEarth evals).  Models that omit the field
   inherit the built-in default. To force a specific size, pass
   ``--image-size 224`` or use ``--image-size none`` for native resolution.
   Explicit run YAML also overrides the preset. The effective size is recorded in the CSV and
   in the resume cache key.

SAM 3 vision encoder
^^^^^^^^^^^^^^^^^^^^

:class:`~geo_eval_harness.models.SAM3Encoder`.  Weights come from the gated Hub
repo `facebook/sam3 <https://huggingface.co/facebook/sam3>`_: accept the terms on
the model page, run ``hf auth login`` once, and the checkpoint downloads into the
HuggingFace cache on first use.

.. code-block:: console

   $ geo-eval-harness run --model sam3_encoder --dataset m-eurosat --bands red,green,blue

The ViT layers bake their RoPE grid in at construction, so the encoder is built
for one fixed input size (``input.image_size``, 224 by default) and rejects
any other size at forward time.  ``--image-size none`` is therefore not supported.

AnySat
^^^^^^

:class:`~geo_eval_harness.models.AnySatBenchModel` wraps AnySat v1 (Base, 126M
parameters; `Astruc et al., CVPR 2025 <https://arxiv.org/abs/2412.14123>`_).  The model
code loads through ``torch.hub`` from a pinned commit of
`gastruc/AnySat <https://github.com/gastruc/AnySat>`_. The weights come from a pinned commit of
`g-astruc/AnySat <https://huggingface.co/g-astruc/AnySat>`_ and are checked against their sha256.

.. code-block:: console

   $ geo-eval-harness run --model anysat_base --dataset m-eurosat --bands all
   $ geo-eval-harness run --model anysat_base_seg --dataset sen1floods11-s2

AnySat has one projector per modality, each built for a fixed band order and ground
sampling distance (GSD). The wrapper adapts each input to its projector as follows:

* It takes the modality from ``BandSpec.sensor`` (``s2``, ``s1``/``sar``, and very-high-resolution
  ``aerial`` RGB to the 1 m ``spot`` projector) and reorders the bands by name.  Presets set
  ``modality`` only where the sensor name does not apply: Landsat and HLS (``l7``), CaFFe and
  the SpaceNet datasets.
* It resamples every tile to the projector's GSD over the tile's true ground extent.
  ``input_gsd`` is the GSD of the pixels the model receives: native GSD × native tile size /
  ``image_size``.
* It sets ``patch_size`` in metres.
* It passes day of year 120 to the time-series projectors.

Projector bands that are missing from the input are filled with the dataset mean only when
``impute_missing`` is set, and the filled bands are logged.  The presets set it, for example
for RGB runs on Sentinel-2 datasets.  The ``s1`` projector also needs the VV/VH ratio of the
raw values.  The wrapper computes it and z-scores it with the training-split ``ratio_stats``
that the preset stores per dataset.

For segmentation, hook ``output_tap``: it receives AnySat's output tokens as
``[B, 768, h, w]``.  A hook on the final cross-attention block itself never fires, because AnySat
calls that block through a custom method.  The fusion blocks ``model.model.blocks.0``-``5`` give
``[B, 1 + h*w, 768]`` for single-modality inputs.  ``dense_output: true`` exposes AnySat's
1536-channel sub-patch output through ``dense_tap``.

AnySat's hub code imports a top-level ``src`` package, and so does UniverSat's.  The two
models therefore cannot be loaded in the same process.  Presets refuse datasets in AnySat's
pretraining data (TreeSatAI, FLAIR) and sub-metre SAR, for which AnySat has no projector.

THOR
^^^^

:class:`~geo_eval_harness.models.ThorBenchModel` wraps THOR (`Forgaard et al., 2026 <https://arxiv.org/abs/2601.16011>`_), a FlexiViT pretrained on Sentinel-1, -2 and -3. The presets ``thor/tiny``, ``thor/small``, ``thor/base`` and ``thor/large`` select the four released sizes. The encoder is reimplemented from `FM4CS/THOR <https://github.com/FM4CS/THOR>`_ at a pinned commit, because the upstream package pins an older ``timm`` and patches ``timm``'s attention globally. The weights come from pinned commits of the ``FM4CS/THOR-1.0-*`` Hugging Face repos and are checked against their sha256.

.. code-block:: console

   $ geo-eval-harness run --model thor/base --dataset m-eurosat --bands all
   $ geo-eval-harness run --model thor/base --dataset sen1floods11-s2 --bands all

THOR embeds each band with its own patch kernel at the band's native GSD and averages the bands of a group (10 m, 20 m and 60 m Sentinel-2, Sentinel-1 VV/VH, Sentinel-1 HH/HV) into one token grid. The wrapper adapts each input as follows:

* It maps ``s2`` bands and ``s1``/``sar`` VV, VH, HH and HV in dB by canonical name. Other sensors, linear-power SAR, and bands THOR was not pretrained on raise an error. Presets drop the 60 m Sentinel-2 bands B01, B09 and B10 (``ignore_bands: [coastal, watervapor, cirrus]``), as the authors' Sentinel-2 fine-tuning config in `FM4CS/thor_terratorch_ext <https://github.com/FM4CS/thor_terratorch_ext>`_ does; B10 is not in THOR at all.
* Tiles keep their native size (``input.image_size: null``). ``input_gsd`` is the GSD of the pixels the model receives, 10 m by default. Each group is resampled to its pretraining GSD over the tile's ground extent, rounded so that every group holds a whole number of patches.
* One ``patch_size`` in pixels applies to all groups. The presets use 4 px for tiles up to 128 px and larger patches for 256 px and 512 px tiles. ``max_tokens`` rejects settings that would exceed 4,096 tokens.

Classification embeddings average each group's final tokens and concatenate the groups (``merge: concat``, THOR's default in its PANGAEA experiments), so their width grows with the number of band groups. For segmentation, hook ``taps.<i>``: it receives block ``i``'s tokens as ``[B, C, h, w]``, with each group upsampled to the finest grid and merged the same way. THOR's pretraining statistics are not offered as model normalization, because its own fine-tuning configs z-score with dataset statistics.

EoS-FM
^^^^^^

:class:`~geo_eval_harness.models.EoSFMBenchModel` wraps EoS-FM (`Adorni et al., ECCV 2026
<https://arxiv.org/abs/2511.21523>`_), an ensemble of 22 ConvNeXtV2-Atto specialists (75M
parameters), each trained with supervision on one remote-sensing dataset.  Each checkpoint entry
stores the names of the bands its specialist was trained on: R,G,B or Sentinel-2 B4,B3,B2 for
most, NIR,R,G for three, VV,VH composites with expressions such as ``VV-VH`` for the SAR
specialists, 13 Sentinel-2 bands for three (one with a cloud band, ``CLD``) and Sentinel-1 plus
Sentinel-2 for one.  The weights come from a pinned commit of `pierreadorni/eos-fm
<https://huggingface.co/pierreadorni/eos-fm>`_ and are checked against their sha256.

.. code-block:: console

   $ geo-eval-harness run --model eosfm --dataset m-eurosat --bands all

The embedding concatenates the average-pooled last stage (320 channels) of every specialist, in
checkpoint order: 7040 features for any input.  Each specialist gets one input built from its
training bands, in its training order, with the band adaptation of upstream's encoder
(``SmartBandAdapter``, pierreadorni/EoS-FM@d61a5ef).  Each training band takes the first input
band that

1. has the same name; otherwise
2. has the same centre wavelength (Sentinel-2 B4 for R); otherwise
3. is the optical band with the nearest centre wavelength (B11 for a missing B10, R for NIR on
   RGB input); otherwise
4. is the next band of a fallback pool: SAR bands for a SAR band, optical bands for an optical
   one, all bands if that pool is empty.  The pool position advances with every fallback of the
   same specialist, so SAR specialists on RGB input get R,G and R-G.

As upstream, optical bands come before SAR bands.  The wrapper translates band names into
upstream's vocabulary: Sentinel-2 bands into B1 to B12 and B8A, other red, green, blue, NIR and
SWIR bands into R, G, B, NIR, SWIR1 and SWIR2, and VV, VH, HH and HV unchanged.  Other bands keep
their name and have no wavelength, so specialists reach them only through the fallback; for
example, So2Sat's complex and Lee-filtered SAR channels are not used by the SAR specialists.
Elevation bands (``kuro_siwo`` and ``flair2`` ``all``) are not passed to the specialists.  On the
released weights the per-level features equal upstream ``EosFM`` (raw features, no fusion)
exactly for every benchmark band set checked.

All specialists share one input normalization (``bandspec_zscore``): each dataset is
standardized with its own band statistics, as the authors recommend; the training statistics
stored in the checkpoint are not applied, as upstream.  Upstream's optional feature BatchNorm,
projection and fusion are not in the checkpoint, so the wrapper concatenates the raw specialist features.

For segmentation, the preset hooks ``taps.3`` to ``taps.0``.  Each tap concatenates one
ConvNeXtV2-Atto stage (strides 32, 16, 8 and 4; 320, 160, 80 and 40 channels) of all 22
specialists: 7040, 3520, 1760 and 880 channels.  This is the feature pyramid that upstream's
UperNet decodes, without its optional fusion convolution; the harness UPerNet projects each level
with its own learned 1x1 convolution.  The specialists have no ``nn.Linear`` layers, so there is
no LoRA preset, and segmentation runs frozen or with ``full_finetune``.  The preset sets
``cache_features: false``, because the 22 specialists take about 10 MB of cached features per
224 px sample.

The EuroSAT and BigEarthNet specialists were trained on the labels of EuroSAT and BigEarthNet
v2, so results on ``eurosat``, ``eurosat-spatial``, ``m-eurosat``, ``benv2`` and
``m-bigearthnet`` are not out-of-distribution.  For segmentation, the LoveDA and CaFFe
specialists were trained on ``loveda`` and ``caffe``, and the LEVIR-CD+ specialist on an
extended version of ``levir-cd-post``.  Two specialists share a task with benchmark datasets:
cloud segmentation on Sentinel-2 (``cloudsen12``) and ETCI 2021 Sentinel-1 flood mapping
(``sen1floods11-s1``, ``kuro_siwo``).  The Kenya crop-type specialist is trained on a dataset
that is not in the benchmark.  The code is licensed CC BY-NC-SA 4.0 and the
weights CC BY-SA 4.0, as stated on GitHub and Hugging Face respectively.

Adding a new model
------------------

There are two contribution pathways.  **Stage 1** lets you benchmark your
model locally and report results in a paper without opening a PR.  **Stage
2** covers the full code contribution: exporting the class, writing tests,
hosting weights, and submitting a PR.

.. seealso::

   :doc:`eval_own_model`
      Stage 1 — evaluate your model locally and report results.

   :doc:`contribute_model`
      Stage 2 — contribute the model as a PR to the shared benchmark.

.. note::

   Two key patterns apply regardless of stage:

   * **Do not put** ``bands`` **in the model YAML.**  The runner reads the
     current dataset's :class:`~geo_eval_harness.datasets.BandSpec` list
     and injects it into the constructor automatically.  Adding ``bands`` to
     the YAML causes a ``TypeError`` (duplicate keyword argument).
   * Forward the constructor's ``normalization`` argument to ``BenchModel``. Wrappers that always normalize internally declare ``handles_own_normalization = True``; see :doc:`eval_own_model` for the complete pattern.

For segmentation models, also pick the
:attr:`segmentation.layers <geo_eval_harness.segmentation_probe.SegmentationProbe>`
that the head will hook into — see :doc:`segmentation-layers` for
verified values per timm backbone family.
