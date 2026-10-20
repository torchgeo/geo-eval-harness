Evaluation methodology
======================

This page describes the evaluation methodology used by
:mod:`torchgeo_bench.main` for each supported task type.  In all cases
the backbone model is kept **frozen** — the benchmark measures the
quality of learned representations, not end-to-end fine-tuning
performance.  The model is reinstantiated fresh for each dataset
because ``num_channels`` (and therefore the input convolution) varies
across datasets and band selections.

Overview
--------

The benchmark loads a pre-trained backbone and one or more geospatial
datasets.  Depending on whether a dataset provides per-pixel masks
(segmentation) or per-image labels (classification), a different
evaluation path is taken:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Dataset type
     - Methods
     - Primary metric
   * - Classification (single-label)
     - ``knn5``, ``linear``
     - accuracy
   * - Classification (multi-label)
     - ``knn5``, ``linear``
     - micro-mAP
   * - Segmentation
     - ``seg-{linear,conv_block,fpn,dpt}``
     - mIoU

Optionally, intrinsic-dimension metrics
(:doc:`/api/intrinsic_dim`) can be emitted alongside the standard
classification rows with YAML ``intrinsic_dim: {enabled: true}``.

Feature extraction (classification)
-----------------------------------

For classification tasks, features are extracted **once** per split and
reused by both KNN and the linear probe.

1. The backbone is placed in ``eval()`` mode with gradients disabled
   (``torch.no_grad`` + ``torch.inference_mode``).
2. Each batch is passed through ``forward_patch_features``, returning
   embeddings of shape ``(B, K)``.
3. Dictionary outputs (e.g. DINO-style models) are flattened on the
   first available key in order: ``"norm"``, ``"global_pool"``,
   ``"head.global_pool"``.
4. 3-D outputs ``(B, tokens, K)`` (typical of ViT models) are
   **mean-pooled** across the token dimension to a single ``(B, K)``
   vector.
5. Embeddings and labels are concatenated across batches into NumPy
   arrays for downstream evaluation.

Feature-spectrum diagnostics
----------------------------

With ``intrinsic_dim.enabled`` enabled, each selected split also emits five
scale-invariant diagnostics from the **centered** embedding matrix.  Let
``s_i`` be its singular values and let

.. math::

   p_i = \frac{s_i^2}{\sum_j s_j^2}

be the fraction of feature variance carried by principal component ``i``.
The reported metrics are:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Metric
     - Definition
   * - ``effective_rank``
     - :math:`\exp(-\sum_{i:p_i>0} p_i \log p_i)`, the exponential spectral entropy.
   * - ``participation_ratio``
     - :math:`1 / \sum_i p_i^2`, an alternative effective dimensionality.
   * - ``pc1_variance_ratio``
     - :math:`p_1`, the variance explained by the leading principal component.
   * - ``pc10_variance_ratio``
     - :math:`\sum_{i=1}^{\min(10,d)} p_i`, the cumulative variance explained by
       the first ten components (or all components when ``d < 10``).
   * - ``spectral_anisotropy``
     - :math:`(d_{\text{eff}} p_1 - 1)/(d_{\text{eff}} - 1)`, leading-component
       dominance normalized so an isotropic spectrum is 0 and a rank-one
       spectrum is 1, where :math:`d_{\text{eff}} = \min(d, n - 1)` is the
       number of singular values centering can leave nonzero. Normalizing by
       the raw feature dimension ``d`` instead would give small splits
       (``n <= d``) an anisotropy floor above zero even for isotropic data.
       The ``d_eff <= 1`` convention is 0 because no direction can dominate
       another.

The same deterministic ``max_samples`` cap and seed used for intrinsic
dimension are applied before centering. Non-finite inputs, fewer than two
samples, and zero total variance raise an error rather than producing a
plausible-looking diagnostic.

KNN (k=5)
---------

**Method name:** ``knn5``.  A non-parametric baseline that measures how
well the feature space clusters by class.

Procedure
^^^^^^^^^

1. Extract train and test embeddings (see above).
2. Fit a fixed **k = 5** nearest-neighbour classifier
   (:class:`~torchgeo_bench.knn.KNNClassifier`, backed by FAISS) on the
   train embeddings.
3. Predict labels for every test sample.
4. Compute test-set accuracy (or micro-mAP for multilabel datasets).
5. Compute **95% bootstrap confidence intervals**
   (``classification.bootstrap_samples`` resamples, default ``200``) by resampling test
   predictions with replacement.

Key details
^^^^^^^^^^^

* No hyperparameter tuning — k is fixed at 5.
* No validation use — the validation split is extracted but ignored.
* FAISS is used for efficient L2 nearest-neighbour search and runs on
  CPU or GPU.
* Feature vectors are cast to ``float32`` and labels to ``int64`` before
  indexing.

Linear probe (logistic regression)
----------------------------------

**Method name:** ``linear``.  Multinomial logistic regression trained on
top of frozen features — the standard linear-evaluation protocol.

Procedure
^^^^^^^^^

1. Extract train, validation, and test embeddings.
2. **Hyperparameter sweep:** train one logistic regression per ``C``
   value in a log-spaced grid
   (``classification.linear``, default 40 values from 10⁻⁶ to 10⁴).
   Each model is evaluated on the validation set to pick the best ``C``.
3. **Final model:** retrain with the chosen ``C``, optionally on
   ``train ∪ val`` (``classification.linear.refit_train_val: true``, the default).
4. Evaluate on the test set; report accuracy / micro-mAP with 95%
   bootstrap confidence intervals.

Implementation
^^^^^^^^^^^^^^

:class:`~torchgeo_bench.linear.LogisticRegression` is a custom PyTorch
implementation matching scikit-learn's objective scaling:

.. math::

   \mathrm{loss} \;=\; \frac{1}{n}\,\mathrm{CrossEntropy}
                       \;+\; \frac{1}{n}\cdot\frac{1}{2C}\,\|W\|^2

* **Architecture:** a single ``nn.Linear(K, num_classes)`` (weight + bias).
* **Solver (sweep):** L-BFGS with strong-Wolfe line search,
  ``max_iter=2000``, ``tol=1e-6``.
* **Solver (final):** same, but ``max_iter=4000`` for tighter convergence.
* **Alternative solver:** mini-batch Adam is available; L-BFGS is the
  default.
* **No feature standardisation** — embeddings are used as-is.
* **TF32** is enabled on CUDA for faster matmul when available.

Hyperparameters
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Default
     - Description
   * - ``classification.linear`` range
     - ``[-6, 4, 40]``
     - log₁₀ start, stop, and number of ``C`` values
   * - ``classification.linear.refit_train_val``
     - ``true``
     - merge train + val for final model training
   * - ``classification.bootstrap_samples``
     - ``200``
     - bootstrap resamples for the confidence interval

Segmentation probes
-------------------

All segmentation methods share a common skeleton:
:class:`~torchgeo_bench.segmentation_probe.SegmentationProbe` registers
forward hooks on the configured
:attr:`segmentation.layers <torchgeo_bench.segmentation_probe.SegmentationProbe>`
to capture intermediate feature maps.  See :doc:`segmentation-layers`
for verified layer names per timm backbone family.

Each layer's feature map is reshaped into a spatial tensor
``(B, C, H, W)``:

* 2-D ``(B, C)`` is reshaped to ``(B, C, 1, 1)``.
* 3-D ViT-style ``(B, L, C)`` is reshaped to ``(B, C, √L, √L)``
  (assuming a square spatial grid).
* 4-D ``(B, C, H, W)`` is used directly.

The probe head is then trained end-to-end (backbone frozen) using
:class:`~torchgeo_bench.segmentation_task.SegmentationSolver`.

Linear segmentation probe
^^^^^^^^^^^^^^^^^^^^^^^^^

**Method name:** ``seg-linear``
(``segmentation: {head: linear}`` in YAML).
A lightweight per-pixel linear classifier per layer, with multi-layer
fusion via learned scalar weights.

* Per layer: ``BatchNorm2d → Conv2d(C, num_classes, 1×1)`` (a per-pixel
  linear classifier).
* Each layer's logits are bilinearly upsampled to the input resolution.
* Multi-layer logits are combined via a learned scalar ``scale_weights``
  parameter.

Convolutional probe
^^^^^^^^^^^^^^^^^^^

**Method name:** ``seg-conv_block``
(``segmentation: {head: conv_block}`` in YAML).
Slightly more expressive: projects and fuses multi-scale features
before classification, testing whether the backbone captures
complementary information at different depths.

1. Per layer: ``Conv2d(C, hidden_dim, 1×1, bias=False) → BatchNorm2d → SiLU``.
2. Bilinearly upsample all projected maps to the largest spatial
   resolution among them (minimum 16×16).
3. Concatenate along the channel dimension to
   ``(B, hidden_dim × num_layers, H, W)``.
4. ``Conv2d(hidden_dim × num_layers, num_classes, 1×1)`` to produce
   logits, then upsample to the input resolution.

FPN probe
^^^^^^^^^

**Method name:** ``seg-fpn`` (YAML ``segmentation: {head: fpn}``).
A Feature-Pyramid-Network-style top-down decoder that fuses multi-scale
maps in coarse-to-fine order — matching common dense-prediction
literature.

1. Layers must be supplied in **coarse-to-fine** order (deepest /
   lowest-resolution first, e.g. ``["layer4", "layer3", "layer2", "layer1"]``
   for a ResNet).
2. Each layer is projected to ``hidden_dim`` channels via a lateral 1×1 conv.
3. **Top-down pathway:** starting from the coarsest scale, each level is
   upsampled 2× and added to the next finer lateral output.
4. Each merged level is refined with a 3×3 conv.
5. All refined levels are upsampled to the finest spatial resolution,
   concatenated, and passed through a 1×1 conv to per-pixel class logits.
6. Logits are bilinearly upsampled to the input resolution.

DPT probe
^^^^^^^^^

**Method name:** ``seg-dpt`` (YAML ``segmentation: {head: dpt}``).
A DPT-style reassemble + fusion-transformer decoder
(:class:`~torchgeo_bench.models.DPTHead`).  Requires exactly four
backbone layers in coarse-to-fine order; otherwise structurally
similar to the FPN probe but with deeper fusion blocks.

Training & evaluation (all heads)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Optimiser:** AdamW, applied **only** to unfrozen probe parameters.
* **Loss:** ``CrossEntropyLoss(ignore_index=255)`` so unlabeled pixels
  are excluded from both loss and metric computation.
* **Schedule:** cosine decay to 1e-6 by default
  (``segmentation.scheduler``); ``none`` disables.
* **Weights evaluated:** the final epoch by default. With
  ``segmentation.early_stopping.enabled``, training stops once validation mIoU
  stops improving and the best validation check is evaluated instead.
* **Metric:** mean Intersection-over-Union (mIoU) via
  ``torchmetrics.MulticlassJaccardIndex``.  Frequency-weighted IoU plus
  macro precision / recall / F1 are also reported in the result row
  (see :doc:`results-format`).
* **Uncertainty:** 95% bootstrap confidence intervals resample held-out
  images and recompute dataset-level mIoU from their aggregated confusion
  matrices.

Segmentation knobs
------------------

All keys live under ``segmentation`` in run YAML or model presets.
Defaults are declared in :class:`~torchgeo_bench.config.schema.SegmentationConfig`.
Explicit run settings override preset defaults, including empty layer lists.

Head type
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``head``
     - Description
   * - ``linear``
     - Per-layer BN + 1×1 conv → upsample.  Multi-layer fused with
       learned scalar weights.
   * - ``conv_block``
     - Per-layer 1×1 projection to ``hidden_dim`` → upsample + concat
       → 1×1 head.
   * - ``fpn``
     - FPN top-down pathway.  Layers must be coarse-to-fine.
   * - ``dpt``
     - DPT-style reassemble + fusion-transformer decoder.

Training knobs
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Option
     - Default
     - Description
   * - ``layers``
     - *(per model)*
     - Backbone layer names to hook.  For FPN / DPT, deepest layer first.
   * - ``epochs``
     - ``10``
     - Training epochs for the probe head.
   * - ``learning_rate``
     - ``1e-3``
     - Initial learning rate (AdamW).
   * - ``learning_rates``
     - ``[]``
     - Optional grid searched instead of ``learning_rate``. Each rate refits
       the same initial head with the same batch order; the rate with the
       best validation mIoU is kept (ties go to the lower rate) and recorded
       as ``best_lr``.
   * - ``scheduler``
     - ``cosine``
     - ``cosine`` (CosineAnnealingLR to 1e-6) or ``none`` (constant).
   * - ``ignore_index``
     - ``255``
     - Unlabeled mask value excluded by cross entropy and segmentation metrics.
   * - ``batch_size``
     - ``64``
     - Batch size when training the probe head.

Validation early stopping
^^^^^^^^^^^^^^^^^^^^^^^^^

Opt-in, under ``segmentation.early_stopping``. Requires ``scheduler: none``
and ``cache_features: true``. Validation mIoU is checked at epoch 0 and every
``check_every`` epochs. Patience resets only after a gain of more than
``min_delta`` over the last reference score, so a run of tiny gains cannot
postpone stopping. When disabled, these settings are left out of the resume
fingerprint, so existing results still resume.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Option
     - Default
     - Description
   * - ``enabled``
     - ``false``
     - Turn validation early stopping on.
   * - ``check_every``
     - ``5``
     - Epochs between validation checks.
   * - ``patience``
     - ``16``
     - Checks without a ``min_delta`` gain before stopping.
   * - ``min_delta``
     - ``0.001``
     - Required cumulative mIoU gain (a fraction, so 0.1 points).
   * - ``min_epochs``
     - ``25``
     - No stopping before this epoch.
   * - ``max_epochs``
     - ``1000``
     - Hard cap; replaces ``epochs`` while early stopping is enabled.

Feature caching
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 18 57

   * - Option
     - Default
     - Description
   * - ``cache_features``
     - ``true``
     - Pre-extract backbone features once per split into RAM.  Stored
       layer-first as contiguous ``(N, C, H, W)`` ``float16`` tensors
       (:class:`~torchgeo_bench.segmentation_probe.CachedFeaturesDataset`).
       GPU transfer is a single memcpy per layer.  Eliminates backbone
       re-runs across epochs — the dominant speedup.
   * - ``cache_dtype``
     - ``float16``
     - Storage dtype for cached features.  ``float16`` halves RAM;
       autocast upcasts during the head forward pass.

Common configuration
--------------------

All evaluation paths share these settings:

.. list-table::
   :header-rows: 1
   :widths: 25 18 57

   * - Setting
     - Default
     - Description
   * - ``runtime.seed``
     - ``0``
     - Random seed for reproducibility (numpy + torch).
   * - ``runtime.device``
     - ``cuda:0``
     - PyTorch device.
   * - ``runtime.batch_size``
     - ``64``
     - Batch size for data loading.
   * - ``input.image_size``
     - ``224``
     - Resize input images (``null`` = preserve native size).
   * - ``input.interpolation``
     - ``bilinear``
     - Resize interpolation method.
   * - ``resume``
     - ``false``
     - Skip already-computed
       ``(dataset, method, model, …)`` combinations.

Resume mode and output schema
-----------------------------

When ``resume=true`` the runner reads the existing CSV at startup and
skips any combination already present.  See :doc:`results-format` for
the full key, the column schema, and the atomic-append guarantee.
