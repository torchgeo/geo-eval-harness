Configuration
=============

All commands use explicit flags and strict Pydantic-validated YAML.
``geo-eval-harness``, ``python -m geo_eval_harness``, and
``python -m geo_eval_harness.cli`` expose the same interface.

Commands that use ``docs/examples/`` paths run from the repository root. With an installed package, save the example YAML locally and pass that path to ``--config``.

For example:

.. code-block:: console

   $ geo-eval-harness run --model timm/resnet50 --dataset m-eurosat \
       --methods knn --bootstrap-samples 100 --device cpu
   $ geo-eval-harness run --config docs/examples/image-run.yaml --methods linear --dry-run
   $ geo-eval-harness run --config-help

``--dry-run`` validates selections and prints reusable YAML without loading weights or dataset samples. For image runs, it checks explicit band names against each dataset's metadata and rejects model-native normalization when the preset declares it unsupported, as RCF does. These checks use resolved model and dataset defaults without writing those defaults into the output YAML. Custom model targets are not imported; constructor-specific requirements, including RGB-only inputs and weight-specific normalization, may still fail at runtime.

``run --config-help`` and ``flops --config-help`` print JSON schemas; each command's ``--help`` lists its flags.

Precedence and validation
-------------------------

Settings are resolved independently for each dataset, in this order from
lowest to highest priority:

1. Built-in schema defaults.
2. Model preset defaults.
3. The preset's defaults for that dataset.
4. Explicit values in the supplied YAML.
5. Explicit command-line flags.

Omitting a field is different from explicitly setting its default. For
example, omitting ``segmentation.layers`` keeps the preset's layers, while
``layers: []`` replaces them with an empty selection. ``image_size: null``
disables resizing, and ``cache_features: false`` disables feature caching.
Neither is silently replaced by a preset value. Mappings merge recursively;
lists and scalar values replace earlier values.

YAML must be a mapping with unique keys and known fields. Types are strict:
write ``false``, not ``"false"``, and numeric values without quotes.
Scientific notation such as ``1e-3`` is supported. Constructor options inside
``model.kwargs`` are passed to the selected constructor; they are not
additional benchmark configuration fields.

All four command configurations use integer ``schema_version: 1`` by default. If supplied, the version must be the integer ``1``; ``1.0``, booleans, strings, and other versions are rejected.

Image runs
----------

A minimal run file selects a model and one or more datasets:

.. code-block:: yaml

   model:
     name: timm/resnet50
   datasets: [m-eurosat]

Run it with ``geo-eval-harness run --config my-run.yaml``. A model selection is
a mapping with a ``name`` field.
Use ``datasets: [all]`` to select the image dataset catalog. On the CLI,
``--dataset`` is repeatable:

.. code-block:: console

   $ geo-eval-harness run --model timm/resnet50 \
       --dataset m-eurosat --dataset m-pv4ger --device cpu

The complete field example is :file:`docs/examples/image-run.yaml`. It explicitly
sets defaults for illustration; remove fields that should inherit preset
settings.

Input and runtime
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   input:
     bands: rgb
     partition: default
     time_steps: null
     image_size: 224
     interpolation: bilinear
     normalization: dataset
   runtime:
     device: cuda:0
     batch_size: 64
     workers: 4
     seed: 0
     verbose: false
     sample_cache: none
     sample_cache_dir: ~/.cache/geo_eval_harness/samples

``bands`` accepts ``rgb``, ``all``, or an ordered YAML list of dataset band
names. The equivalent flag accepts comma-separated names, for example
``--bands red,green,blue``. Band names differ by dataset; see :doc:`datasets`.
``time_steps`` selects a positive temporal length where supported.
``image_size: null`` (``--image-size none``) retains native dimensions.
Interpolation is ``area``, ``bilinear``, ``bicubic``, or ``nearest`` and applies to images; segmentation masks are always resized with ``nearest-exact``, which samples each output pixel at its centre like ``bilinear`` and ``bicubic`` do.

``runtime.sample_cache`` serves segmentation splits from a copy of the preprocessed samples (after band selection and resizing), so each sample is decoded once instead of once per epoch. ``disk`` writes ``.npy`` memory maps under ``sample_cache_dir`` (built once per dataset, split, bands, size, interpolation, and code/data fingerprint; concurrent runs share them through the OS page cache), ``host`` keeps them in process memory, and ``gpu`` on the training device. Images stay float32 and batch order matches the uncached loader exactly, so the setting does not change results or ``config_hash``. ``disk`` needs free disk space and benefits from free RAM; ``host`` and ``gpu`` need the whole split in RAM or device memory.

For ``run`` and ``profile``, ``--image-size`` accepts a positive integer or case-insensitive ``none``. FLOPs measurements require a concrete positive image size.

Normalization choices map to the model interface as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - YAML / flag
     - Model strategy
     - Meaning
   * - ``dataset``
     - ``bandspec_zscore``
     - Per-band dataset statistics.
   * - ``model``
     - ``model_native``
     - Model-declared input units and pretraining statistics.
   * - ``minmax``
     - ``minmax``
     - Per-band scaling using dataset ``BandSpec.min`` and ``BandSpec.max``.
   * - ``minmax_zscore``
     - ``minmax_zscore``
     - Min-max scaling followed by z-scoring with the scaled dataset statistics.
   * - ``none``
     - ``identity``
     - No normalization.

``model`` requires a wrapper that declares its expected units; it is not an
automatic substitute for dataset normalization. Wrappers that own their
normalization retain their documented behavior.

Torch execution uses one device policy across image runs, profiling, synthetic compute measurements, coordinate encoders/probes, and intrinsic-dimension estimation. ``cpu`` uses CPU without checking CUDA. ``auto`` selects the current CUDA device when available, otherwise CPU. Explicit ``cuda`` selects the current CUDA device; ``cuda:N`` selects index N. Explicit CUDA requests raise an error when CUDA is unavailable or the index is invalid. Use ``auto`` to allow CPU fallback. The image default is ``cuda:0``.

Runtime flags are ``--device``, ``--batch-size``, ``--workers``,
``--seed``, and ``--verbose`` / ``--no-verbose``. Input flags are ``--bands``,
``--partition``, ``--time-steps``, ``--image-size``, ``--interpolation``, and
``--normalization``.

Classification
^^^^^^^^^^^^^^

.. code-block:: yaml

   classification:
     methods: [knn, linear]
     knn_k: 5
     knn_device: null
     bootstrap_samples: 200
     feature_standardize: false
     linear:
       c_log10_start: -6.0
       c_log10_stop: 4.0
       c_count: 40
       refit_train_val: true
     calibration:
       n_bins_knn: null
       n_bins_linear: 15
       temp_scale: false

Select ``--methods knn``, ``--methods linear``, or ``--methods knn linear``.
Linear-only runs do not require a KNN result. ``--knn-k``,
``--knn-device``, and ``--bootstrap-samples`` override the corresponding
fields. A null KNN device inherits the feature-extraction device; the backend
logs a CPU fallback if GPU KNN is unavailable.

The linear probe selects regularization on validation data from ``c_count``
log-spaced values between the two log10 endpoints. It then optionally refits
on train plus validation. ``--refit-train-val`` /
``--no-refit-train-val`` control that final refit. Temperature scaling
(``--temp-scale`` / ``--no-temp-scale``) requires linear probing and
``refit_train_val: false`` so validation logits stay held out.
``n_bins_knn: null`` uses ``knn_k + 1`` calibration bins.

``feature_standardize: true`` z-scores every embedding dimension with a mean and standard deviation fit on the training split, ``(x - mean) / sqrt(var + 1e-6)``, before KNN and the linear probe. Results are written under the methods ``knn<k>-std`` and ``linear-std``. Intrinsic-dimension and spectrum measurements use the unscaled embeddings.

Segmentation
^^^^^^^^^^^^

.. code-block:: yaml

   segmentation:
     head: fpn
     neck: none
     neck_norm: layernorm
     upsampler: none
     weight_decay: 0.0
     head_dropout: 0.0
     layers: []
     learning_rate: 1e-3
     epochs: 10
     early_stopping_patience: null
     batch_size: 64
     temporal_pool: mean
     scheduler: cosine
     ignore_index: 255
     cache_features: true
     cache_dtype: float16
     amp: fp16
     lora_rank: null
     lora_target_modules: null
     lora_alpha: 16.0
     lora_last_blocks: null
     backbone_learning_rate: null
     full_finetune: false
     feature_standardize: false
     gradient_checkpointing: false
     sweep_learning_rates: []
     sweep_weight_decays: [0.0]

These YAML settings apply to segmentation datasets, independently of
``classification.methods``. Supported heads are ``linear``, ``conv_block``,
``fpn``, ``dpt``, ``patch_linear``, ``upernet``, and ``unet``; not every head works with every
backbone. ``unet`` (:class:`~geo_eval_harness.models.UNetHead`) is a U-Net decoder with ``hidden_dim`` channels per stage that stops at the finest hooked map, like ``upernet``; it needs a feature pyramid, so ViT blocks require ``neck: vit``. ``neck: vit`` adds the ViT neck (:class:`~geo_eval_harness.models.ViTNeck`) before a ``linear``, ``fpn``, ``upernet``, or ``unet`` head. It takes either one feature map or four layers that share one resolution and is recorded as ``neck: vit`` in the result row (see :ref:`segmentation-setup-columns`). ``neck_norm: layernorm`` (default) applies a learnable per-position LayerNorm to each level before the neck; ``none`` reproduces the neck without it. ``upsampler: anyup`` (``upernet`` only, ``cache_features: false``) puts frozen pretrained AnyUp between the fused features and the classifier, guided by the input's RGB bands, recorded as ``upsampler: anyup`` in the result row; the pinned code and checkpoint download through ``torch.hub``. Omit ``layers`` to inherit the model's selection. An explicit
``[]`` clears it and is rejected by image segmentation, which requires explicit
feature layers. See
:doc:`segmentation-layers` for verified layer paths.

``runtime.batch_size`` controls image loading, feature extraction, and uncached training. ``segmentation.batch_size`` controls training on cached features. ``temporal_pool`` is
``mean`` or ``max``, ``scheduler`` is ``cosine`` or ``none``, and the cache
dtype is ``float16`` or ``float32``. The loss is cross entropy with the
configured ``ignore_index``; there is no constructor-style criterion block.

``amp: fp16`` (default) trains and evaluates under CUDA float16 autocast with gradient scaling,
runs a frozen backbone under autocast, and keeps device feature caches in float16. Backbones
whose activations exceed the float16 range (about 65,504) overflow to ``inf``; ``amp: off`` runs
all of this in float32 at the cost of speed and memory. With cached frozen features it requires
``cache_dtype: float32``. Whatever the precision, a training
batch whose backbone features are not finite is skipped before the head runs, so head
BatchNorm statistics stay finite, and counted as ``nonfinite_batches`` in the epoch record; an
epoch that skips more than 5% of its batches fails. ``feature_standardize`` leaves such samples
out of its statistics under the same limit. Non-finite features or logits during evaluation fail
the run instead of scoring a constant prediction.
Segmentation confidence intervals use ``classification.bootstrap_samples``.

See :ref:`segmentation-model-selection` for the validation requirements, model-selection policy, and optional ``early_stopping_patience``.

Set ``lora_rank`` and ``backbone_learning_rate`` to fine-tune with LoRA adapters; this requires the ``lora`` extra. ``lora_target_modules`` is a regular expression that fully matches the adapted ``nn.Linear`` module names in the same namespace as ``layers``. Presets that support LoRA define the targets. Adapter updates are scaled by ``lora_alpha / lora_rank``.

``lora_last_blocks: k`` restricts adapters to the last ``k`` blocks. A block is the module path through its last numeric index, such as ``blocks.11`` or ``stages.3.blocks.26``. Earlier blocks stay frozen and skip the backward pass.

``full_finetune: true`` trains every backbone weight at ``backbone_learning_rate``. Both fine-tuning modes keep pretrained dropout and normalization statistics in eval mode. Result rows record ``adaptation: lora`` or ``adaptation: full``. Without LoRA, keep ``lora_alpha`` at its default and leave ``lora_last_blocks`` unset; set ``backbone_learning_rate`` only for a trainable backbone.

Fine-tuning reads images each epoch, so ``cache_features`` does not apply and ``runtime.batch_size`` controls training batches. A model whose forward pass disables gradients, for example with ``@torch.no_grad()``, is rejected at the first training step because its backbone would not train. See :ref:`segmentation-setup-columns` for the recorded settings.

``gradient_checkpointing: true`` checkpoints the backbone's transformer blocks while LoRA or full fine-tuning trains: their activations are recomputed in the backward pass instead of being stored, which cuts activation memory at the cost of about one extra backbone forward per step. Losses and gradients are unchanged up to floating-point reordering. It is supported for timm SwinV2 (timm's ``set_grad_checkpointing``), TerraTorch Prithvi-EO and OlmoEarth (each block wrapped with ``torch.utils.checkpoint``, ``use_reentrant=False``); any other backbone raises. It has no effect on a frozen backbone. The setting is recorded in ``config_hash``.

.. code-block:: yaml

   segmentation:
     head: upernet
     lora_rank: 8
     backbone_learning_rate: 1e-4

``feature_standardize: true`` z-scores every hooked feature channel with a mean and standard deviation fit on the training split, ``(x - mean) / sqrt(var + 1e-6)``, before the head (method suffix ``-std``). The statistics use every training position, from the feature cache or, with ``cache_features: false``, from one extra backbone pass over the training split. It requires a frozen backbone.

``sweep_learning_rates`` (with ``head: patch_linear``, no neck, frozen backbone) trains one LayerNorm + linear ``patch_linear`` probe per (learning rate, weight decay) cell of ``sweep_learning_rates`` x ``sweep_weight_decays`` in one vectorised pass, each with its own AdamW group and a cosine decay to zero. Every head keeps its best validation epoch; the head with the highest best-epoch validation mIoU is restored and alone scored on test. ``learning_rate`` and ``weight_decay`` are ignored; the selected values are written to ``best_lr`` and the ``selected`` epoch-metrics event. The method gains ``-sweep<n>`` with ``n`` cells. Together they reproduce UniverSat's ``LP_eval.py`` segmentation probe:

.. code-block:: yaml

   segmentation:
     head: patch_linear
     layers: [blocks.11]
     feature_standardize: true
     sweep_learning_rates: [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
     sweep_weight_decays: [0.0, 0.01, 0.1]
     epochs: 50

Optional image measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   profile:
     enabled: false
     n_warmup: 3
     n_measure: 20
     cpu_throughput:
       enabled: false
       batch_size: 8
       n_warmup: 1
       n_measure: 5
       time_budget_s: 300.0
   intrinsic_dim:
     enabled: false
     estimators: [TwoNN, MLE, lPCA]
     splits: [train]
     max_samples: 10000
     device: null

These are additive passes within an image run, not the standalone
``profile`` command. Intrinsic-dimension estimators require the ``id`` extra;
``estimators: []`` requests only the dependency-free feature-spectrum
diagnostics. ``splits`` accepts ``train``, ``val``, and ``test``.
``max_samples: null`` removes the sample cap.

Results and resume
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   output:
     directory: results
     file: null
     resume: false
     epoch_metrics: null

``--output-dir`` sets ``output.directory``, the common root for all command outputs. Image runs append classification and segmentation CSVs under ``models/``, profiling CSVs under ``profiles/``, and intrinsic-dimension CSVs under ``intrinsic_dim/``. Filenames remain ``<model name>.csv``.

To keep every output from an image run outside the tracked reference results:

.. code-block:: bash

   geo-eval-harness run --config docs/examples/image-run.yaml \
     --output-dir outputs/my-run

The same flags apply to ``flops``, ``coord``, and standalone ``profile``. Beneath the root, they write ``compute_cost.csv``, ``coordbench_results.csv``, and ``profile.json``, respectively. The CSV commands default to the ``results`` root; standalone profiling defaults to stdout when neither destination is supplied.

``--output`` / ``output.file`` selects an exact path, relative to the working directory or absolute. An explicit file takes precedence over the directory, even when the file comes from YAML; clear ``output.file`` to ``null`` to use the directory instead. For image runs, the explicit file combines all selected measurement kinds in one CSV. Each CLI flag overrides its corresponding YAML field, regardless of flag order.

``output.epoch_metrics`` names a JSON Lines file for per-epoch segmentation training records. It is unset by default and has no CLI flag; see :ref:`segmentation-epoch-metrics`.

``--resume`` / ``--no-resume`` control skipping completed measurements.
Resume compares method/config keys and metric completeness; additive profile
or intrinsic-dimension selections do not invalidate otherwise equivalent
classification rows. Completed evaluations are appended atomically before
later evaluations run. See :doc:`results-format` for stored columns and keys.

Model presets and custom models
--------------------------------

``geo-eval-harness models`` lists packaged presets and
``geo-eval-harness models timm/resnet50`` displays one. Presets live under
:file:`src/geo_eval_harness/conf/model/`, including ``timm/`` and ``torchgeo/``
subdirectories. They are validated separately from run files:

.. code-block:: yaml

   name: my-model
   target: my_package.MyBenchModel
   track: image
   seed_from_run: false
   kwargs:
     feature_dim: 256
   input:
     image_size: 224
   classification:
     linear:
       c_count: 20
   segmentation:
     head: linear
   dataset_overrides:
     caffe:
       input:
         image_size: 128

Only ``kwargs`` are constructor options. ``name``, ``target``, ``track``,
``seed_from_run``, preprocessing/evaluation defaults, and dataset overrides
are preset metadata. ``seed_from_run: true`` explicitly injects the run seed
for constructors that support it, without interpolation.

For a one-off custom model, a run file can instead contain:

.. code-block:: yaml

   model:
     name: my-model
     target: my_package.MyBenchModel
     kwargs:
       feature_dim: 256
   datasets: [m-eurosat]

The target must be importable in the active Python environment. There is no
recursive instantiation inside ``kwargs``. Runtime ``BandSpec`` objects and
empirical-RCF datasets are passed explicitly to ``build_model``, never
serialized into YAML. See :doc:`/api/cli` for typed loading and resolution,
and :doc:`eval_own_model` for the model interface.

Standalone profiling
--------------------

Profiling preserves acquisition dates supplied by the dataset, including during FLOP counting and CPU measurements. Standalone JSON records include ``doy_sha256`` when date metadata is present.

``profile`` repeatedly measures one fixed real dataset batch, with the same model/dataset input-default resolution as an image run. It writes one JSON record to stdout by default. Use ``--output-dir`` to write ``profile.json`` beneath a directory, or ``--output`` for an exact filename; file output leaves stdout empty and replaces any existing JSON file.

.. code-block:: console

   $ geo-eval-harness profile --model rcf --dataset m-eurosat --device cpu \
       --batch-size 8 --warmup 1 --measurements 5 > profile.json
   $ geo-eval-harness profile --config docs/examples/profile.yaml --dry-run
   $ geo-eval-harness profile --model rcf --dataset m-eurosat --output-dir outputs/profile

Its YAML uses singular ``dataset`` and top-level measurement settings:

.. code-block:: yaml

   model:
     name: rcf
   dataset: m-eurosat
   input:
     bands: rgb
   runtime:
     device: cpu
     batch_size: 8
     seed: 0
   warmup: 1
   measurements: 5
   precision: float32
   count_flops: false
   output:
     directory: null
     file: null

It supports ``--model``, ``--dataset``, ``--device``, ``--batch-size``,
``--seed``, ``--bands``, ``--partition``, ``--image-size``,
``--interpolation``, and ``--normalization``. Timing options are
``--warmup`` and ``--measurements``. ``--precision`` is ``float32``,
``float16``, or ``bfloat16``. Reduced precision uses PyTorch autocast;
operation support depends on the model and device.
``--count-flops`` / ``--no-count-flops`` add or omit FLOP counting.
The JSON includes effective preprocessing, device, precision, timing, and
memory metadata. Local dataset samples must already be available.
Profiling supports only CPU and CUDA measurement devices.

Synthetic compute measurements
-------------------------------

``flops`` measures backbone and probe compute from synthetic tensors and
appends to ``results/compute_cost.csv`` by default. It reads dataset band
metadata but not samples. Model weights may still need downloading unless
the selected model is an offline baseline or already cached.

.. code-block:: console

   $ geo-eval-harness flops --config docs/examples/flops.yaml --dry-run
   $ geo-eval-harness flops --model rcf --device cpu --band-configs rgb \
       --seg-heads --output results/my_compute_cost.csv

Its YAML schema is independent of image runs:

.. code-block:: yaml

   model:
     name: rcf
   runtime:
     device: cpu
     seed: 0
     verbose: true
   input:
     band_source: cloudsen12
     band_configs: [rgb, s2]
     image_size: 224
     normalization: dataset
   classification:
     head: linear
     num_classes: 10
   segmentation:
     heads: [fpn, dpt]
     band_configs: [rgb, s2]
     num_classes: 4
   timing:
     batch_size: 64
     n_warmup: 3
     n_measure: 20
   output:
     directory: results
     file: null
     resume: true

``--band-source`` selects the dataset metadata. ``--band-configs rgb all`` selects RGB or all source bands. ``s2`` also selects all source bands, including twelve Sentinel-2 bands for the default CloudSen12 source. Use ``all`` for SAR or other sensor inputs so the CSV labels the selection correctly.
``--image-size`` must be positive; synthetic inputs cannot use native
dataset dimensions. ``--normalization`` has the same five choices as image
runs. ``--probe-head`` is ``linear`` or ``mlp``, and
``--probe-num-classes`` sets the classification output width.

Use ``--seg-heads`` with any of ``linear``, ``conv_block``, ``fpn``, ``dpt``, ``patch_linear``, ``upernet``, and ``unet``; supplying the flag without values disables segmentation. ``--seg-band-configs`` likewise accepts an empty selection.
``--seg-num-classes``, ``--seg-layers``, and ``--temporal-pool`` configure
segmentation accounting. YAML may include a ``segmentation.probe`` block
with the image segmentation fields. Omitted probe layers inherit model
and dataset defaults; explicit ``[]`` clears them.

``segmentation.neck_policy: auto`` (``--seg-neck-policy auto``) inspects the configured feature maps to choose the ViT pyramid neck for one or four equal-resolution transformer maps, or the ``vit_fine`` neck for three equal-resolution maps followed by a finer map. Native feature pyramids retain their spatial scales. An explicit non-default ``probe.neck`` is preserved. The linear segmentation head also accepts the ViT neck.

Segmentation rows count the complete backbone, neck, and decoder forward pass and time that same inference path in float32. ``gflops_head`` includes the neck; ``gflops_total`` includes the backbone, neck, and head. Rows record the actual hooked ``feature_shapes``, ``layers``, ``neck``, GPU name, precision, and timing batch size. ``n_tokens`` is omitted when the model defines patch size in metres instead of pixels. FLOP totals cover registered PyTorch operators, so unregistered operations can make them lower bounds.

Resume identifies the resolved model, settings, hardware, and image size with ``config_hash``. Changing resolution, probe settings, timing, or GPU starts a distinct measurement.

``python experiments/run_compute_cost_sweep.py --image-sizes 224 256 512 --heads linear upernet unet --device cuda:0 --output results/compute_sweep`` runs the current image-model catalog sequentially on one GPU. It uses batch size 1, five warmup passes, twenty timed passes, four output classes, and automatic neck selection. Wrapper resize settings and fixed encoder image grids, including Clay's positional grid, follow the requested image dimensions; the encoder's learned patch stride stays unchanged. Swin models use timm's dynamic window padding (``model.kwargs.dynamic_img_size: true``) to support grids that do not divide their pretrained attention windows. Panopticon uses the nearest multiple of its 14-pixel patch stride (224, 252, and 518 for the default sweep); CSV rows label those actual image sizes. EarthLoc is measured once at its native 320-pixel input because its learned spatial aggregation fixes that grid. TerraFM keeps the pretrained position tensor and uses its forward-time interpolation at the requested image size; wrapper resizing is disabled. Unsupported image grids remain missing cells with their errors in the logs. AnySat and UniverSat use their native sensor GSD, so increasing image size increases ground extent instead of inheriting dataset overrides that assume a fixed-tile resize. ``manifest.json`` lists the requested cells and benchmark names without matching configurations. ``coverage.json`` records measured and missing cells for every attempted model/band/size job. Models without declared spatial taps receive classification measurements only. A failed decoder does not prevent attempts for the other decoders.

``--timing-batch-size``, ``--n-warmup``, and ``--n-measure`` control timing;
FLOPs are always counted for one sample. ``--output-dir`` relocates ``compute_cost.csv``, ``--output`` selects an exact CSV path, and ``--no-resume`` disables the default per-cell resume behavior.
``--model-target`` and ``--model-kwargs`` accept a custom constructor and a
YAML mapping of constructor options; reusable custom models can use
``model.target`` and ``model.kwargs`` in the file instead.

Coordinate evaluation
----------------------

``coord`` has its own ``model``, ``datasets``, ``evaluation``, ``runtime``,
and ``output`` sections. It does not accept image-run ``input`` or
``classification`` blocks:

.. code-block:: console

   $ geo-eval-harness coord --model sincos --dataset california_housing \
       --methods linear --folds 2 --device cpu
   $ geo-eval-harness coord --config docs/examples/coord-run.yaml --dry-run

See :doc:`coordbench` for the full YAML, encoder targets, method selection,
random/spatial/official splits, and coordinate CSV resume semantics.
