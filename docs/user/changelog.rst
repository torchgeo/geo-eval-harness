Changelog
=========

Unreleased
----------

Changed
^^^^^^^

* Segmentation ``method`` is ``seg-<head>`` for every setup. Result rows gain setup columns (``adaptation``, ``decoder``, ``neck``, ``upsampler``, ``epochs``, ``head_lr``, ``backbone_lr``, ``weight_decay``, ``head_dropout``, ``lora_rank``, ``lora_alpha``, ``lora_last_blocks``, ``class_weights``, ``selection_metric``, ``early_stopping_patience``) that join the de-duplication key of ``load_results``; ``config_hash`` and the resume key are unchanged. ``scripts/migrations/migrate_setup_columns.py`` adds the empty columns to existing CSVs; ``results/models`` is migrated. See :ref:`segmentation-setup-columns`.
* The ranking explorer (``scripts/regen_leaderboard.py``) builds segmentation views from the setup columns: Adaptation (frozen / LoRA / full), Decoder (UPerNet / U-Net) and Pyramid (native / ViT neck / AnyUp) selectors, a per-dataset canonical protocol (``SEG_PROTOCOL``) with a "Show variants" toggle for other epoch and learning-rate setups, and seed mean ± sd with the seed count. Legacy segmentation rows without setup columns are no longer shown. Explicit ``red,green,blue`` band lists count as RGB.
* The ranking explorer keeps segmentation input sizes apart: ``image_size`` joins the seed-aggregation group, and a row whose size differs from the dataset's canonical size (``SEG_IMAGE_SIZE``, 224 px on every dataset for now) is an ``image_size=<n>`` protocol variant, so native-resolution rows (for example 512 px SpaceNet) are never averaged with 224 px seeds or shown in the default view. Rows with no recorded size are ``image_size=unset``. Current leaderboard output is unchanged.
* The ranking explorer adds a segmentation **Native bands** band class: each model is ranked on its own input bands per dataset, its per-modality configurations merged (DEO Swin RGB/S2, TerraFM S1/S2; the better cell counts where both cover a dataset), on the datasets it has (average rank and improvability; no ELO). A **Default** pyramid (each model's native features or ViT neck) puts hierarchical and ViT backbones in one view.
* The results explorer (``scripts/regen_results_explorer.py``) carries the setup columns, ``best_val_epoch`` and ``best_val_miou`` in its snapshots, keeps only segmentation rows with setup columns, and offers ``seg-<head>`` by adaptation in its method menus.
* ``open_cities`` now trains on a fixed 10,000-chip seed-0 subset by default. ``--partition full`` restores all 76,818 training chips; val and test are unchanged. Earlier Open Cities rows with ``partition=default`` used the full split, so use a fresh output file to avoid incorrect resume matches. See :doc:`datasets`.
* UniverSat and AnySat share their torch.hub helpers (pinned source, sensor grouping, per-pixel chunking, block checkpointing). Neither model's outputs nor its hashes change. UniverSat now refuses to load in a process that already imported AnySat's hub code: both repos use a top-level ``src`` package, and the second model would otherwise silently reuse the first one's modules.
* Package, CLI, and repository name: ``geo-eval-harness``. Python imports and fully qualified model targets use ``geo_eval_harness``. Stored results and resume hashes remain compatible; update custom imports and YAML targets as described in :doc:`installation`.
* TerraTorch models use a separate source installation because the published backend's TorchGeo requirement is incompatible. See :doc:`installation` for the pinned command.
* ``hotosm_buildings`` trains on a fixed 5,000-tile subset of the official train split by default. This is the benchmark and leaderboard version. ``--partition full`` restores the whole 57,890-tile train split; val and test are unchanged. HOTOSM results produced before this change used the full split and correspond to ``--partition full``, although their result rows say ``partition=default``. Write new HOTOSM runs to a fresh output file so resume does not treat those rows as done. See :doc:`datasets`.

Added
^^^^^

* Date-aware backbones can receive each sample's acquisition day of year (``doy``) through ``BenchModel.forward_batch``. PASTIS snapshots (``time_steps`` unset or 1) carry the date of the Sentinel-2 frame the loader selects, and feature extraction, the segmentation probe and solver, and the host, GPU, and disk sample caches pass it through. Other models ignore it; their results and ``config_hash`` values are unchanged. PASTIS sample caches are rebuilt once.
* ConvNeXt DINOv3 Tiny, Small, and Base presets (``timm/convnext_{tiny,small,base}_dinov3``, timm ``*.dinov3_lvd1689m``) with the same segmentation layers and LoRA targets as ``timm/convnext_large_dinov3``. ``timm/convnext_small`` gains the ConvNeXt LoRA targets; frozen and full fine-tuning hashes are unchanged.
* AnySat v1 (Base) for classification and segmentation (``anysat_base``, ``anysat_base_seg``). Model code and weights are pinned, and the weights are checked against their sha256.
  - Each input is resampled to its projector's GSD over the tile's ground extent (``input_gsd``), with patch sizes in metres.
  - Bands are mapped by name. Missing bands are imputed only with ``impute_missing``.
  - The S1 VV/VH ratio is computed from raw values and z-scored with stored training-split ``ratio_stats``.
  - Segmentation reads the output tokens through ``output_tap``.
  - LoRA targets fusion blocks 0-5 and the cross-attention block.
  - The presets refuse TreeSatAI and FLAIR (pretraining overlap) and sub-metre SAR.
  See :doc:`models`.
* ``segmentation.amp`` (``fp16`` default, or ``off``) switches segmentation training and evaluation between CUDA float16 autocast and float32 for backbones whose activations overflow float16. ``fp16`` leaves hashes unchanged; ``off`` with cached frozen features requires ``cache_dtype: float32``. Epoch records gain ``nonfinite_batches``.
* Classification dataset classes declare ``class_names`` in label-index order, so per-class results can be labelled without loading data.
* ``bright-post-sar-rgb`` repeats BRIGHT's post-event SAR channel into identical ``red``/``green``/``blue`` bands (sensor ``aerial``, no wavelengths) for RGB model inputs. It shares ``bright-post-sar`` tiles, labels, splits, normalization statistics, and installation; the single-band dataset is unchanged.
* Linear-probe protocol (UniverSat ``LP_eval.py``) for any frozen backbone, off by default with unchanged hashes. ``segmentation.sweep_learning_rates`` x ``sweep_weight_decays`` trains one ``patch_linear`` head per cell in a single vectorised pass, each with its own AdamW group and a cosine decay to zero, and evaluates only the head with the best validation epoch on test (the selected learning rate and weight decay are reported; method suffix ``-sweep<n>``). ``segmentation.feature_standardize`` z-scores each hooked feature channel with exact training-split statistics, from the feature cache or one extra backbone pass (method suffix ``-std``).
* ``segmentation.weight_decay`` (AdamW, all trainable parameters) and ``segmentation.head_dropout`` (``Dropout2d`` before the UPerNet classifier) are configurable; both default to ``0`` and leave hashes unchanged.
* CaFFe serves its grayscale SAR channel as identical ``red``/``green``/``blue`` bands (sensor ``aerial``, no wavelengths), so every model uses its standard RGB setup; earlier CaFFe results used one channel and are not comparable.
* OpenEarthMap-SAR accepts opt-in ``vv``/``vh`` band aliases that feed its single ``sar`` channel (same statistics), so Sentinel-1-only models can run it; the default selection stays ``sar``.
* OpenEarthMap-SAR (``openearthmap-sar``): SAR-only eight-class segmentation on the official 4,333 / 210 / 490 splits, verified multi-record downloads, training-split normalization statistics, and GeoTIFF-derived geography. Native zero is ignored in loss and metrics.
* ``segmentation.upsampler: anyup`` adds frozen pretrained AnyUp (pinned ``wimmerth/anyup``, checkpoint sha256-verified) before the UPerNet classifier, guided by the input's RGB bands; ``none`` leaves hashes unchanged.
* ``segmentation.full_finetune`` trains the whole backbone at ``backbone_learning_rate`` as a reference regime.
* ``runtime.sample_cache`` (``disk``, ``host``, ``gpu``) serves segmentation batches from an exact cache of preprocessed samples; results and hashes are unchanged.
* ``segmentation.gradient_checkpointing`` recomputes backbone block activations in the backward pass during LoRA and full fine-tuning (timm SwinV2, TerraTorch Prithvi-EO, OlmoEarth), so large backbones fit at 512 px with an unchanged batch size; unset leaves hashes unchanged.
* ``segmentation.lora_last_blocks`` restricts LoRA adapters to the deepest ``k`` backbone blocks; unset leaves hashes unchanged.
* UniverSat supports segmentation and LoRA through the ``universat_base_seg`` preset: ``patch_px`` sets the patch size in input pixels (16 gives a 14×14 token grid at 224 px), and UPerNet reads self-attention blocks 3/6/9/12 with the register tokens removed.
* Open Cities AI Challenge building segmentation (``open_cities``): tier-1 drone scenes of
  seven African cities, cut into 512x512 chips on each scene's native grid (0.02-0.20 m GSD).
  Train/val/test are assigned per 500 m geographic block and fixed in a packaged index
  (76,818 / 11,236 / 22,408 chips). Download with ``geo-eval-harness download open_cities``.
* ``scripts/compute_band_statistics.py --ignore-index`` leaves nodata mask pixels out of the
  band statistics.
* Segmentation evaluations report Boundary IoU alongside mIoU, with the boundary width in pixels recorded in the result row.
* ``output.epoch_metrics`` streams per-epoch segmentation training records (training and validation loss, training and validation mIoU, learning rates, step counts, wall time, peak CUDA memory) to a JSON Lines file, tagged with the result row's ``config_hash``. Segmentation result rows gain ``best_val_epoch`` and ``best_val_miou``.
* ``segmentation.early_stopping_patience`` (default ``null``, off) stops segmentation training once validation mIoU has not improved for that many epochs; the best validation epoch is still restored. Hashes are unchanged when unset.
* UC Merced Land Use (``ucmerced``) via torchgeo: 2,100 RGB images, 21 classes, with published 1,260 / 420 / 420 splits and training-split normalization statistics. Download with ``geo-eval-harness download ucmerced``.
* NWPU-RESISC45 (``resisc45``) via torchgeo: 31,500 RGB scenes, 45 classes, on
  torchgeo's published 18,900 / 6,300 / 6,300 split.  Downloadable with
  ``geo-eval-harness download resisc45``.  It is the most-divergent benchmark in
  the GFM literature, so having it under a fixed protocol is the point.
* ``scripts/compute_band_statistics.py`` computes the per-band ``BandSpec``
  statistics a new dataset needs, from its train split only.
* AID (``aid``), rehosted at Hugging Face ``isaaccorley/aid`` (pinned commit,
  checksum-verified): 10,000 RGB scenes, 30 classes.  No official split
  exists upstream, so ``scripts/generate_aid_splits.py`` derives a
  deterministic, stratified 60/20/20 split per class (6,000 / 2,000 / 2,000).
  Downloadable with ``geo-eval-harness download aid``.  One of the three
  most-used benchmarks in the GFM literature.
* HOTOSM building segmentation (``hotosm_buildings``) from Hugging Face
  ``hotosm/vhr-building-segmentation``: 72,363 RGB tiles at about 0.3 m with
  OpenStreetMap building masks, on the official 57,890 / 7,237 / 7,236 split.
  ``geo-eval-harness download hotosm_buildings`` converts the upstream parquet
  into compact per-split files held in RAM.

Fixed
^^^^^

* A float16 overflow in the backbone no longer poisons segmentation heads. Training batches with non-finite backbone features are skipped before the head, so BatchNorm running statistics stay finite (previously one overflowing batch made every later evaluation predict a constant class); an epoch that skips more than 5% of its batches fails. Evaluation fails on non-finite features or logits instead of scoring a constant prediction.
* Segmentation evaluation follows :ref:`segmentation-model-selection`. Resume distinguishes this protocol from final-epoch results without changing classification identities or CSV columns; see :doc:`results-format` for rerun implications.
* UniverSat pins its pretrained weights to a fixed Hugging Face commit (``weights_revision``), as it already pinned the torch.hub code. Before, an update to ``g-astruc/UniverSat`` could change results silently. Set ``weights_revision: null`` to follow the default branch. Hashes and resume keys are unchanged.

0.5.0 (2026-08-10)
------------------

Added
^^^^^

* CoordBench coordinate-only evaluation with random and spatial-block
  cross-validation, KNN and ridge-linear probes, MIND and sine/cosine encoders,
  optional SatCLIP/GeoCLIP/Climplicit/SINR wrappers, checked-in results, and a
  mean-rank leaderboard.
* A dedicated ``geo-eval-harness flops`` pipeline for per-sample backbone, head,
  and probe compute, with a checked-in compute-cost table.
* OlmoEarth v1.2 Nano, Tiny, Small, and Base presets, including Landsat-as-S2
  routes for ``m-forestnet``.
* Representative segmentation sweeps and protocol-validation artifacts.
* GeoBench V1 subset downloads through ``download geobench_v1 --datasets ...``.

Changed
^^^^^^^

* Image-benchmark result rows now record ``num_classes`` and include it in the
  resume key. Rows from an older label schema cannot be silently reused.
* Linux x86_64 installs use the CUDA FAISS backend; other platforms use CPU
  FAISS. CoordBench KNN remains CPU by default unless ``coord.knn_device`` is
  overridden.
* SpaceNet2 and SpaceNet7 now use their native two-class building masks instead
  of GeoBench's unused third background class.
* TerraMind RGB routing, band selection, and pretrained normalization now match
  the selected modality.

Fixed
^^^^^

* The DPT segmentation head now follows the reference decoder's projection,
  fusion, and output path.
* PyTorch checkpoints are loaded with ``weights_only=True`` where applicable.
* OlmoEarth rejects ambiguous sensor routes that collapse multiple input bands
  onto the same pretrained channel.
* Resume-only datasets are skipped before data and model initialization.

Result migration
^^^^^^^^^^^^^^^^

The 128 checked-in SpaceNet2/7 rows produced under the obsolete three-class
task were removed. They remain available through the 0.4.0 tag but are not
comparable to current two-class runs. Other pre-0.5 rows do not carry
``num_classes`` and therefore cannot satisfy the new resume key.

`Full 0.4.0 to 0.5.0 diff <https://github.com/torchgeo/geo-eval-harness/compare/v0.4.0...v0.5.0>`__

Older releases
--------------

Earlier release notes remain available on
`GitHub Releases <https://github.com/torchgeo/geo-eval-harness/releases>`__.
