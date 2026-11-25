Changelog
=========

Unreleased
----------

* KNN uses exact brute-force search in PyTorch on CPU, CUDA, and MPS, so ``faissknn`` and FAISS are no longer installed, including by the ``id`` extra (``torchid>=0.6``). ``KNNClassifier`` drops its ``use_fp16`` argument, and the ``cosine`` and ``ip`` metrics now apply on CPU as well as GPU.

0.5.0 (2026-11-23)
------------------

Package and configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

* The distribution and CLI are renamed from ``torchgeo-bench`` to ``geo-eval-harness``, and the Python namespace from ``torchgeo_bench`` to ``geo_eval_harness``. Versions 0.1.0 to 0.4.0 are on PyPI as ``torchgeo-bench``.
* Commands take explicit flags and Pydantic-validated YAML, and model presets keep constructor options separate from input and evaluation settings. Model constructors raise ``TypeError`` for keyword arguments they do not name, so a misspelled ``model.kwargs`` entry fails instead of being ignored.
* ``--output-dir`` without ``--output`` writes beneath that directory even when the YAML sets ``output.file``.
* Image runs check ``intrinsic_dim`` settings when the configuration loads and reject unknown estimator names, ``max_samples`` below 2, and enabled estimators when ``torchid`` is not installed.
* Linux x86_64 installations use CUDA FAISS and other platforms use CPU FAISS. TerraTorch models need the separate source installation described in :doc:`installation`.
* The wheel and source distribution include the dataset geography records.

New models
^^^^^^^^^^

* Tessera V2 Nano, Small, Medium, and Large pixel students support Sentinel-2 snapshot classification, frozen segmentation, and LoRA. PASTIS supplies acquisition dates, and undated inputs use the configured fallback day of year (see :doc:`models` for accepted datasets).
* OlmoEarth v1, v1.1, v1.2, and v1.3 presets cover the supported encoder sizes and feature outputs.
* Clay v1 Base and Clay v1.5 Large presets provide mean-patch and CLS-token embeddings.
* THOR Tiny, Small, Base, and Large support classification and segmentation, with named band mapping and inputs prepared for each band's ground resolution.
* SMARTIES Base and Large support optical and SAR bands with sensor-specific projections and pretraining normalization.
* TerraFM supports dataset normalization and its model-native scaling with pretraining or dataset statistics.
* AnySat v1 Base supports classification, segmentation, and optional LoRA, with pinned code and weights and sensor-specific projectors.
* UniverSat supports segmentation and LoRA through ``universat_base_seg``, with pretrained weights pinned to a Hugging Face revision.
* EoS-FM combines 22 supervised specialists for classification and segmentation (see :doc:`models` for their pretraining overlap with benchmark datasets).
* ConvNeXt DINOv3 Tiny, Small, Base, and Large presets include segmentation layers and LoRA targets.

Model fixes
^^^^^^^^^^^

* timm ViT and DeiT models keep their pretrained final normalization when ``global_pool: avg`` averages the patch tokens. With ``global_pool: ""``, channels-last feature maps are averaged over their spatial axes, register tokens are left out of the patch average, and ``use_cls_token`` is rejected for backbones without a class token.
* CROMA reads the Sentinel-2 bands of datasets whose other sensors share band names with them, such as TreeSatAI. CROMA and Clay segmentation features are the outputs of the configured transformer blocks.
* ``torchgeo/resnet50_s2_all_moco`` uses all bands by default and feeds its 13 Sentinel-2 bands by name. It zero-fills B10 for L2A datasets and rejects band selections that lack any other band.
* Scale-MAE accepts ordered RGB bands under any recognized band name, such as Planet's ``r``, ``g``, ``b``. Its segmentation probes run at the constructed input size, and input sizes that are not multiples of 16 are rejected.
* SeCo ResNet model normalization converts RGB inputs to its 8-bit image scale, applies per-band quantiles, and standardizes with ImageNet statistics.
* torchgeo wrappers apply their model-native normalization out of place, so forwarding the same batch again gives the same features.
* OlmoEarth fills missing bands as upstream does for each version (v1 copies the nearest present band, and v1.1 and later set them to 0 after normalization). Its Sentinel-1 channels take only VV and VH backscatter bands, a sensor whose selected bands fill none of its channels is rejected, and SAR in an undeclared unit is normalized with dataset statistics.
* OlmoEarth and THOR encode resized tiles at their true ground size. OlmoEarth scales ``input_res`` by the resize factor, and THOR keeps each tile's native ground extent.
* TerraFM model normalization with pretraining statistics rejects Sentinel-1 bands that do not declare unit ``db`` or ``linear_power``.
* Prithvi-EO reads Landsat's NIR band as its narrow-NIR input, so ``m-forestnet`` and ``forestnet`` use all six pretrained bands.
* ImageStats computes its statistics from the raw inputs and records ``identity`` normalization whatever ``--normalization`` requests, so ``--resume`` matches its rows and the ranking explorer ranks them.
* OlmoEarth and SAM 3 weights download from pinned Hugging Face commits and are verified with SHA-256 before loading. Local SAM 3 and Clay v1.5 checkpoints are verified against the same digests.

Evaluation and results
^^^^^^^^^^^^^^^^^^^^^^

* CoordBench evaluates location encoders on random, spatial-block, and official holdouts with KNN and ridge-linear probes. It reads its benchmark tables and pretrained encoders from Hugging Face at pinned commits and checks the tables against packaged SHA-256 checksums. The ``mind`` preset reads ``taylor-geospatial/MIND``, and ``mind-small`` is removed because its weights are no longer published.
* The ``flops`` command measures backbone, decoder, and probe compute, and the ``profile`` command measures execution on a fixed dataset batch. FLOP counts include scaled dot-product attention on CPU and ``nn.MultiheadAttention`` and ``nn.TransformerEncoderLayer`` modules in evaluation mode.
* ``flops`` rows record ``measurement_variant``: ``catalog`` for a packaged preset without ``model.target`` or ``model.kwargs``, and ``custom`` otherwise.
* Segmentation supports frozen backbones, LoRA, and full fine-tuning, with configurable weight decay, head dropout, AMP, gradient checkpointing, and adapter block selection. ViT necks build feature pyramids for dense decoders, and UPerNet can use frozen AnyUp upsampling guided by RGB inputs.
* Segmentation evaluates the test split with the epoch that has the highest validation mIoU, and optional early stopping uses the same criterion. Patch-linear sweeps train one head per learning-rate and weight-decay pair and keep the best validation head and epoch.
* Segmentation reports Boundary IoU alongside mIoU and can write per-epoch training records to JSON Lines.
* Training skips batches with non-finite backbone features and fails an epoch when more than 5% of its batches are skipped. Evaluation rejects non-finite features and logits.
* Sample caches store preprocessed segmentation samples on disk, in host memory, or on the training device.
* Classification and frozen segmentation can standardize features with training-split statistics.
* Each dataset starts from ``runtime.seed``, so its result does not depend on the datasets evaluated before it in the same run.
* ``--matmul-precision`` (``runtime.matmul_precision``, default ``highest``) sets float32 matrix multiplication precision once per run.
* The ``config_hash`` of classification, profile, and intrinsic-dimension rows leaves out ``segmentation`` settings and covers ``runtime.matmul_precision`` only when it is not ``highest``, so segmentation-only changes and the default precision keep ``--resume`` matching stored rows.
* ``--resume`` matches OlmoEarth rows, which record ``model_native`` normalization whatever ``--normalization`` requests, so a relaunch skips finished OlmoEarth runs instead of recomputing them.
* Result rows record label counts and segmentation setup columns. ``load_results`` keeps a train-only linear probe separate from one refit on train plus validation.
* The ranking explorer separates adaptation, decoder, pyramid, input size, and training settings, ranks only the refit linear probes, and shows a model's recorded pooling under its name when it is not the patch mean. Native bands views allow incomplete dataset coverage.

Datasets
^^^^^^^^

* Infra-Bench CLS provides 13-class Sentinel-1/2 infrastructure classification with spatial splits.
* AID, NWPU-RESISC45, and UC Merced provide RGB scene classification with fixed splits and training-split statistics.
* PLANTEUR provides crop-type segmentation with monthly Sentinel-1/2 inputs and five leave-one-island-out folds.
* OpenEarthMap-SAR provides eight-class SAR segmentation, with optional ``vv`` and ``vh`` aliases for its single input band.
* GeoBench V1 supports collection and subset downloads with JSON metadata and verified archives. ``geo-eval-harness download`` accepts every registered dataset name, and a name that shares another dataset's files downloads that dataset.
* HOTOSM buildings and Open Cities use fixed training subsets of 5,000 tiles and 10,000 chips, respectively, and ``--partition full`` selects their complete training splits. A non-default ``--partition`` is an error for datasets without training partitions.
* CaFFe and ``bright-post-sar-rgb`` expose grayscale SAR as three identical channels for RGB models.
* PASTIS time series return ``(T, C, H, W)`` samples with every selected sensor on the channel axis.
* Multi-sensor GeoBench V2 datasets reject explicit band lists whose order the loader cannot return.
* Classification datasets declare class names in label-index order.

Dataset fixes
^^^^^^^^^^^^^

* BENV2 and So2Sat map their stored Sentinel-1 channels in VH, VV order.
* Dynamic EarthNet maps its stored Sentinel-2 channels in B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09 order, and band selections need at least one Planet band.
* TreeSatAI maps its stored aerial channels in NIR, green, blue, red order, so ``bands: rgb`` selects the stored red, green, and blue channels.
* TreeSatAI returns its Sentinel-1 ``vv`` and ``vh`` bands in dB, decoded from the stored ``uint16`` values, with no-data pixels set to the channel's mean over the sample's valid pixels.
* PASTIS declares its ascending and descending Sentinel-1 bands in dB, including the VV/VH ratio channels (VV minus VH in dB), so SMARTIES accepts one orbit's VV and VH under ``--normalization model``.
* ``forestnet`` declares the Landsat-8 bands of ``m-forestnet`` (``blue``, ``green``, ``red``, ``nir``, ``swir_1``, ``swir_2``), because GeoBench V2 repackages the same chips and split.
* Burn Scars has two classes, background and burn scar, and its no-data pixels are ignored in the loss and metrics.
* SpaceNet2 and SpaceNet7 use binary building masks.
