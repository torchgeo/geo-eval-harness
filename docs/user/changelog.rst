Changelog
=========

Unreleased
----------

Package and configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

* The distribution and CLI are named ``geo-eval-harness``; the Python namespace is ``geo_eval_harness``.
* Commands use explicit flags and Pydantic-validated YAML. Model presets separate constructor options from input and evaluation settings.
* Dataset geography records are included in the wheel and source distribution.
* Linux x86_64 installations use CUDA FAISS; other platforms use CPU FAISS.
* TerraTorch models require the separate source installation documented in :doc:`installation`.

Models
^^^^^^

* Tessera V2 Nano, Small, Medium, and Large pixel students support Sentinel-2 snapshot classification, frozen segmentation, and LoRA. PASTIS supplies acquisition dates; undated inputs use the configured fallback day of year. See :doc:`models` for accepted datasets and input requirements.
* Clay v1 Base and Clay v1.5 Large presets provide mean-patch and CLS-token embeddings.
* TerraFM supports dataset normalization and its model-native scaling with pretraining or dataset statistics.
* SMARTIES Base and Large presets support optical and SAR bands with sensor-specific projections and pretraining normalization.
* Scale-MAE segmentation probes run at the constructed input size, and input sizes that are not multiples of 16 are rejected.
* THOR Tiny, Small, Base, and Large presets support classification and segmentation with named band mapping and ground-resolution-aware input preparation.
* AnySat v1 Base supports classification and segmentation with pinned code and weights, sensor-specific projectors, and optional LoRA.
* UniverSat supports segmentation and LoRA through ``universat_base_seg`` and pins its pretrained weights to a Hugging Face revision.
* EoS-FM combines 22 supervised specialists for classification and segmentation. See :doc:`models` for pretraining overlap with benchmark datasets.
* ConvNeXt DINOv3 Tiny, Small, Base, and Large presets include segmentation layers and LoRA targets.
* OlmoEarth v1, v1.1, v1.2, and v1.3 presets expose their supported encoder sizes and feature outputs.

Evaluation and results
^^^^^^^^^^^^^^^^^^^^^^

* CoordBench evaluates location encoders with random, spatial-block, and official holdouts, using KNN and ridge-linear probes.
* The ``flops`` command measures backbone, decoder, and probe compute. The ``profile`` command measures execution on a fixed dataset batch.
* Classification and frozen segmentation can standardize features with training-split statistics.
* Segmentation supports frozen backbones, LoRA, and full fine-tuning, with configurable weight decay, head dropout, AMP, gradient checkpointing, and adapter block selection.
* ViT necks form feature pyramids for dense decoders. UPerNet can use frozen AnyUp upsampling guided by RGB inputs.
* Patch-linear sweeps train one head per learning-rate and weight-decay combination, then select the best validation head and epoch.
* Segmentation restores the epoch with highest validation mIoU for test evaluation. Optional early stopping uses the same validation criterion.
* Segmentation reports Boundary IoU alongside mIoU and can write per-epoch training records to JSON Lines.
* Non-finite backbone features are counted and skipped during training; an epoch fails if more than 5% of batches are skipped. Evaluation rejects non-finite features and logits.
* Sample caches store preprocessed segmentation samples on disk, in host memory, or on the training device.
* Result rows record label counts and segmentation setup columns. The ranking explorer separates adaptation, decoder, pyramid, input size, and training settings; Native bands views allow incomplete dataset coverage.
* The results explorer dictionary-encodes repeated data and enforces a 50 MB limit for generated pages and snapshots.

Datasets
^^^^^^^^

* Infra-Bench CLS provides 13-class Sentinel-1/2 infrastructure classification with spatial splits.
* GeoBench V1 supports collection and subset downloads with JSON metadata and verified archives.
* HOTOSM buildings and Open Cities use fixed training subsets of 5,000 tiles and 10,000 chips, respectively. ``--partition full`` selects their complete training splits.
* OpenEarthMap-SAR provides eight-class SAR segmentation and optional ``vv``/``vh`` aliases for its single input band.
* CaFFe and ``bright-post-sar-rgb`` expose grayscale SAR as three identical RGB channels for RGB model inputs.
* AID, NWPU-RESISC45, and UC Merced provide RGB scene classification with fixed splits and training-split statistics.
* PLANTEUR provides crop-type segmentation with monthly Sentinel-1/2 inputs and five leave-one-island-out folds.
* SpaceNet2 and SpaceNet7 use binary building masks. BENV2 maps its stored Sentinel-1 channels in VH, VV order.
* Classification datasets declare class names in label-index order.
* Band-statistics tools support excluding ignored segmentation pixels; geography extraction and coverage checks include the registered datasets.
