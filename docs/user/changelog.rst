Changelog
=========

Unreleased
----------

Added
^^^^^

* Opt-in validation early stopping for cached segmentation probes
  (``segmentation.early_stopping``), which keeps the best validation
  checkpoint, and an optional validation-selected learning-rate grid
  (``segmentation.learning_rates``). Defaults are unchanged, and existing
  results keep their resume fingerprint.
* UC Merced Land Use (``ucmerced``) via torchgeo: 2,100 RGB images, 21 classes, with published 1,260 / 420 / 420 splits and training-split normalization statistics. Download with ``torchgeo-bench download ucmerced``.
* NWPU-RESISC45 (``resisc45``) via torchgeo: 31,500 RGB scenes, 45 classes, on
  torchgeo's published 18,900 / 6,300 / 6,300 split.  Downloadable with
  ``torchgeo-bench download resisc45``.  It is the most-divergent benchmark in
  the GFM literature, so having it under a fixed protocol is the point.
* ``scripts/compute_band_statistics.py`` computes the per-band ``BandSpec``
  statistics a new dataset needs, from its train split only.
* AID (``aid``), rehosted at Hugging Face ``isaaccorley/aid`` (pinned commit,
  checksum-verified): 10,000 RGB scenes, 30 classes.  No official split
  exists upstream, so ``scripts/generate_aid_splits.py`` derives a
  deterministic, stratified 60/20/20 split per class (6,000 / 2,000 / 2,000).
  Downloadable with ``torchgeo-bench download aid``.  One of the three
  most-used benchmarks in the GFM literature.

0.5.0 (2026-08-10)
------------------

Added
^^^^^

* CoordBench coordinate-only evaluation with random and spatial-block
  cross-validation, KNN and ridge-linear probes, MIND and sine/cosine encoders,
  optional SatCLIP/GeoCLIP/Climplicit/SINR wrappers, checked-in results, and a
  mean-rank leaderboard.
* A dedicated ``torchgeo-bench flops`` pipeline for per-sample backbone, head,
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

`Full 0.4.0 to 0.5.0 diff <https://github.com/torchgeo/torchgeo-bench/compare/v0.4.0...v0.5.0>`__

Older releases
--------------

Earlier release notes remain available on
`GitHub Releases <https://github.com/torchgeo/torchgeo-bench/releases>`__.
