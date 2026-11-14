Datasets
========

``geo-eval-harness`` supports two generations of GeoBench datasets — V1 and
V2 — plus wrappers around torchgeo's standalone EuroSAT, NWPU-RESISC45, and UC Merced datasets, AID and Infra-Bench CLS (Hugging Face), HOTOSM building segmentation (Hugging Face), OpenEarthMap-SAR segmentation, the Open Cities AI Challenge building segmentation set, LoveDA land-cover segmentation (torchgeo), a LEVIR-CD post-image segmentation proxy, Sen1Floods11 hand-labelled flood-water segmentation (Sentinel-1 or Sentinel-2), BRIGHT post-event SAR building damage segmentation (Hugging Face), and OSCD post-image change segmentation (torchgeo). All datasets share the
:class:`~geo_eval_harness.datasets.BenchDataset` interface and are
registered by name so they can be selected without importing every loader.

Commands that use ``docs/examples/`` paths run from the repository root. With an installed package, save the example YAML locally and pass that path to ``--config``.

Filesystem layout
-----------------

All data lives under ``./data/`` relative to the current working
directory.  Paths are **fixed** — the runner does not honour environment
variables like ``GEOBENCH_ROOT``; if you keep data elsewhere, symlink
``data/`` to the real location.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Family
     - Default destination
     - Source
   * - ``geobench_v1``
     - ``data/classification_v1.0_wds/<name>/``
     - Hugging Face ``calebrob6/geobenchv1-webdataset``
   * - ``geobench_v2``
     - ``data/geobenchv2/<name>/``
     - Hugging Face ``aialliance/<name>``
   * - ``eurosat``
     - ``data/eurosat/``
     - torchgeo's ``EuroSAT`` and ``EuroSATSpatial`` downloaders
   * - ``resisc45``
     - ``data/resisc45/``
     - torchgeo's ``RESISC45`` downloader
   * - ``aid``
     - ``data/aid/``
     - Hugging Face ``isaaccorley/aid``, pinned commit + checksum-verified
   * - ``infrabench-cls``
     - ``data/infrabench_cls/``
     - Pinned Hugging Face tiles and GitHub spatial split, both checksum-verified
   * - ``open_cities``
     - ``data/open_cities/<city>/``
     - source.coop ``open-cities/ai-challenge`` tier 1, checksum-verified; masks built locally

   * - ``ucmerced``
     - ``data/ucmerced/``
     - torchgeo's ``UCMerced`` downloader
   * - ``hotosm_buildings``
     - ``data/hotosm_buildings/``
     - Hugging Face ``hotosm/vhr-building-segmentation``, pinned commit, converted to per-split parquet
   * - ``openearthmap-sar``
     - ``data/openearthmap-sar/``
     - Four checksum-verified DFC2025 Track I Zenodo archives
   * - ``loveda``
     - ``data/loveda/``
     - torchgeo's ``LoveDA`` downloader (labelled ``Train.zip`` and ``Val.zip`` only)
   * - ``levir-cd-post``
     - ``data/levir-cd-post/``
     - torchgeo's ``LEVIRCD`` downloader
   * - ``sen1floods11-s1``, ``sen1floods11-s2``
     - ``data/sen1floods11/``
     - Public GCS bucket ``sen1floods11`` (v1.1) over HTTPS, checksum-verified per file
   * - ``bright-post-sar``, ``bright-post-sar-rgb``
     - ``data/bright-post-sar/``
     - Hugging Face ``Kullervo/BRIGHT``, pinned commit, two SHA-256-verified ZIPs
   * - ``oscd-post``
     - ``data/oscd-post/``
     - torchgeo's ``OSCD`` downloader
   * - ``planteur``
     - ``data/planteur/``
     - Hugging Face ``allenai/planteur-benchmark``, pinned commit, monthly mosaics stacked into per-split arrays

Downloading
-----------

The :doc:`/api/cli` accepts one or more dataset names. Collection aliases are available for downloading a whole GeoBench family or a selected subset:

.. code-block:: console

   $ geo-eval-harness download m-eurosat burn_scars resisc45            # selected datasets
   $ geo-eval-harness download geobench_v1                              # all six V1 datasets
   $ geo-eval-harness download geobench_v1 --datasets m-eurosat         # V1 subset
   $ geo-eval-harness download geobench_v2                              # default V2 set
   $ geo-eval-harness download geobench_v2 --datasets benv2,burn_scars  # V2 subset
   $ geo-eval-harness download eurosat                                  # torchgeo EuroSAT
   $ geo-eval-harness download resisc45                                 # torchgeo RESISC45
   $ geo-eval-harness download aid                                      # isaaccorley/aid rehost
   $ geo-eval-harness download infrabench-cls
   $ geo-eval-harness download open_cities                              # source.coop, ~34 GB

   $ geo-eval-harness download ucmerced                                 # torchgeo UC Merced
   $ geo-eval-harness download hotosm_buildings                         # HOTOSM VHR buildings
   $ geo-eval-harness download openearthmap-sar                         # OpenEarthMap-SAR
   $ geo-eval-harness download loveda                                   # torchgeo LoveDA
   $ geo-eval-harness download levir-cd-post                            # LEVIR-CD post image
   $ geo-eval-harness download sen1floods11-s1                          # Sen1Floods11 (S1 + S2)
   $ geo-eval-harness download bright-post-sar                          # BRIGHT post-event SAR
   $ geo-eval-harness download oscd-post                                # torchgeo OSCD (post image)
   $ geo-eval-harness download planteur                                 # PLANTEUR crop types
   $ geo-eval-harness download geobench_v2 --output-dir /scratch/data   # custom root

Do not mix collection aliases and individual names in one invocation. ``--datasets`` applies only to a collection alias. ``--output-dir`` changes the download destination, not the runner's fixed ``data/`` location; link the downloaded root to ``data/`` before loading it from another working directory.

The default V2 download set is: ``benv2``, ``burn_scars``, ``caffe``,
``cloudsen12``, ``dynamic_earthnet``, ``flair2``, ``forestnet``,
``fotw``, ``kuro_siwo``, ``pastis``, ``so2sat``, ``spacenet2``,
``spacenet7``, ``treesatai``.

GeoBench V1 — classification
----------------------------

V1 datasets use the ``m-`` prefix on the command line.

Download V1 data before running a benchmark. ``geo-eval-harness download geobench_v1`` downloads all six classification datasets; ``--datasets`` selects a subset.

Downloads use the pickle-free ``calebrob6/geobenchv1-webdataset`` mirror at a pinned revision. They write JSON-metadata tar shards under ``data/classification_v1.0_wds/<name>/`` and verify each downloaded archive against the SHA-256 list bundled with the package. These are archive-level integrity checks, not per-sample metadata approvals.

V1 metadata format
^^^^^^^^^^^^^^^^^^

V1 readers use JSON metadata only. Tar shards pair ``<sample_id>.bands.npz`` with ``<sample_id>.meta.json``; image arrays are loaded with ``allow_pickle=False``. Custom HDF5 samples under ``data/classification_v1.0/<name>/`` must store a UTF-8 JSON string in the ``metadata_json`` attribute. The JSON object must contain a numeric ``label`` (or a numeric list for multilabel data) and a non-empty ``bands_order`` list of source-band names.

.. code-block:: json

   {
     "label": 2,
     "bands_order": ["04 - Red", "03 - Green", "02 - Blue"],
     "04 - Red": {
       "transform": [10, 0, 500000, 0, -10, 5200000],
       "crs": "EPSG:32631"
     }
   }

Per-band ``transform`` and ``crs`` entries are optional. Geographic extraction accepts six affine coefficients ``[a, b, c, d, e, f]`` (or the nine-element affine matrix) and a CRS string; use ``null`` when georeferencing is unavailable. Python objects such as affine/CRS classes and NumPy label arrays must be exported as JSON numbers, lists, and strings by the data producer.

If an archive checksum fails, remove the named corrupt archive and retry the download. Custom datasets must supply JSON metadata.

Geography extraction reads the JSON metadata directly from the default shards or custom HDF5 files. ``scripts/repack_geobench_v1.py`` repacks custom JSON-metadata HDF5 files into JSON-based shards.

Supported V1 datasets
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 12 22 32

   * - CLI name
     - #cls
     - bands
     - multilabel
     - sensor
     - Class
   * - ``m-bigearthnet``
     - 43
     - 12
     - **yes**
     - Sentinel-2
     - :class:`~geo_eval_harness.datasets.MBigEarthNet`
   * - ``m-brick-kiln``
     - 2
     - 13
     - no
     - Sentinel-2
     - :class:`~geo_eval_harness.datasets.MBrickKiln`
   * - ``m-eurosat``
     - 10
     - 13
     - no
     - Sentinel-2
     - :class:`~geo_eval_harness.datasets.MEurosat`
   * - ``m-forestnet``
     - 12
     - 6
     - no
     - Landsat
     - :class:`~geo_eval_harness.datasets.MForestnet`
   * - ``m-pv4ger``
     - 2
     - 3
     - no
     - Aerial RGB
     - :class:`~geo_eval_harness.datasets.MPv4ger`
   * - ``m-so2sat``
     - 17
     - 18
     - no
     - Sentinel-1 + S2
     - :class:`~geo_eval_harness.datasets.MSo2Sat`

Multi-label datasets (``m-bigearthnet``) report the ``micro_mAP`` metric
instead of accuracy.

GeoBench V2 — classification
----------------------------

================== ====== ===== ============ ====================================== =====================================
CLI name           #cls   bands multilabel   sensor                                 Class
================== ====== ===== ============ ====================================== =====================================
``benv2``          19     14    **yes**      Sentinel-1 + Sentinel-2 (multi-modal)  :class:`~geo_eval_harness.datasets.BENV2`
``forestnet``      12     6     no           Sentinel-2                             :class:`~geo_eval_harness.datasets.Forestnet`
``so2sat``         17     12    no           Sentinel-1 + Sentinel-2                :class:`~geo_eval_harness.datasets.So2Sat`
``treesatai``      13     19    **yes**      Aerial + S2 + S1 (multi-modal)         :class:`~geo_eval_harness.datasets.TreeSatAI`
================== ====== ===== ============ ====================================== =====================================

V2 datasets are stored as a single ``.tortilla`` file each, hosted under
``aialliance/<name>`` on the Hugging Face Hub. Download the selected datasets
before running a benchmark, for example with
``geo-eval-harness download geobench_v2 --datasets benv2,burn_scars``.

.. note::

   V1 and V2 share several CLI names but the underlying data are
   *different* datasets — different sensors, splits, label spaces, or
   normalisation conventions:

   * ``m-bigearthnet`` (V1) — original BigEarthNet, 43-class
     **multilabel** scene tags, S2 Level-2A DN.
     ``benv2`` (V2) — BigEarthNet v2.0 with 19-class multilabel and
     stacked S1 + S2.
   * ``m-so2sat`` (V1) — 18-band stack including LCZ context.
     ``so2sat`` (V2) — 10 S2 + 2 S1 bands stored at reflectance scale.
   * ``m-forestnet`` (V1) — Landsat-8 6-band uint8.
     ``forestnet`` (V2) — Sentinel-2 6-band uint8 with a different split
     than V1 despite the same channel count.

   The leaderboard prefixes panel titles with ``GeoBench V1`` /
   ``GeoBench V2`` so V1 and V2 results are never compared on the same
   axis.

GeoBench V2 — segmentation
--------------------------

==================== ====== ===== ============================================ ==============================================================
CLI name             #cls   bands notes                                        Class
==================== ====== ===== ============================================ ==============================================================
``burn_scars``       3      6                                                  :class:`~geo_eval_harness.datasets.BurnScars`
``caffe``            4      3     grayscale SAR repeated as red/green/blue     :class:`~geo_eval_harness.datasets.CaFFe`
``cloudsen12``       4      12                                                 :class:`~geo_eval_harness.datasets.CloudSEN12`
``dynamic_earthnet`` 7      16                                                 :class:`~geo_eval_harness.datasets.DynamicEarthNet`
``flair2``           13     5     aerial + Sentinel-2                          :class:`~geo_eval_harness.datasets.FLAIR2`
``fotw``             3      4     Fields of the World; unknown (3) ignored     :class:`~geo_eval_harness.datasets.FieldsOfTheWorld`
``kuro_siwo``        4      3     SAR ``vv`` / ``vh`` + DEM (no RGB triplet)   :class:`~geo_eval_harness.datasets.KuroSiwo`
``pastis``           20     16    Sentinel-2 + Sentinel-1 (multi-modal)        :class:`~geo_eval_harness.datasets.PASTIS`
``spacenet2``        2      9     WorldView 8-band + pan                       :class:`~geo_eval_harness.datasets.SpaceNet2`
``spacenet7``        2      3                                                  :class:`~geo_eval_harness.datasets.SpaceNet7`
==================== ====== ===== ============================================ ==============================================================

Other
-----

======================= ============================================================================
CLI name                Class
======================= ============================================================================
``eurosat``             :class:`~geo_eval_harness.datasets.EuroSAT`  (torchgeo wrapper, random split)
``eurosat-spatial``     :class:`~geo_eval_harness.datasets.EuroSATSpatial`  (longitude-based split)
``resisc45``            :class:`~geo_eval_harness.datasets.RESISC45`  (45-class aerial scenes, RGB)
``aid``                 :class:`~geo_eval_harness.datasets.AID`  (30-class aerial scenes, RGB)
``infrabench-cls``       :class:`~geo_eval_harness.datasets.InfraBenchCLS` (13 infrastructure classes, S2 + S1)

``ucmerced``            :class:`~geo_eval_harness.datasets.UCMerced`  (21-class aerial land use, RGB)
``hotosm_buildings``    :class:`~geo_eval_harness.datasets.HOTOSMBuildings`  (building segmentation, RGB; trains on a fixed 5k subset, see below)
``openearthmap-sar``    :class:`~geo_eval_harness.datasets.OpenEarthMapSAR`  (land-cover segmentation, SAR)
``loveda``              :class:`~geo_eval_harness.datasets.LoveDA`  (land-cover segmentation, RGB)
``levir-cd-post``       :class:`~geo_eval_harness.datasets.LEVIRCDPost`  (post-image change proxy, RGB)
``sen1floods11-s1``     :class:`~geo_eval_harness.datasets.Sen1Floods11S1`  (flood-water segmentation, S1 VV/VH dB)
``sen1floods11-s2``     :class:`~geo_eval_harness.datasets.Sen1Floods11S2`  (flood-water segmentation, S2 L1C)
``bright-post-sar``     :class:`~geo_eval_harness.datasets.BRIGHTPostSAR`  (building damage segmentation, SAR)
``bright-post-sar-rgb`` :class:`~geo_eval_harness.datasets.BRIGHTPostSARRGB`  (building damage segmentation, repeated SAR)
``oscd-post``           :class:`~geo_eval_harness.datasets.OSCDPost`  (urban change from the post image, S2 L1C)
``planteur``            :class:`~geo_eval_harness.datasets.PLANTEUR`  (crop-type segmentation, S2 + S1, 12 months)
``planteur_loio_*``     ``PLANTEURLOIOGuadeloupe`` … ``PLANTEURLOIOMayotte``  (leave-one-island-out folds)
======================= ============================================================================

``resisc45`` is 31,500 RGB scenes at 256x256 across 45 classes, on torchgeo's
published 60/20/20 split.  It is the most-divergent benchmark in the GFM
literature — the same Scale-MAE checkpoint is reported at 33.0 and 89.6
linear-probe accuracy by different papers under the same nominal protocol —
which is why it is worth running under a fixed harness.  The imagery carries
no geolocation, so it appears on the coverage map as an explicit gap rather
than being silently omitted.

``aid`` contains 10,000 RGB scenes at 600x600 across 30 classes (Xia et al., 2017). With no official split, ``scripts/generate_aid_splits.py`` generates a deterministic, stratified 60/20/20 split. Downloads use the pinned, checksum-verified ``isaaccorley/aid`` rehost. Upstream specifies no license or image geolocation. Band subsets and ordering are applied before user transforms.

``infrabench-cls`` contains 18,756 OpenStreetMap infrastructure tiles in 13 classes, with seven Sentinel-2 bands and Sentinel-1 VV/VH. The spatial split assigns 13,087 / 2,856 / 2,813 tiles by asset ID; duplicated border assets stay within one split. Images are resized bilinearly to 60x60 pixels. Sentinel-2 values use clipped 0--255 scaling, including the upstream L2A processing offset; Sentinel-1 backscatter is clipped to -30--10 dB. Labels come from OSM tags. The dataset is distributed under ODbL 1.0; see `Infra-Bench CLS <https://doi.org/10.5281/zenodo.22118891>`_.

``ucmerced`` contains 2,100 RGB aerial images across 21 land use classes, resized to 256x256 by torchgeo. Its published split files contain 1,260 training, 420 validation, and 420 test images. Normalization statistics use the training split only. The source resolution is 1 ft (0.3048 m); the shared ``aerial`` sensor tag uses an approximate 1 m GSD for resolution-aware models. Images have no georeferencing, which is recorded in the coverage map.

.. code-block:: console

   $ geo-eval-harness run --model imagestats --dataset ucmerced --device cpu

``hotosm_buildings`` provides binary building segmentation on RGB imagery. See `HOTOSM buildings — segmentation`_ for its training subset, split definitions, download, and memory requirements.

``openearthmap-sar`` uses the official IEEE GRSS Data Fusion Contest 2025 Track I splits: **4,333 training, 210 validation, and 490 test** image/label pairs. Images are single-band ``uint8`` SAR GeoTIFFs at **1024 x 1024** pixels, with published GSD of **0.15-0.5 m**.

The eight classes, in model output order, are bareland, grass/rangeland, pavement/developed space, road, tree, water, cropland/agricultural land, and building. Native labels ``1..8`` map to model classes ``0..7``; native ``0`` maps to ``255`` and is ignored in both training loss and mIoU. This deliberately differs from the official baseline, which trains a ninth background output but excludes it from foreground mIoU.

Only the single ``sar`` band is installed and exposed. Training RGB is intentionally excluded because validation and test inputs are SAR-only. The default ``rgb`` selector resolves to ``sar``, following the grayscale dataset convention. For models that accept only Sentinel-1 polarisations, ``--bands vv`` or ``--bands vv,vh`` opts in to feeding the same ``sar`` channel under those names, with the same statistics; the source polarisation is not declared, so this is an adapter for comparison, not a physical match. The adapter emits raw float32 values in ``0..255`` units, without scaling, augmentation, or cropping. Dataset normalization uses statistics measured over all 4,333 native-resolution training SAR images; the runner owns resizing and the model owns normalization. The official baseline instead scales bytes to ``[0, 1]`` and uses random 512-pixel training crops.

.. code-block:: text

   data/openearthmap-sar/
     train/
       sar_images/TrainArea_001.tif ... TrainArea_4333.tif
       labels/TrainArea_001.tif ... TrainArea_4333.tif
     val/
       sar_images/ValArea_001.tif ... ValArea_210.tif
       labels/ValArea_001.tif ... ValArea_210.tif
     test/
       sar_images/TestArea_001.tif ... TestArea_490.tif
       labels/TestArea_001.tif ... TestArea_490.tif

Membership is fixed, paired by basename, and ordered numerically. Missing or unexpected TIFFs, invalid raster dimensions/dtypes, and labels outside ``0..8`` are errors. Custom partitions are unsupported.

.. code-block:: console

   $ geo-eval-harness download openearthmap-sar
   $ geo-eval-harness run --config docs/examples/openearthmap-sar-run.yaml --dry-run
   $ geo-eval-harness run --config docs/examples/openearthmap-sar-run.yaml

The :download:`example config </examples/openearthmap-sar-run.yaml>` uses a randomly initialized ResNet-18 with a linear segmentation head as a no-weight-download smoke configuration, not a published accuracy baseline.

The downloader retrieves about **9.29 GB** across four immutable archives: ``dfc25_track1_trainval.zip`` from `record 14622048 <https://zenodo.org/records/14622048>`__, ``dfc25_track1_test.zip`` from `record 14948851 <https://zenodo.org/records/14948851>`__, and ``dfc25_track1_val_labels.zip`` / ``dfc25_track1_test_labels.zip`` from `record 14950559 <https://zenodo.org/records/14950559>`__. It verifies published MD5 digests and byte counts, extracts only expected SAR/label TIFFs, and retains the ZIPs. Re-running the download revalidates the cached archives and repairs extracted files without repeating the network transfer.

Geography is read from SAR GeoTIFF headers without requiring label files. The release provides usable locations for **210 of 5,033 samples**, all in the validation split; training and test SAR files have no usable georeferencing. Images without a CRS or with an identity transform are omitted from location counts, not assigned guessed coordinates. Raster read errors and invalid projected coordinates are errors.

Sources and attribution: `OpenEarthMap-SAR <https://github.com/cliffbb/OpenEarthMap-SAR>`__, the `official DFC2025 baseline <https://github.com/cliffbb/DFC2025-OEM-SAR-Baseline>`__, and the `dataset paper <https://arxiv.org/abs/2501.10891v2>`__. Zenodo identifies the Umbra Open Data Program SAR imagery as CC BY 4.0. Optical-source and label licensing varies by source; consult the release and source-specific terms rather than treating the whole installed benchmark as uniformly CC BY 4.0.

``levir-cd-post`` is a **single-image proxy task, not change detection**: each sample is a tile of the LEVIR-CD *post* image (``B``), segmented against the building-change mask (the Fields of the World ``image_b`` precedent). The pre image is never loaded, so scores are **not comparable to published LEVIR-CD change-detection F1**. The ``-post`` suffix keeps ``levir-cd`` free for a future bi-temporal variant.

LEVIR-CD has the official **445 / 64 / 128** train/val/test images (Google Earth RGB, **0.5 m**, **1024 x 1024**). Each image is cut into 16 non-overlapping **256 x 256** tiles at native resolution (**7,120 / 1,024 / 2,048** tiles), ordered by image, row and column, before the runner resizes them. Classes are 0 = no change and 1 = change; as in torchgeo, every non-zero mask value is change, so there are no ignore pixels. Change covers only **4.6 / 4.2 / 5.1 %** of train/val/test pixels, so read the reported mIoU against the all-no-change baseline of about 0.475 on test. Dataset normalization uses statistics from the training tiles of the ``aerial`` red/green/blue bands (raw ``0..255``).

.. code-block:: text

   data/levir-cd-post/
     A/       pre images (downloaded but unused)
     B/       train_1.png ... train_445.png, val_1.png ... val_64.png, test_1.png ... test_128.png
     label/   change masks with the same names
     train.zip  val.zip  test.zip

.. code-block:: console

   $ geo-eval-harness download levir-cd-post
   $ geo-eval-harness run --config docs/examples/levir-cd-post-run.yaml --dry-run
   $ geo-eval-harness run --config docs/examples/levir-cd-post-run.yaml

The download uses torchgeo's ``LEVIRCD`` (2.5 GB of MD5-verified archives from a pinned Hugging Face commit, about 4.7 GB on disk after extraction) and then checks the exact split membership. The :download:`example config </examples/levir-cd-post-run.yaml>` is a randomly initialized ResNet-18 smoke configuration, not an accuracy baseline. The images are not georeferenced.

Sources and attribution: `LEVIR-CD <https://justchenhao.github.io/LEVIR/>`__ and the `dataset paper <https://doi.org/10.3390/rs12101662>`__ (Chen & Shi, *Remote Sensing* 12(10):1662, 2020). The dataset is distributed for academic research only; imagery is from Google Earth and subject to Google's terms.

``sen1floods11-s1`` and ``sen1floods11-s2`` are the **446 hand-labelled** chips of Sen1Floods11 v1.1 (Bonafilia et al., CVPRW 2020), seen through two sensors: Sentinel-1 GRD backscatter (``vv``, ``vh``; float32 dB, ``unit="db"``) or all 13 Sentinel-2 L1C bands (``b01``..``b12``, ``b8a``; int16 top-of-atmosphere reflectance x 10\ :sup:`4`, so the unit is detected as S2 DN). Both ids share the official ``flood_handlabeled`` splits, **252 training, 89 validation (``valid``), and 90 test** chips, and exactly the same masks, so S1 and S2 results are directly comparable. The 15-chip Bolivia hold-out and the 4,385 weakly labelled chips are not used. Chips are **512 x 512** pixels in EPSG:4326 at about 10 m and are not tiled; the runner resizes them.

Classes are 0 (not water) and 1 (water). Native ``-1`` (no data, mostly clouds) maps to ``255``. Pixels where Sentinel-1 is not finite or all Sentinel-2 bands are zero (scene-edge fill) also map to ``255`` for **both** ids: 26,037 / 20,571 / 73,408 otherwise-labelled train/val/test pixels. Non-finite Sentinel-1 values are replaced with the band's training mean, so the S1 dataset only emits finite values. Normalization statistics are measured on the training split over valid (non-``255``) pixels only.

Label quality: about 13% of pixels are no data and about 10% of the valid pixels are water (9.5 / 11.0 / 12.5% in train/val/test). One training, three validation, and one test chip are entirely no data; they are kept because they are part of the official membership and contribute nothing to the scores. An independent visual inspection (Arshad, MSc thesis, University of Twente, 2026) found the hand labels sufficiently accurate; its ~35% error rate concerns the weak labels, which are excluded here. Otsu thresholding of S1 agrees with the hand labels at only about 0.5 IoU, so S1-only models have a lower ceiling than S2. The splits are random chips from the same 10 flood events, so they are spatially correlated and overstate generalisation. Also report the water-class IoU, not only mIoU. The protocol (train, val, and test on hand labels, as in PANGAEA and TerraTorch) differs from the original paper, which trains on weak labels and tests on all 446 hand-labelled chips, so the numbers are not comparable.

.. code-block:: text

   data/sen1floods11/
     splits/flood_handlabeled/
       flood_train_data.csv  flood_valid_data.csv  flood_test_data.csv
     data/flood_events/HandLabeled/
       S1Hand/<event>_<chip>_S1Hand.tif        (431 files)
       S2Hand/<event>_<chip>_S2Hand.tif        (431 files)
       LabelHand/<event>_<chip>_LabelHand.tif  (431 files)

Membership follows the official CSVs; missing files and CSVs with the wrong chip count are errors. Custom partitions are unsupported.

.. code-block:: console

   $ geo-eval-harness download sen1floods11-s1
   $ geo-eval-harness run --config docs/examples/sen1floods11-s1-run.yaml --dry-run
   $ geo-eval-harness run --config docs/examples/sen1floods11-s1-run.yaml

Either id downloads the shared data: the 1,293 TIFFs referenced by the three split CSVs plus the CSVs, about **1.7 GB**, from ``https://storage.googleapis.com/sen1floods11/v1.1/`` over plain HTTPS (no ``gsutil``). Upstream publishes no checksums, so each file is verified against the packaged ``checksums/sen1floods11.sha256`` manifest. Re-running the download re-hashes local files and fetches only missing or corrupt ones. The :download:`example config </examples/sen1floods11-s1-run.yaml>` uses a randomly initialized ResNet-18 with a linear segmentation head as a no-weight-download smoke configuration, not a published accuracy baseline; :download:`sen1floods11-s2-run.yaml </examples/sen1floods11-s2-run.yaml>` is the S2 equivalent.

For OlmoEarth on ``sen1floods11-s1``, use ``olmoearth_nano`` (or another preset that leaves ``sar_log_scale`` unset), not ``olmoearth_nano_sar_db``: the values are already in dB, and that preset converts linear power to dB.

Sources and attribution: `Sen1Floods11 <https://github.com/cloudtostreet/Sen1Floods11>`__, generated by Cloud to Street. Cite Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020), *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*, CVPR Workshops. The imagery contains modified Copernicus Sentinel data. Upstream specifies no license for the dataset; check the upstream terms before redistribution.

``bright-post-sar`` is four-class building damage segmentation from very-high-resolution **post-event SAR only**, taken from BRIGHT (Chen et al., 2025), the dataset of IEEE GRSS DFC 2025 Track II. It covers 14 disasters (earthquakes, floods, hurricanes, wildfires, volcanoes, explosions, and a conflict) with Capella and Umbra X-band SAR at **0.3-0.6 m** GSD. The pre-event optical images are not used or installed.

It uses the authors' ``standard_ML`` split, which the BRIGHT maintainers recommend now that the contest is over: **2,918 training, 450 validation, and 878 test** tiles, with every event present in every split (event-stratified, not event-held-out). The memberships are packaged with the harness, copied and sorted from ``bda_benchmark/dataset/splitname/standard_ML`` of the `BRIGHT repository <https://github.com/ChenHongruixuan/BRIGHT>`__ at commit ``59269142``. All three splits are labelled. The test split contains torchgeo's ``BRIGHTDFC2025`` ``val`` (contest holdout) split; torchgeo's ``test`` split (contest validation, whose labels are now public) is part of this validation split.

Classes are 0 background, 1 intact, 2 damaged, and 3 destroyed, kept as native labels; no pixel is ignored. Training pixels are 86.0% background, 11.7% intact, 1.5% damaged, and 0.9% destroyed; validation and test are similar.

Images are single-band ``uint8`` amplitude (display-scaled, not calibrated backscatter) at **1024 x 1024** pixels, exposed as one ``sar`` band in raw float32 ``0..255`` units; the default ``rgb`` selector resolves to ``sar``. The polarisation is not recorded in the files, so the band has no ``vv``/``hh`` alias and no ``unit``. Zero is the swath fill value (12.5% of training pixels). It is kept as input, and its labels are kept, because labelled buildings overlap it. Dataset normalization uses statistics over all 2,918 native-resolution training images, zeros included. The runner owns resizing; tiles are not cropped.

``bright-post-sar-rgb`` serves the same SAR tile as three identical ``red``/``green``/``blue`` channels, using sensor ``aerial``, no wavelengths, and the SAR statistics for each channel (mean 56.8231, std 45.8553). Both the default ``rgb`` selector and ``all`` select all three; explicit band names select fewer copies. This pseudo-RGB representation lets models use their RGB input paths; it is not optical imagery. Tiles, masks, split memberships, geography, and the installation below are identical to ``bright-post-sar``. Download once with ``geo-eval-harness download bright-post-sar``, then select ``--dataset bright-post-sar-rgb`` for evaluation. The two dataset ids distinguish result rows and run hashes.

.. code-block:: text

   data/bright-post-sar/
     post-event.zip  target.zip
     post-event/<event>_<id>_post_disaster.tif     (4,246 files)
     target/<event>_<id>_building_damage.tif       (4,246 files)

Missing files, invalid raster dimensions or dtypes, and labels outside ``0..3`` are errors. Custom partitions are unsupported.

.. code-block:: console

   $ geo-eval-harness download bright-post-sar
   $ geo-eval-harness run --config docs/examples/bright-post-sar-run.yaml --dry-run
   $ geo-eval-harness run --config docs/examples/bright-post-sar-run.yaml

The :download:`example config </examples/bright-post-sar-run.yaml>` uses a randomly initialized ResNet-18 with a linear segmentation head as a smoke configuration that downloads no weights; it is not an accuracy baseline.

The downloader fetches ``post-event.zip`` (3.29 GB) and ``target.zip`` (48 MB) from `Kullervo/BRIGHT <https://huggingface.co/datasets/Kullervo/BRIGHT>`__ at commit ``46f202d9``, verifies their SHA-256 digests, extracts the TIFFs, and keeps the ZIPs. The installation needs about **12 GB** because the extracted TIFFs are uncompressed. Rerunning the download reverifies the archives and repairs missing or truncated files without downloading again.

Geography is read from the post-event GeoTIFF headers; all 4,246 tiles are georeferenced.

Sources and attribution: the `BRIGHT paper <https://essd.copernicus.org/articles/17/6217/2025/>`__ (Chen et al., Earth System Science Data 17, 2025) and the `BRIGHT repository <https://github.com/ChenHongruixuan/BRIGHT>`__. The SAR images come from the Capella Open Data Gallery and the Umbra Open Data Program under CC BY 4.0. The labels carry the licence of each event's optical imagery, which is mostly Maxar Open Data under CC BY-NC 4.0. Check these terms before any non-research use.

``oscd-post`` poses the Onera Satellite Change Detection dataset (OSCD; Caye Daudt et al., IGARSS 2018) as **single-image binary segmentation of the post-change image**. Each of the 24 cities has a Sentinel-2 **Level-1C** image pair and a pixel-level urban change map. Only the later image is used. There is no bi-temporal input, so a model must recognise new construction from appearance alone, and scores are not comparable with published change-detection results on OSCD.

* **Splits (by city).** ``test`` is the 10 official test cities. ``val`` is three of the 14 official training cities, ``cupertino``, ``rennes``, and ``saclay_e``, drawn with ``sorted(random.Random(0).sample(sorted(official_train), 3))``. ``train`` is the remaining 11.
* **Tiles.** Cities range from 241 to 1180 px a side, all at 10 m, so they are tiled rather than resized. Each city is cut row-major into 256 x 256 tiles (2.56 km), giving **119 / 31 / 69** train/val/test tiles. The last tile in each row and column is shifted inward to end at the image border, so it shows real imagery instead of padding. The part of it that overlaps the previous tile is masked ``255``, so every pixel is scored exactly once; this is about a third of all tile pixels. Only norcia, which is 241 px tall, is zero-padded, and its padded pixels are also ``255``.
* **Labels.** These come from ``<city>-cm.tif``, where native ``1`` = no change and ``2`` = change. They map to classes ``0`` (no change) and ``1`` (change). The ``cm.png`` previews are not used because the aguasclaras PNG is anti-aliased. Change covers 2.4 % / 2.0 % / 5.2 % of the labelled train/val/test pixels.
* **Bands.** All 13 bands come from the authors' ``imgs_2_rect`` stacks, which are resampled to 10 m: ``b01`` ... ``b12`` and ``b8a``. The cirrus band ``b10`` is included because the data is L1C; models that do not use cirrus drop it through band mapping. ``rgb`` is ``b04, b03, b02``.
* **Values and normalization.** Values are raw L1C digital numbers (reflectance x 10\ :sup:`4`) as float32. A few edge pixels of the 20 m and 60 m bands are 0 from the resampling and are kept. Dataset normalization uses statistics over every pixel of the 11 training cities.
* **L1C models.** ``BandSpec`` has no processing-level field, so select an L1C model variant explicitly, for example ``terrafm/base_s2_l1c``.

The dataset is tiny and the change class is rare, so expect noisy scores and report several seeds.

The data is torchgeo's ``OSCD`` layout under ``data/oscd-post/``. Only ``<city>/imgs_2_rect/*.tif`` and the ``<city>-cm.tif`` change maps are read. Missing files and tile counts that differ from the split sizes are errors. Custom partitions are unsupported.

.. code-block:: console

   $ geo-eval-harness download oscd-post
   $ geo-eval-harness run --config docs/examples/oscd-post-run.yaml --dry-run
   $ geo-eval-harness run --config docs/examples/oscd-post-run.yaml

The :download:`example config </examples/oscd-post-run.yaml>` uses a randomly initialized ResNet-18 with a linear head as a no-weight-download smoke test. The download is about 490 MB. torchgeo verifies the SHA-256 digests and extracts the archives, which leaves 1.2 GB on disk.

Sources and attribution: the `OSCD project page <https://rcdaudt.github.io/oscd/>`__ and the `paper <https://doi.org/10.1109/IGARSS.2018.8518015>`__. The imagery contains modified Copernicus Sentinel data (2015-2018). The change labels are released under **CC BY-NC-SA** (non-commercial); contact the authors for commercial use.

HOTOSM buildings — segmentation
-------------------------------

``hotosm_buildings`` is building footprint segmentation on 256 x 256 OpenAerialMap RGB tiles (about 0.3 m) from 93 HOT Tasking Manager projects, with OpenStreetMap labels. Validation (**7,237** tiles) and test (**7,236**) are the official splits.

.. important::

   **The benchmark trains on a fixed 5,000-tile subset of the official train split.** This
   ``default`` partition is the version reported on the leaderboard. HOTOSM results on the
   full train split are not comparable with leaderboard rows.

======================== ============ ===============================================================
``--partition``          train tiles  use
======================== ============ ===============================================================
``default``              5,000        benchmark and leaderboard version
``full``                 57,890       the whole official train split
======================== ============ ===============================================================

The subset is packaged as ``hotosm_buildings_train_5k.csv.gz``, which lists each tile's ``project_id``, ``tile_id``, and row in the converted ``train.parquet``. Loading checks that every listed row holds the listed tile, so a train file converted differently fails instead of silently training on other tiles. The tiles were drawn per project in proportion to its size, after excluding train tiles that also appear in val or test, so the subset keeps the project and country mix of the full split (76% Myanmar).

In a training-budget study (13 models, frozen and LoRA, UPerNet, 10 epochs, budgets 1k to all 57,880 eligible tiles, 3 subsets x 3 training seeds), the model ranking at 5k tiles matched full-data training with Kendall tau-b 0.97 (frozen) and 0.88 (LoRA). Rankings from two training seeds on the full data agree only at 0.87 to 0.94, so 5k is within seed noise, at about 1/12 of the training compute. The packaged subset is the study's subset seed 1. Normalization statistics remain those of the full train split.

``geo-eval-harness download hotosm_buildings`` fetches the pinned parquet (5.9 GB), keeps the image bytes, re-encodes masks as PNGs, and removes the upstream shards, leaving about 1.1 GB. Labels are 0 for background, 1 for building, and 255 for transparent nodata, which loss and metrics ignore. The complete encoded splits use about 1.5 GB of RAM. With spawn or forkserver workers, including on macOS and Windows, use ``--workers 0`` to avoid copying that data. Set ``segmentation: {cache_features: false}`` when the feature cache would exceed available RAM.

The full training split contains three Myanmar projects with 76% of its tiles; Eswatini appears only in validation and test. Five tile coordinates occur in both full train and validation, and overlapping projects can repeat tiles within a split. The default subset excludes train tiles shared with validation or test.

Imagery is CC-BY 4.0 (OpenAerialMap); labels are ODbL 1.0 (OpenStreetMap).

Open Cities AI Challenge — segmentation
---------------------------------------

==================== ====== ===== ============================================ ==============================================================
CLI name             #cls   bands notes                                        Class
==================== ====== ===== ============================================ ==============================================================
``open_cities``      2      3     drone RGB, native 0.02-0.20 m GSD            :class:`~geo_eval_harness.datasets.OpenCities`
==================== ====== ===== ============================================ ==============================================================

``open_cities`` is building footprint segmentation on drone imagery of seven African cities (Accra, Dar es Salaam, Kampala, Monrovia, Niamey, Pointe-Noire, and Zanzibar), from the tier-1 scenes of the Open Cities AI Challenge (GFDRR Labs, 2020). The upstream release contains 31 large georeferenced scenes with OpenStreetMap footprints and a test set without labels, so the benchmark defines its own chips and splits. Samples are 512x512 chips cut on each scene's native pixel grid. Since the native ground sampling distance ranges from 0.02 m to 0.20 m, a chip covers between 10 m and 100 m on the ground, and any resizing is left to the model's transforms.

The chips and their train/val/test assignment are fixed in the packaged ``open_cities_index.csv.gz``, which ``scripts/generate_open_cities_index.py`` generated once. Within each city, 500 m geographic blocks are assigned to the splits so that chip count and building pixels both come close to 70/10/20, which gives 76,818 train, 11,236 validation, and 22,408 test chips. A chip is kept when at least half of it is valid imagery, and chips that duplicate an overlapping scene are dropped. Zanzibar is mostly rural bush, so its blocks with less than 1% building cover are dropped as well. ``scripts/plot_open_cities_splits.py`` draws the split map of each city and writes the block polygons as GeoJSON for inspection in a GIS.

The default training partition is a fixed 10,000-chip subset: the sorted first 10,000 entries of ``numpy.random.default_rng(0).permutation(76818)``. This is exactly the ``train-n10000-s0`` subset of the training-budget study, independent of the training seed. The packaged ``open_cities_train_10k.csv.gz`` pins the train rows and chip IDs with a SHA-256 checksum; loading also checks chip identities against the full index. ``scripts/generate_open_cities_subset.py`` reproduces the manifest. ``--partition full`` restores all 76,818 training chips; validation and test remain complete for both partitions.

``geo-eval-harness download open_cities`` fetches the 31 scenes and their label GeoJSONs (about 34 GB) from source.coop and verifies them against packaged SHA-256 checksums. The imagery is kept byte-identical. The download then rasterises one mask GeoTIFF per scene on the same pixel grid, with 0 for background, 1 for building, and 255 where the image is transparent or black, which the loss and metrics ignore. Normalization statistics exclude these nodata pixels.

A label audit against Google and Microsoft building footprints found the tier-1 labels well aligned with the imagery, while the Dar es Salaam scenes miss about 5% of buildings. In addition, Monrovia scenes ``207cc7`` and ``401175`` are stored in UTM zone 36N instead of 29N, so their ground pixels are not square. Labels are ODbL 1.0; the imagery is CC BY 4.0 or ODbL depending on the scene.

LoveDA — segmentation
---------------------

``loveda`` is seven-class land-cover segmentation of 0.3 m Google Earth RGB tiles from Nanjing, Changzhou, and Wuhan (Wang et al., 2021), with the urban and rural scenes combined. Tiles are **1024 x 1024** RGB PNGs. The official Test release has no public labels, so **test** is the official Val release (**1,669** tiles) and **train / val** (**2,269 / 253**) hold out 10% of each scene of the official Train release by a seeded hash. The memberships are checked in under ``src/geo_eval_harness/datasets/splits/loveda/``. Tile ids encode no location, so validation shares districts with train; the test release is region-disjoint from Train.

The classes, in model output order, are background, building, road, water, barren, forest, and agriculture (native ``1..7``). Native ``0`` is no-data (3.7% of training pixels, on near-black padding); it maps to ``255`` and is ignored in loss and mIoU. Images are raw float32 RGB in ``0..255``; normalization statistics use the training pixels that have a valid label.

.. code-block:: text

   data/loveda/
     Train.zip  Val.zip
     Train/{Urban,Rural}/{images_png,masks_png}/<id>.png
     Val/{Urban,Rural}/{images_png,masks_png}/<id>.png

.. code-block:: console

   $ geo-eval-harness download loveda
   $ geo-eval-harness run --config docs/examples/loveda-run.yaml --dry-run
   $ geo-eval-harness run --config docs/examples/loveda-run.yaml

The download fetches the MD5-verified ``Train.zip`` (4.0 GB) and ``Val.zip`` (2.4 GB) from `Zenodo record 5706578 <https://zenodo.org/records/5706578>`__ through torchgeo and needs about 13 GB of disk. The unlabelled ``Test.zip`` is not downloaded. The :download:`example config </examples/loveda-run.yaml>` is a no-weight-download smoke configuration, not an accuracy baseline.

Sources and attribution: the `LoveDA repository <https://github.com/Junjue-Wang/LoveDA>`__ and `paper <https://arxiv.org/abs/2110.08733>`__. The repository states CC BY-NC-SA 4.0 (no commercial use), while the Zenodo record is labelled CC BY 4.0; follow the more restrictive terms.

PLANTEUR — segmentation
-----------------------

``planteur`` is crop-type segmentation in France's overseas territories (Guadeloupe, Martinique, Guyane, Réunion, Mayotte and Saint-Martin), the tropical counterpart to PASTIS. It has 2,750 patches of 128x128 px at 10 m with twelve 2019 monthly mosaics of Sentinel-2 L2A (12 bands) and Sentinel-1 RTC (``vv``, ``vh``), whose linear backscatter the download converts to dB as in PASTIS (no-data zeros become -100 dB), labelled from the French Land Parcel Identification System (RPG 2019) into 8 classes: Background, Meadow, Fruits/vegetables/flowers, Orchard, Sugarcane, Banana, Pineapple and Tropical tuber. Crops outside that taxonomy are void (255) and ignored. It uses the official spatial-block, territory-stratified 1,896 / 411 / 443 split. ``geo-eval-harness download planteur`` fetches the pinned monthly mosaics and labels (17 GB), stacks them into memory-mapped per-split arrays, verifies them against the bundled checksums, and deletes the per-patch files. As with ``pastis``, ``--time-steps`` selects evenly spaced months (``(T, C, H, W)``); without it each sample is the first month (``(C, H, W)``). Data: ODC-BY, https://huggingface.co/datasets/allenai/planteur-benchmark.

The five leave-one-island-out folds, ``planteur_loio_guadeloupe``, ``planteur_loio_martinique``, ``planteur_loio_guyane``, ``planteur_loio_reunion`` and ``planteur_loio_mayotte``, read the same downloaded arrays. Each fold tests on every patch of its held-out territory (735, 550, 506, 813 and 140 patches) and trains and validates on the other territories, following the release's ``splits/loio.json``. The LOIO train and val splits share 0.1° blocks, so val scores are optimistic, while test never shares a block with them. Saint-Martin stays in train for every fold. Band statistics come from the default train split, which includes every island, so they are not held out for a fold.

Selecting datasets
------------------

Repeat ``--dataset`` for multiple selections, or use ``--dataset all`` to
evaluate every registered dataset. YAML accepts ``datasets: [m-eurosat, caffe]``:

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat
   $ geo-eval-harness run --model timm/resnet18 --dataset burn_scars --dataset pastis --dataset flair2
   $ geo-eval-harness run --model timm/resnet18 --dataset all

Bands selection
---------------

Each dataset declares an ordered list of :class:`~geo_eval_harness.datasets.BandSpec`
objects.  Three modes are supported:

* ``--bands rgb`` *(default)* — only the bands listed in
  :attr:`~geo_eval_harness.datasets.BenchDataset.rgb_bands`.
* ``--bands all`` — every band the dataset exposes.
* ``--bands red,green,blue,nir`` — an explicit subset.

The runner derives ``num_channels`` from the loaded tensor and constructs
the matching ``list[BandSpec]`` so the model wrapper can size its input
layer and per-channel normalization correctly.  The selected ``bands``
value is recorded in the results CSV so multiple runs writing to the same
file (and ``--resume``) keep RGB and multispectral results
distinguishable.

.. code-block:: console

   $ # All 13 Sentinel-2 bands on EuroSAT with a pretrained timm ResNet-18
   $ geo-eval-harness run --model timm/resnet18 --dataset m-eurosat --bands all

Multi-modality (V2)
-------------------

Several V2 datasets are multi-sensor (e.g. ``treesatai`` = aerial + S2 +
S1, ``pastis`` = S2 + S1, ``kuro_siwo`` = SAR + DEM).  Their wrappers
set ``band_order_strategy = "by_sensor"`` and the V2 base class groups
``BandSpec`` entries by sensor before passing them to the upstream
``geobench_v2`` loader.  End users do not need to do anything special —
set ``--bands all`` (or an explicit subset) and the right
per-modality tensors are concatenated into a single ``image`` key.

Model compatibility
^^^^^^^^^^^^^^^^^^^

* :doc:`timm <models>` wrappers rebuild the input conv for any
  ``num_channels``.
* :class:`~geo_eval_harness.models.RCFBench` and
  :class:`~geo_eval_harness.models.ImageStatsBench` are band-agnostic.
* The torchgeo RGB-only wrappers hold fixed-channel pretrained weights
  and don't currently adapt to non-RGB inputs — see
  `#16 <https://github.com/torchgeo/geo-eval-harness/issues/16>`__.
* :class:`~geo_eval_harness.models.TorchGeoDOFABench` resolves one wavelength per selected band from ``BandSpec`` metadata or known optical band names, with a placeholder for SAR. Unknown bands require an explicit ``wavelengths`` constructor argument.

Data partitions
---------------

V1 datasets, ``hotosm_buildings``, and ``open_cities`` support ``--partition``. HOTOSM and Open Cities use fixed training subsets by default; ``full`` selects their complete training splits. For V1, ``--partition`` selects a distributed partition JSON file. Other datasets ignore or reject non-default values.

.. code-block:: console

   $ # Train on 1% of the V1 training split, write to a separate CSV
   $ geo-eval-harness run --model rcf --dataset m-eurosat --partition 0.01x_train --output results/1pct.csv

Common partition values: ``default``, ``0.01x_train``, ``0.02x_train``,
``0.05x_train``, ``0.10x_train``, ``0.20x_train``, ``0.50x_train``,
``1.00x_train``.  The exact set available depends on which partition
JSON files ship with the dataset.
