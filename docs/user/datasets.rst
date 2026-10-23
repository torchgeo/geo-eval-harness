Datasets
========

``geo-eval-harness`` supports two generations of GeoBench datasets — V1 and
V2 — plus wrappers around torchgeo's standalone EuroSAT, NWPU-RESISC45, and UC Merced datasets, AID (rehosted from Hugging Face), and HOTOSM building segmentation (Hugging Face). All datasets share the
:class:`~geo_eval_harness.datasets.BenchDataset` interface and are
registered by name so they can be selected without importing every loader.

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

   * - ``ucmerced``
     - ``data/ucmerced/``
     - torchgeo's ``UCMerced`` downloader
   * - ``hotosm_buildings``
     - ``data/hotosm_buildings/``
     - Hugging Face ``hotosm/vhr-building-segmentation``, pinned commit, converted to per-split parquet
   * - ``planteur``
     - ``data/planteur/``
     - Hugging Face ``allenai/planteur-benchmark``, pinned commit, monthly mosaics stacked into per-split arrays

Downloading
-----------

The :doc:`/api/cli` accepts one or more dataset names. Collection aliases remain available for downloading a whole GeoBench family or a selected subset:

.. code-block:: console

   $ geo-eval-harness download m-eurosat burn_scars resisc45            # selected datasets
   $ geo-eval-harness download geobench_v1                              # all six V1 datasets
   $ geo-eval-harness download geobench_v1 --datasets m-eurosat         # V1 subset
   $ geo-eval-harness download geobench_v2                              # default V2 set
   $ geo-eval-harness download geobench_v2 --datasets benv2,burn_scars  # V2 subset
   $ geo-eval-harness download eurosat                                  # torchgeo EuroSAT
   $ geo-eval-harness download resisc45                                 # torchgeo RESISC45
   $ geo-eval-harness download aid                                      # isaaccorley/aid rehost

   $ geo-eval-harness download ucmerced                                 # torchgeo UC Merced
   $ geo-eval-harness download hotosm_buildings                         # HOTOSM VHR buildings
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

.. warning::

   Existing pickle-based V1 caches are not converted or unpickled. Replace them with ``geo-eval-harness download geobench_v1`` (or ``--datasets m-eurosat`` for a subset). If an archive checksum fails, remove the named corrupt archive and retry the download. Custom datasets must supply JSON metadata.

Geography extraction reads the JSON metadata directly from the default shards or custom HDF5 files. ``scripts/repack_geobench_v1.py`` repacks custom JSON-metadata HDF5 files into JSON-based shards; it does not read or convert pickle metadata.

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
     **multilabel** scene tags, S2 top-of-atmosphere DN.
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
``caffe``            4      1     aerial grayscale                             :class:`~geo_eval_harness.datasets.CaFFe`
``cloudsen12``       4      12                                                 :class:`~geo_eval_harness.datasets.CloudSEN12`
``dynamic_earthnet`` 7      16                                                 :class:`~geo_eval_harness.datasets.DynamicEarthNet`
``flair2``           13     5     aerial + Sentinel-2                          :class:`~geo_eval_harness.datasets.FLAIR2`
``fotw``             4      4     Fields of the World                          :class:`~geo_eval_harness.datasets.FieldsOfTheWorld`
``kuro_siwo``        4      3     SAR ``vv`` / ``vh`` + DEM (no RGB triplet)   :class:`~geo_eval_harness.datasets.KuroSiwo`
``pastis``           20     16    Sentinel-2 + Sentinel-1 (multi-modal)        :class:`~geo_eval_harness.datasets.PASTIS`
``spacenet2``        2      9     WorldView 8-band + pan                       :class:`~geo_eval_harness.datasets.SpaceNet2`
``spacenet7``        2      3                                                  :class:`~geo_eval_harness.datasets.SpaceNet7`
==================== ====== ===== ============================================ ==============================================================

Other
-----

==================== ============================================================================
CLI name             Class
==================== ============================================================================
``eurosat``          :class:`~geo_eval_harness.datasets.EuroSAT`  (torchgeo wrapper, random split)
``eurosat-spatial``  :class:`~geo_eval_harness.datasets.EuroSATSpatial`  (longitude-based split)
``resisc45``         :class:`~geo_eval_harness.datasets.RESISC45`  (45-class aerial scenes, RGB)
``aid``              :class:`~geo_eval_harness.datasets.AID`  (30-class aerial scenes, RGB)

``ucmerced``         :class:`~geo_eval_harness.datasets.UCMerced`  (21-class aerial land use, RGB)
``hotosm_buildings`` :class:`~geo_eval_harness.datasets.HOTOSMBuildings`  (building segmentation, RGB)
``planteur``         :class:`~geo_eval_harness.datasets.PLANTEUR`  (crop-type segmentation, S2 + S1, 12 months)
``planteur_loio_*``  ``PLANTEURLOIOGuadeloupe`` … ``PLANTEURLOIOMayotte``  (leave-one-island-out folds)
==================== ============================================================================

``resisc45`` is 31,500 RGB scenes at 256x256 across 45 classes, on torchgeo's
published 60/20/20 split.  It is the most-divergent benchmark in the GFM
literature — the same Scale-MAE checkpoint is reported at 33.0 and 89.6
linear-probe accuracy by different papers under the same nominal protocol —
which is why it is worth running under a fixed harness.  The imagery carries
no geolocation, so it appears on the coverage map as an explicit gap rather
than being silently omitted.

``aid`` contains 10,000 RGB scenes at 600x600 across 30 classes (Xia et al., 2017). With no official split, ``scripts/generate_aid_splits.py`` generates a deterministic, stratified 60/20/20 split. Downloads use the pinned, checksum-verified ``isaaccorley/aid`` rehost. Upstream specifies no license or image geolocation. Band subsets and ordering are applied before user transforms.

``ucmerced`` contains 2,100 RGB aerial images across 21 land use classes, resized to 256x256 by torchgeo. Its published split files contain 1,260 training, 420 validation, and 420 test images. Normalization statistics use the training split only. The source resolution is 1 ft (0.3048 m); the shared ``aerial`` sensor tag uses an approximate 1 m GSD for resolution-aware models. Images have no georeferencing, which is recorded in the coverage map.

.. code-block:: console

   $ geo-eval-harness run --model imagestats --dataset ucmerced --device cpu

``hotosm_buildings`` is binary building segmentation on 72,363 RGB tiles (256x256, zoom 19, about 0.3 m GSD) from 93 Humanitarian OpenStreetMap Team projects. Imagery comes from OpenAerialMap (CC-BY 4.0) and labels from OpenStreetMap (ODbL 1.0). It uses the official 57,890 / 7,237 / 7,236 train/val/test split, which is grouped by project; 5 tile coordinates appear in both train and val, and overlapping projects repeat some tiles within a split. Train is skewed: three Myanmar projects hold 76% of its tiles, and Eswatini appears in val and test only. ``geo-eval-harness download hotosm_buildings`` fetches the pinned parquet (5.9 GB), keeps the image bytes, re-encodes the masks as small PNGs, and deletes the upstream shards, leaving about 1.1 GB. Masks are 0 (background), 1 (building), and 255 where the image is transparent nodata, which the segmentation probe ignores. All three splits load into about 1.5 GB of RAM. With spawn or forkserver multiprocessing (including the defaults on macOS and Windows), use ``--workers 0``; forked workers share the encoded splits through copy-on-write. The full train split is too large for the default in-memory feature cache, so set ``segmentation: {cache_features: false}`` in the ``--config`` YAML for every model.

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
* :class:`~geo_eval_harness.models.TorchGeoDOFABench` accepts variable
  channels via wavelength tokens but the current wrapper hard-codes
  Sentinel-2 RGB wavelengths — see
  `#15 <https://github.com/torchgeo/geo-eval-harness/issues/15>`__.

Data partitions (V1 only)
-------------------------

V1 datasets honour the ``--partition`` argument (which selects one
of the partition JSON files distributed with each dataset).  V2 datasets
ignore it.

.. code-block:: console

   $ # Train on 1% of the V1 training split, write to a separate CSV
   $ geo-eval-harness run --model rcf --dataset m-eurosat --partition 0.01x_train --output results/1pct.csv

Common partition values: ``default``, ``0.01x_train``, ``0.02x_train``,
``0.05x_train``, ``0.10x_train``, ``0.20x_train``, ``0.50x_train``,
``1.00x_train``.  The exact set available depends on which partition
JSON files ship with the dataset.
