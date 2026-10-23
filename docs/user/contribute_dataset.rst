Add a Dataset
=============

This page explains how to wire a new geospatial dataset into geo-eval-harness
so that any registered model can be evaluated on it automatically.

Prerequisites
-------------

Clone the repository and install the development dependencies:

.. code-block:: console

   $ git clone https://github.com/torchgeo/geo-eval-harness.git
   $ cd geo-eval-harness
   $ uv sync --extra dev

Use ``uv run`` for commands in that environment. Alternatively, activate the conda environment and run ``pip install -e ".[dev]"``; do not use ``uv sync`` to install into conda.

Download the data before loading it. ``download`` accepts one or more dataset names, including names from different families. GeoBench collection aliases also accept ``--datasets``:

.. code-block:: console

   $ uv run geo-eval-harness download burn_scars
   $ uv run geo-eval-harness download m-eurosat burn_scars resisc45
   $ uv run geo-eval-harness download geobench_v2 --datasets burn_scars

If you are adding a dataset that no existing family covers, you will wire up
its own download target below.

Implement BenchDataset
----------------------

Create a new module under :file:`src/geo_eval_harness/datasets/` and subclass
:class:`~geo_eval_harness.datasets.base.BenchDataset`:

.. code-block:: python

   from collections.abc import Callable
   from pathlib import Path
   from typing import ClassVar

   from torch import Tensor
   from torch.utils.data import Dataset

   from geo_eval_harness.datasets.base import BandSpec, BenchDataset

   class MyDataset(BenchDataset):
       name = "my_dataset"
       task = "classification"        # or "segmentation"
       num_classes = 10
       multilabel = False
       supports_partitions = False
       bands: ClassVar[list[BandSpec]] = [...]  # Supply measured BandSpecs.
       rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
       split_sizes: ClassVar[dict[str, int]] = {"train": 5000, "val": 1000, "test": 2000}

       @classmethod
       def data_root(cls) -> Path:
           return Path("data/my_dataset")

       def get_dataset(
           self,
           split: str,
           *,
           partition: str = "default",
           bands: tuple[str, ...] | None = None,
           transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
       ) -> Dataset[dict[str, Tensor]]:
           raise NotImplementedError("Implement loading and apply the requested bands/transform")

Required class-level attributes:

* ``name`` — unique string identifier used by the dataset registry and CLI
* ``task`` — ``"classification"`` or ``"segmentation"``
* ``num_classes`` — integer label count
* ``bands`` — list of :class:`~geo_eval_harness.datasets.BandSpec` objects
  supplying per-channel sensor / wavelength / normalisation stats.  See
  `Compute the band statistics`_ — these must be measured, not copied.
* ``rgb_bands`` — short names of the bands used in RGB-only mode
* ``split_sizes`` — dict with ``train``, ``val``, and ``test`` keys
* ``multilabel`` — ``True`` for multi-hot labels (``m-bigearthnet``, ``benv2``, ``treesatai``).
  Selects micro-mAP over accuracy as the reported metric, so getting it wrong
  silently reports the wrong number.
* ``supports_partitions`` — ``True`` only for V1 GeoBench datasets, which ship
  partition JSON files.  When ``False``, ``get_datasets`` warns and ignores a
  non-default ``input.partition``.

The ``get_dataset`` method takes ``split`` (``"train"``, ``"val"``, or
``"test"``) plus the keyword-only arguments ``partition``, ``bands`` (the
subset of bands requested by the model), and ``transform``.  It returns a
:class:`torch.utils.data.Dataset` whose ``__getitem__`` yields **dict**
samples — ``{"image": tensor, "label": tensor}`` for classification, with
``"mask"`` in place of ``"label"`` for segmentation.  Datasets always emit
raw float32 values; normalization is the model's job.

If you inherit from ``_V1Dataset`` or ``_V2Dataset``, both ``data_root`` and
``get_dataset`` are already implemented — your subclass is pure metadata.  See
:file:`src/geo_eval_harness/datasets/m_eurosat.py` for a minimal example.

.. note::

   **Loader families.** V1 datasets (``m-`` prefix) inherit from :class:`~geo_eval_harness.datasets.geobench_v1._V1Dataset` and normally use JSON-metadata tar shards under ``data/classification_v1.0_wds/``. Custom HDF5 with JSON metadata is also supported; pickle metadata is not. V2 wrappers inherit from :class:`~geo_eval_harness.datasets.geobench_v2._V2Dataset` and dispatch to ``geobench_v2.datasets`` classes over ``.tortilla`` files. A standalone torchgeo dataset, such as RESISC45, implements ``BenchDataset`` directly.

   For V2 loaders that accept bands grouped by modality, set ``band_order_strategy = "by_sensor"``. Override ``canonicalize_sample`` only when upstream image/mask keys or temporal shapes need adapting. Confirm the emitted channel order matches the selected ``BandSpec`` objects.

Band selection when the loader has no ``bands`` argument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runner passes ``get_dataset`` the band subset a model asked for, and then
checks that the loaded tensor's channel count matches the ``BandSpec`` list it
built.  How you honour that subset depends on the upstream loader:

* **The loader accepts bands.** Forward them and you are done —
  :class:`~geo_eval_harness.datasets.EuroSAT` passes ``source_name`` codes
  straight to :class:`torchgeo.datasets.EuroSAT`.
* **The loader does not.** Most torchgeo classification datasets are
  fixed-channel ``ImageFolder`` wrappers with no band argument at all, so the
  subset has to be applied in *your* wrapper, as a transform that indexes the
  channel axis.  :class:`~geo_eval_harness.datasets.RESISC45` is the worked
  example: see ``_make_band_select`` and the ``Compose`` call in
  :file:`src/geo_eval_harness/datasets/resisc45.py`.

Two details matter in the second case.  Compose your selection **before** the
``transform`` the caller handed you — that argument is the resize built by
:func:`~geo_eval_harness.datasets.get_datasets`, and it should only see channels
that survive selection.  And return ``None`` rather than an identity transform
when the selection is a no-op, so the common ``--bands rgb`` / ``--bands all``
paths add no per-sample work.

Compute the band statistics
---------------------------

Each :class:`~geo_eval_harness.datasets.BandSpec` carries ``mean``, ``std``, ``min``, and ``max``. They supply the default ``input.normalization: dataset`` strategy (``bandspec_zscore`` inside the model). Raw magnitudes also inform unit detection for ``input.normalization: model``. Incorrect statistics can silently mis-normalize inputs.

So measure them.  Two rules:

* **Train split only.**  Statistics that include val or test leak evaluation
  data into the normalisation every model sees.
* **Raw sensor units.**  Do not pre-scale to ``[0, 1]``; unit detection depends
  on the raw magnitudes.

Once your class is registered with placeholder statistics and loads, run:

.. code-block:: console

   $ uv run python scripts/compute_band_statistics.py --dataset my_dataset

It accumulates in float64 over the train split and prints a ``bands = [...]`` block. Copy the generated values into your wrapper while retaining the ``bands: ClassVar[list[BandSpec]]`` annotation required for mutable class metadata. Record in a comment that the numbers came from this script, so the next person knows they are measured rather than copied from a paper.

Register and configure
----------------------

**1. Register the class** by adding an entry to ``_REGISTRY_SPEC`` in
:file:`src/geo_eval_harness/datasets/loading.py`, mapping the dataset name to
its ``(submodule, class_name, task)``:

.. code-block:: python

   "my_dataset": ("my_dataset", "MyDataset", "classification"),

This entry enables loading, CLI listing/details, and run selection. Its task must be ``"classification"`` or ``"segmentation"`` and match the wrapper's ``task``. ``list_datasets()`` and ``get_dataset_task(name)`` read the registry without importing dataset wrappers. There is no second list to update in ``image_cli.py`` and no per-dataset YAML config.

The registry stores strings rather than imported classes so metadata queries stay cheap. ``get_bench_dataset_class`` imports only the requested wrapper.

**2. Export the class** from :file:`src/geo_eval_harness/datasets/__init__.py`
by adding an ``__all__`` entry and a matching ``_LAZY_CLASSES`` mapping:

.. code-block:: python

   __all__ = [
       # Keep the existing exports.
       "MyDataset",
   ]

   _LAZY_CLASSES: dict[str, str] = {
       # Keep the existing lazy mappings.
       "MyDataset": "my_dataset",
   }

Individual dataset classes load lazily through module ``__getattr__`` so that
``import geo_eval_harness.datasets`` — and CLI startup — stays fast.  Do not add
an eager ``from .my_dataset import MyDataset`` import; register the name in
``_LAZY_CLASSES`` instead.

Keep both lists alphabetically sorted; they are read by humans far more often
than by the loader.

**3. Wire up the download.** Which files you touch depends on the family:

* **GeoBench V2** — add the name → upstream class mapping to ``_V2_REGISTRY``
  in :file:`src/geo_eval_harness/datasets/geobench_v2.py`.
  ``DEFAULT_V2_DATASETS`` is derived from that registry automatically.
* **A standalone torchgeo wrapper** — add a ``download_<name>`` helper in :file:`src/geo_eval_harness/download.py`, include its name in ``TORCHGEO_DATASETS``, and route it in ``download_datasets``. ``DOWNLOADABLE_DATASETS`` is derived from the family lists. ``download_resisc45`` is the reference implementation. For a different download backend, extend the same explicit name validation and dispatch rather than adding parser-specific choices.

The CLI delegates through :file:`src/geo_eval_harness/commands/_download.py`; there is no ``_cmd_download`` function or download-target ``choices`` list to edit in ``cli.py``. Also verify the missing-data hint produced by ``download_command`` in ``datasets/loading.py`` points to a working download invocation. Downloading somewhere other than ``data/`` does not change the loader's fixed paths.

**4. Add the expected split sizes** to ``EXPECTED_SIZES`` in
:file:`tests/test_split_sizes.py`.  Those cases are marked
``@pytest.mark.slow`` and the default ``addopts`` deselect them, so verify
yours against the data on disk explicitly:

.. code-block:: console

   $ uv run pytest tests/test_split_sizes.py -m slow -k my_dataset

**5. Document it.** Three files, none optional:

* :file:`docs/user/datasets.rst` — a row in the relevant family table, plus
  the filesystem-layout table and the ``download`` command block if you added
  a new target.
* :file:`docs/api/datasets.rst` — an ``.. autoclass::`` entry, or your class
  gets no API page.
* :file:`docs/user/changelog.rst` — an entry under ``Unreleased``.

Run the smoke test
------------------

With the dataset on disk, run a quick benchmark to verify the dataset loads
and produces sensible results:

.. code-block:: console

   $ uv run geo-eval-harness datasets my_dataset
   $ uv run geo-eval-harness run --model imagestats --dataset my_dataset --device cpu \
       --methods knn --bootstrap-samples 10 --dry-run
   $ uv run geo-eval-harness run --model imagestats --dataset my_dataset --device cpu \
       --methods knn --bootstrap-samples 10

These are classification smoke commands. ``imagestats`` computes four summary statistics per channel, so RGB inputs produce 12-dimensional embeddings without downloading weights. Use a separate ``output.file`` in a run YAML when recording exploratory results. The default runtime device is ``cuda:0``; pass ``--device cpu`` when no GPU is available.

Compare class counts, sample labels, image channels, and the reported score with a simple baseline. Above-chance accuracy is a useful sanity check, not proof of correct loading; near-chance results can also mean the features are unsuitable.

For segmentation, use a spatial backbone with a compatible head and feature layers, not ``imagestats``. Configure ``segmentation`` in the run YAML, check masks and ignore labels on a small sample, and exercise the loader with the existing segmentation tests. See :doc:`segmentation-layers` for backbone layers.

Add fast tests using tiny local fixtures for split selection, band order, raw dtype, label/mask shape, and missing-data errors. Keep real-data tests marked ``slow`` and run the applicable ones locally.

Add the geographic metadata
---------------------------

Every registered dataset carries a record in the committed geographic store
under :file:`docs/_static/_dataset_geography/`, which drives the spatial
coverage map.  Generate yours:

.. code-block:: console

   $ uv run python scripts/extract_dataset_geography.py --dataset my_dataset

Commit the resulting :file:`docs/_static/_dataset_geography/my_dataset.json`
together with the regenerated :file:`index.json`.

:file:`tests/test_geography.py` fails when a registered dataset has no record,
so this step is not optional — but "no coordinates" is a perfectly valid
answer.  If the dataset genuinely carries no geolocation, add it to ``NO_GEO``
in :file:`src/geo_eval_harness/geography.py` with the reason you verified, and
the map will disclose the gap rather than quietly omit the dataset.

Coordinates are read from V2 ``.tortilla`` metadata or V1 JSON affine/CRS metadata in shards and custom HDF5. A different storage layout needs an explicit extraction path in ``extract_geography``.

Once results look sensible, follow the PR workflow described in
:doc:`contribute_model` to open a pull request.

A worked example
----------------

`#234 <https://github.com/torchgeo/geo-eval-harness/pull/234>`__ adds NWPU-RESISC45
and touches every step on this page: a torchgeo wrapper whose loader takes no
``bands`` argument, measured band statistics, its own download target, a
``no_geo`` record with the check that justified it, and unit tests that run
without the data on disk.  Read
:file:`src/geo_eval_harness/datasets/resisc45.py` alongside this guide.
