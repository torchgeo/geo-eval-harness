Evaluate your own model (Stage 1)
==================================

This guide demonstrates how you can benchmark any frozen pretrained geospatial model against
the included benchmark datasets. If you want to contribute a new open-source model with available weights such that the broader community can easily access your model, see
:doc:`contribute_model`.

.. _eval-prerequisites:

Prerequisites
-------------

Clone the repository and install the package with uv:

.. code-block:: console

   $ git clone https://github.com/torchgeo/geo-eval-harness.git
   $ cd geo-eval-harness
   $ uv sync --extra dev

Run commands through ``uv run`` so they use that installation. Alternatively, activate the ``geo-eval-harness`` conda environment, run ``pip install -e ".[dev]"``, and omit ``uv run`` below. Do not combine conda activation with ``uv sync``: uv manages a separate environment.

If your model requires optional dependencies (e.g. a special
:doc:`model library </user/models>` or a custom tokenizer), install the
matching extra:

.. code-block:: console

   $ uv sync --extra dev --extra <newextra>

You can check how to download one or more dataset for evaluation in the :doc:`datasets` guide.

.. _eval-implement:

Implement your model
--------------------

We provide a template file to give you a general setup and fill in the gaps that are unique to your model and ensure
that each of those parts will be used correctly in the benchmark pipeline. Copy the template file to your working directory and fill in the ``TODO``
sections:

.. code-block:: console

   $ cp src/geo_eval_harness/models/contrib_template.py ./new_model.py

The template defines ``NewModel``. Its constructor receives the selected ``bands`` and the requested ``normalization`` strategy. Forward both to ``BenchModel`` rather than hard-coding a strategy or silently discarding the caller's choice. Implement ``_forward_patch_features``; leave ``forward_patch_features`` unchanged so normalization is applied exactly once.

**Normalization strategy decision table**

Pick the strategy that matches how your backbone was trained:

.. list-table::
   :header-rows: 1
   :widths: 20 42 38

   * - CLI / YAML choice
     - When to use — in-repo examples
     - How to set it
   * - ``dataset`` (default)
     - The framework z-scores each channel from
       the dataset's BandSpec statistics, with the goal of producing ~N(0, 1) inputs regardless
       of source sensor unit.

     - Forward the constructor's ``normalization`` argument. The runner maps this choice to ``bandspec_zscore``.
   * - ``none``
     - Your backbone ships its own normalizer and must receive raw sensor
       values — applying a second normalization on top would corrupt the
       inputs.

       *In-repo example:* OlmoEarth — its internal ``Normalizer`` consumes
       raw DN/reflectance directly and auto-detects the sensor scale.  See
       :class:`~geo_eval_harness.models.OlmoEarthBenchModel` and
       :file:`src/geo_eval_harness/models/olmoearth.py` for the pattern.
     - The runner passes ``identity``. If the model always normalizes internally, declare ``handles_own_normalization = True`` on the wrapper, as OlmoEarth does.
   * - ``model``
     - The exact pretraining input scale is published and you can declare it
       explicitly.  The framework converts the dataset's sensor unit to the
       backbone's expected unit, then applies any declared per-channel
       mean/std.

     - Declare ``expected_input_unit`` using ``InputUnit`` from ``geo_eval_harness.models``, plus matching ``pretrain_mean`` and ``pretrain_std``, or implement the wrapper's explicit model-native normalizer. Select ``--normalization model``; declaring the attributes alone does not change the default.
   * - ``minmax``
     - Scale each band using its dataset ``BandSpec.min`` and ``BandSpec.max``.
     - Forward ``normalization`` unchanged; the constructor strategy is also named ``minmax``.
   * - ``minmax_zscore``
     - Min-max scale each channel, then z-score using the correspondingly scaled dataset statistics.
     - Forward ``normalization`` unchanged; the constructor uses the same name.

For the full list of available strategies and their exact semantics, see
:file:`src/geo_eval_harness/models/_normalization.py`.

Accessing band metadata
^^^^^^^^^^^^^^^^^^^^^^^

The template shows ``backbone(images)`` as the minimal forward call, but many
models need more than raw pixels — for example a wavelength list for
band-agnostic ViTs, or sensor-conditional routing.

The framework makes this straightforward.  The pipeline **reinstantiates your
class once per dataset**, so the ``bands`` argument passed to ``__init__``
always reflects exactly the channels being loaded for that run.  Every
:class:`~geo_eval_harness.datasets.BandSpec` in that list carries the
dataset-level metadata that is available:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``wavelength_um``
     - Centre wavelength in micrometres (``None`` for non-optical bands such as
       SAR backscatter or DEM elevation).  Use this to drive wavelength-aware
       embeddings (e.g. DOFA).
   * - ``sensor``
     - Sensor family string — ``"s2"``, ``"landsat"``, ``"sar"``, ``"aerial"``,
       ``"planet"``, ``"worldview"``.  Use this for sensor-conditional routing
       or to detect unsupported modalities at construction time.
   * - ``name``
     - Canonical short band name — ``"red"``, ``"nir"``, ``"vv"``, ``"b02"``.
       Use this when your backbone expects bands in a named order.

Extract the metadata you need from ``bands`` in ``__init__``, then use it in ``_forward_patch_features``. For example, after replacing the backbone placeholder:

.. code-block:: python

   import torch

   from geo_eval_harness.datasets.base import BandSpec
   from geo_eval_harness.models.interface import BenchModel

   class NewModel(BenchModel):
       def __init__(
           self,
           bands: list[BandSpec],
           *,
           normalization: str = "bandspec_zscore",
           **kwargs: object,
       ) -> None:
           super().__init__(bands=bands, normalization=normalization, **kwargs)

           # Each model instance receives the bands selected for its dataset.
           self.wavelengths = [b.wavelength_um for b in bands]  # Missing for radar and elevation bands.
           self.sensors = [b.sensor for b in bands]             # e.g. "s2", "landsat"
           self.band_names = [b.name for b in bands]            # e.g. "red", "nir"

           self.backbone = ...  # Load your model here.

       def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
           return self.backbone(images, wavelengths=self.wavelengths)

For a complete example see ``TorchGeoDOFABench`` in
:file:`src/geo_eval_harness/models/torchgeo_models.py`, which reads
``wavelength_um`` from each ``BandSpec`` at construction and passes the
resulting list to ``backbone.forward_features(images, wavelengths=...)``.

.. _eval-model-config:

Create a model config
---------------------

Create a model YAML file at :file:`src/geo_eval_harness/conf/model/new_model.yaml`.
The preset identifies your importable class and its result name. Add any constructor options your model needs:

.. code-block:: yaml

   name: new_model
   target: new_model.NewModel
   track: image
   kwargs:
     pretrained: true

Only ``kwargs`` are passed as constructor options. Keep input and evaluation defaults in the preset's ``input``, ``classification``, and ``segmentation`` sections, not inside ``kwargs``. Dataset-specific defaults belong in ``dataset_overrides``. These are Pydantic-validated mappings; ``_target_``, interpolation, and recursive instantiation are not supported.

.. note::

   **Do not put** ``bands`` **in the YAML.** The pipeline selects the current dataset's :class:`~geo_eval_harness.datasets.BandSpec` objects at runtime. Configure normalization through ``input.normalization`` or ``--normalization``, not by overriding the model constructor in the preset.

If your class is not importable from the default Python path, add the
parent directory to ``PYTHONPATH`` before running:

.. code-block:: console

   $ export PYTHONPATH="$PWD:$PYTHONPATH"

.. _eval-run:

Run the benchmark
-----------------

Select the preset with ``--model`` and repeat ``--dataset`` for each applicable dataset. The CLI accepts explicit flags and Pydantic-validated YAML; the old ``key=value`` interface is retired:

.. code-block:: console

   $ uv run geo-eval-harness models new_model
   $ uv run geo-eval-harness run --model new_model --dataset m-eurosat --device cpu --dry-run
   $ uv run geo-eval-harness run --model new_model --dataset m-eurosat --device cpu
   $ uv run geo-eval-harness run --model new_model \
       --dataset m-eurosat --dataset m-bigearthnet --dataset benv2 --device cuda:0

For segmentation, the wrapper must expose suitable spatial backbone layers. Configure ``segmentation.layers`` and a compatible head in a run YAML; a pooled ``(B, K)`` output alone is not enough. See :doc:`segmentation-layers`.

Skip the (slow) linear probe and reduce bootstrap samples for a quick trial:

.. code-block:: console

   $ uv run geo-eval-harness run --model new_model --dataset m-eurosat \
       --device cpu --methods knn --bootstrap-samples 10

For a reusable run, create ``new_model_run.yaml``. This is a run configuration, separate from the model preset:

.. code-block:: yaml

   model:
     name: new_model
   datasets: [m-eurosat]
   input:
     normalization: dataset
   runtime:
     device: cpu
   output:
     file: results/new_model_results.csv

Validate it before running, then use ``--resume`` to continue against the same output:

.. code-block:: console

   $ uv run geo-eval-harness run --config new_model_run.yaml --dry-run
   $ uv run geo-eval-harness run --config new_model_run.yaml
   $ uv run geo-eval-harness run --config new_model_run.yaml --resume

Only explicitly supplied flags override YAML values. ``run --config-help`` describes the supported fields, and :file:`docs/examples/image-run.yaml` provides a complete example.

You can also evaluate an external class without adding a packaged preset. In the run YAML, provide its importable target and constructor options directly:

.. code-block:: yaml

   model:
     name: new_model
     target: new_model.NewModel
     kwargs:
       pretrained: true
   datasets: [m-eurosat]
   runtime:
     device: cpu
   output:
     file: results/new_model_results.csv

Run this file with ``--config``. The copied module's directory must still be on ``PYTHONPATH``. For either form, precedence is built-in defaults, preset defaults, dataset-specific defaults, explicit run YAML, then explicit flags.

.. _eval-results:

Results
-------

Results are appended to ``results/models/<model name>.csv`` by default, or to ``output.file`` when supplied. Use a separate file for exploratory runs so they do not change the reference results. ``--resume`` skips completed work only when the effective configuration matches; changing a result-affecting setting causes it to run again.

The standalone ``profile`` command emits a JSON record to stdout by default; ``--output-dir`` or ``--output`` writes that record to a file instead. See :doc:`configuration` for output paths and optional measurement settings, and :doc:`results-format` for the CSV columns.
