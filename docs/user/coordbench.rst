CoordBench location encoders
============================

CoordBench evaluates frozen location encoders without imagery. Each benchmark
maps a point ``(longitude, latitude[, year])`` to one or more downstream labels.
The runner embeds each point once, then fits KNN or ridge-linear probes under
random, spatial-block, or official held-out splits.

The normalized benchmark tables stream from the
`taylor-geospatial/coordbench <https://huggingface.co/datasets/taylor-geospatial/coordbench>`_
dataset on Hugging Face. There is no separate download command and no local
``data/`` layout to prepare.

First run
---------

The base installation includes the dependency-free sine/cosine baseline. This
small CPU example evaluates one regression benchmark with two random folds:

.. code-block:: console

   $ geo-eval-harness coord --model sincos --dataset california_housing \
       --methods linear --folds 2 --device cpu \
       --output results/coordbench_quickstart.csv

Results are appended to ``coordbench_results.csv`` beneath ``output.directory`` (``--output-dir``, default ``results``), unless ``output.file`` (``--output``) selects an exact path. Add ``--resume`` to skip rows that already match ``(dataset, task, method, model_name, split)``.
Unlike image-run resume, this key does not include a configuration hash;
use a separate output file when changing encoder kwargs or evaluation
settings that are not part of the key.

``python -m geo_eval_harness coord`` and ``python -m geo_eval_harness.cli coord``
are equivalent entry points. The former ``run mode=coord`` and dotted
``key=value`` overrides are no longer supported.

Included encoders
-----------------

================  ==============  =============================================
Model preset      Installation    Description
================  ==============  =============================================
``sincos``        base            Four-dimensional periodic coordinate baseline.
``mind``          base            64-dimensional MIND Matryoshka prefix.
``mind-small``    base            128-dimensional distilled MIND student.
``climplicit``    ``coordbench``  Climplicit climate-specialist encoder.
``geoclip``       ``coordbench``  GeoCLIP Equal-Earth/RFF encoder.
``satclip``       ``coordbench``  SatCLIP spherical-harmonic encoder.
``sinr``          ``coordbench``  SINR species-distribution encoder.
================  ==============  =============================================

Install the optional reference encoders from PyPI with:

.. code-block:: console

   $ pip install "geo-eval-harness[coordbench]"

For example, run SatCLIP on its five downstream benchmarks with random and
spatial-block cross-validation:

.. code-block:: console

   $ geo-eval-harness coord --model satclip --dataset satclip --split both

Benchmarks and probes
---------------------

The ``linear`` probe uses ridge with an unpenalized intercept, matching ``sklearn.linear_model.Ridge(fit_intercept=True)`` on the same preprocessed features. Each fit centers features and targets using training-fold means and restores the target mean for prediction. Feature standardization and alpha selection remain fold-local.

By default, the probe selects the best mean CV score over 21 half-decade penalties from ``1e-4`` to ``1e6``. Set ``evaluation.ridge_alphas: [0.01, 0.1, 1.0, 10.0]`` in YAML or ``--ridge-alphas 0.01 0.1 1 10`` to choose a grid. Supply one positive value, such as ``--ridge-alphas 1``, to disable alpha tuning. With an official holdout, tuning uses only the training pool; otherwise, the reported folds are also used to select alpha. Use a fresh output CSV when changing the grid because resume keys do not include it.

``--dataset`` accepts one or more family or benchmark names, for example
``--dataset pdfm satclip``. The corresponding YAML field is a list:
``datasets: [pdfm, satclip]``. Use ``--dataset all`` or ``datasets: [all]``
to select the full suite; do not combine ``all`` with other names.
Available families are ``pdfm``,
``air_temp``, ``california_housing``, ``satclip``, ``sustainbench``,
``better_together``, ``cdc_places``, ``usavars``, ``country``, ``ecoregions``,
``worldclim``, ``soilgrids``, and ``deepmind``.

Classification tasks report accuracy and support ``knn`` and ``linear``.
Regression tasks report R2 and use ``linear``; requested KNN rows are skipped
because this track does not define a KNN regressor. ``evaluation.split``
(``--split``) controls the
holdout:

``random``
   Seeded k-fold cross-validation.
``spatial``
   K-fold cross-validation over geographic grid cells. ``evaluation.cell_deg``
   (``--cell-deg``) sets the cell width in degrees.
``both``
   Run both cross-validation protocols. Benchmarks with an official test mask
   use that fixed holdout once instead.

Benchmarks with an official test mask use that fixed holdout regardless of the
requested cross-validation protocol; ``both`` does not duplicate it. CSV split
labels are ``random``, ``spatial``, or ``official``. Cross-validation intervals
report mean plus/minus the standard deviation across folds, not image-style
bootstrapped confidence intervals.

YAML configuration
------------------

Pass a file to ``geo-eval-harness coord --config my-coord.yaml``:

.. code-block:: yaml

   schema_version: 1
   model:
     name: sincos
   datasets: [california_housing]
   evaluation:
     methods: [knn, linear]
     split: random
     folds: 5
     cell_deg: 10.0
     knn_k: 5
     knn_device: cpu
   runtime:
     device: cpu
     seed: 0
   output:
     directory: results
     file: null
     resume: false

Explicit flags override YAML, including ``--no-resume``.
``--methods knn``, ``--methods linear``, and ``--methods knn linear`` select
probes. ``--folds`` must be at least two. Other evaluation flags are
``--cell-deg``, ``--knn-k``, and ``--knn-device``. ``--device`` selects the
encoder/linear-probe device, and ``--seed`` controls reproducibility.

``--dry-run`` validates and prints YAML without loading encoder weights or
remote tables:

.. code-block:: console

   $ geo-eval-harness coord --config docs/examples/coord-run.yaml --dry-run

The coordinate schema deliberately has no image ``input``, ``segmentation``,
or ``classification`` sections. See :doc:`/api/coordbench` for
``CoordConfig`` and the Python loader.

Add a location encoder
----------------------

A custom encoder subclasses :class:`~geo_eval_harness.coordbench.LocationEncoder`
and implements ``_encode(lon, lat, year)``. The method receives one batch of
NumPy arrays and returns a finite ``(N, D)`` feature matrix.

The repository includes a complete Fourier-feature example in
:file:`docs/examples/coordbench_location_encoder.py`. Run it from the repository
root using :file:`docs/examples/coord-run.yaml`:

.. code-block:: yaml

   model:
     name: fourier
     target: docs.examples.coordbench_location_encoder.FourierLocationEncoder
     kwargs:
       num_frequencies: 8
   datasets: [california_housing]
   evaluation:
     methods: [linear]
     folds: 2
   runtime:
     device: cpu
   output:
     file: results/fourier_coordbench.csv

.. code-block:: console

   $ PYTHONPATH=. uv run geo-eval-harness coord --config docs/examples/coord-run.yaml

For a reusable integration, place the class in an installed package and add a
model preset under :file:`src/geo_eval_harness/conf/model/`:

.. code-block:: yaml

   name: fourier
   target: my_package.FourierLocationEncoder
   track: coord
   kwargs:
     num_frequencies: 8

``track`` is preset metadata, not a constructor argument. The runner passes
the execution device explicitly when constructing the encoder. See
:doc:`configuration` for preset resolution and :doc:`/api/coordbench` for
the public classes and probe functions.

Aggregate results
-----------------

The leaderboard helper groups scores by task family and reports mean rank for
random and spatial holdouts:

.. code-block:: console

   $ python -m geo_eval_harness.coordbench.leaderboard \
       results/coordbench_results.csv --method linear
