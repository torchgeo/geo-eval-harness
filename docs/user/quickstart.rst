Quickstart
==========

Install ``geo-eval-harness`` (see :doc:`installation`), download a dataset,
and run a frozen-backbone benchmark. Commands accept explicit flags or
strict YAML; module entry points use the same interface.

Download data
-------------

Datasets live under ``./data/`` relative to the current working directory.
For a small first run, download EuroSAT V1:

.. code-block:: console

   $ geo-eval-harness download m-eurosat

You can also download multiple named datasets or a collection:

.. code-block:: console

   $ geo-eval-harness download m-eurosat m-pv4ger
   $ geo-eval-harness download geobench_v1 --datasets m-eurosat
   $ geo-eval-harness download geobench_v2 --datasets benv2,burn_scars
   $ geo-eval-harness download eurosat

``download geobench_v1`` without a subset fetches the entire V1
classification collection. V1 uses the pickle-free JSON-metadata mirror;
old pickle-based caches must be replaced. Benchmark runs require local
data and report a download command if it is missing. See :doc:`datasets`
for names and canonical destination directories.

Run a benchmark
---------------

Inspect the catalogs without loading models or data:

.. code-block:: console

   $ geo-eval-harness models
   $ geo-eval-harness models rcf
   $ geo-eval-harness datasets
   $ geo-eval-harness datasets m-eurosat

Run Random Convolutional Features on EuroSAT V1 with KNN-5, linear probing,
and 200 bootstrap resamples:

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu \
       --output results/my_run.csv

The image default is ``cuda:0``. Use ``--device cpu`` on a machine without
CUDA, or ``--device auto`` to choose automatically. A pretrained backbone
may download its weights on first use:

.. code-block:: console

   $ geo-eval-harness run --model timm/resnet50 \
       --dataset m-eurosat --dataset m-pv4ger --device cpu

Select just the probe you need:

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu \
       --methods knn --bootstrap-samples 100
   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu \
       --methods linear

Use ``--knn-device cpu`` to keep KNN on CPU while extracting features on
another device. If the installed FAISS backend lacks GPU support, the runner
logs its KNN CPU fallback.

Use a YAML file
---------------

Save the following as ``my-run.yaml``:

.. code-block:: yaml

   model:
     name: rcf
   datasets: [m-eurosat]
   classification:
     methods: [knn]
     bootstrap_samples: 100
   runtime:
     device: cpu
   output:
     file: results/my_run.csv

Validate without loading a model or data, then execute:

.. code-block:: console

   $ geo-eval-harness run --config my-run.yaml --dry-run
   $ geo-eval-harness run --config my-run.yaml

Explicit flags override YAML. Omitted YAML fields inherit model and dataset
defaults; explicit ``null``, ``false``, and ``[]`` are preserved.
See :doc:`configuration` and :file:`docs/examples/image-run.yaml` for calibration,
segmentation, temporal inputs, optional profiling, and intrinsic dimension.

.. warning::

   Old ``key=value`` / ``+key=value`` commands are no longer accepted.
   For example, use ``--model rcf``, not ``model=rcf``. This also applies
   to ``python -m geo_eval_harness`` and ``python -m geo_eval_harness.cli``.

Profile a model
---------------

The standalone profiler measures a fixed real dataset batch and writes
JSON to stdout:

.. code-block:: console

   $ geo-eval-harness profile --model rcf --dataset m-eurosat --device cpu \
       --batch-size 8 --warmup 1 --measurements 5 > profile.json

For synthetic per-sample compute accounting without dataset samples:

.. code-block:: console

   $ geo-eval-harness flops --model rcf --device cpu --band-configs rgb \
       --seg-heads --output results/my_compute_cost.csv

See :doc:`configuration` for the distinct profile and FLOPs YAML schemas.

Benchmark a location encoder
----------------------------

CoordBench does not require an image download. It loads point-label tables
from Hugging Face, so uncached tables require network access:

.. code-block:: console

   $ geo-eval-harness coord --model sincos --dataset california_housing \
       --methods linear --folds 2 --device cpu \
       --output results/coordbench_quickstart.csv

See :doc:`coordbench` for pretrained encoders, spatial cross-validation,
official holdouts, and a complete custom-encoder example.

Resume and inspect results
--------------------------

Re-run the same image command with ``--resume`` to skip completed
method/config combinations:

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu \
       --output results/my_run.csv --resume

Without an explicit output file, image metrics append to
``results/models/<model name>.csv``. Those files **ship pre-populated** with
reference results; ``--output`` keeps your run separate. Optional image-run
profile and intrinsic-dimension measurements normally use their own
directories, but an explicit output file combines them with image metrics.
Each evaluation is saved when it finishes.

Rows follow :class:`~geo_eval_harness.main.EvaluationResult` and can be
loaded with pandas:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("results/my_run.csv")
   print(df.groupby(["dataset", "method"])["metric_value"].mean())

See :doc:`results-format` for the full schema and resume keys.
