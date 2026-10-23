.. module:: geo_eval_harness

geo-eval-harness
================

A lightweight benchmarking framework for frozen geospatial foundation models.
Choose your path below.

.. grid:: 1 2 2 4

   .. grid-item-card:: Benchmark a model
      :link: user/quickstart
      :link-type: doc

      Evaluate any frozen backbone on GeoBench V1 / V2.

   .. grid-item-card:: Bring your own model
      :link: user/eval_own_model
      :link-type: doc

      Implement ``BenchModel`` and run your pretrained GeoFM against the full suite.

   .. grid-item-card:: Benchmark a location encoder
      :link: user/coordbench
      :link-type: doc

      Evaluate coordinate-only encoders on CoordBench.

   .. grid-item-card:: Add a dataset
      :link: user/contribute_dataset
      :link-type: doc

      Wire a new geospatial dataset into the benchmark and generate results.

.. toctree::
   :maxdepth: 2
   :hidden:

   user/index
   api/index
