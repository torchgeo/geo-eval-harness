.. module:: geo_eval_harness

geo-eval-harness
================

Evaluate geospatial image models and location encoders with classification, segmentation, and coordinate prediction tasks.

.. grid:: 1 2 2 4

   .. grid-item-card:: Benchmark a model
      :link: user/quickstart
      :link-type: doc

      Run frozen-backbone probes, LoRA, or full fine-tuning.

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

      Register a geospatial dataset and evaluate models on it.

.. toctree::
   :maxdepth: 2
   :hidden:

   user/index
   api/index
