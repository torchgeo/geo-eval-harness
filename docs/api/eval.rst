Evaluation
==========

The evaluation pipeline lives in :mod:`torchgeo_bench.main` and a few
focused sub-modules.  Each evaluation method (KNN-5, linear probe,
segmentation, intrinsic dimension) consumes per-split feature embeddings
or raw images and produces one :class:`EvaluationResult` row per metric.
Image orchestration accepts
:class:`~torchgeo_bench.config.run.RunConfig`; see :doc:`cli` for loading,
preset resolution, and the separate standalone measurement schemas.

Result schema
-------------

.. currentmodule:: torchgeo_bench.main

.. autoclass:: EvaluationResult

Feature extraction
------------------

.. autofunction:: embed_split

Bootstrap helpers
-----------------

.. autofunction:: bootstrap_accuracy
.. autofunction:: bootstrap_map

KNN-5 evaluation
----------------

.. autofunction:: evaluate_knn

.. currentmodule:: torchgeo_bench.knn
.. autoclass:: KNNClassifier

Linear probing
--------------

.. currentmodule:: torchgeo_bench.main
.. autofunction:: evaluate_logistic

.. currentmodule:: torchgeo_bench.linear
.. autoclass:: LogisticRegression

Segmentation
------------

.. currentmodule:: torchgeo_bench.main
.. autofunction:: evaluate_segmentation
.. autoclass:: SegmentationOutcome

.. currentmodule:: torchgeo_bench.segmentation_probe
.. autoclass:: SegmentationProbe

.. currentmodule:: torchgeo_bench.segmentation_task
.. autoclass:: SegmentationSolver
.. autoclass:: SegMetrics
.. autoclass:: EpochRecord

.. currentmodule:: torchgeo_bench.epoch_metrics
.. autoclass:: EpochMetricsWriter

Intrinsic dimension
-------------------

See :doc:`intrinsic_dim` for the standalone module API; the orchestration
function lives in :mod:`torchgeo_bench.main`:

.. currentmodule:: torchgeo_bench.main
.. autofunction:: evaluate_intrinsic_dim

Result I/O
----------

.. autofunction:: append_rows_atomic
