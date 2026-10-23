Evaluation
==========

The evaluation pipeline lives in :mod:`geo_eval_harness.main` and a few
focused sub-modules.  Each evaluation method (KNN-5, linear probe,
segmentation, intrinsic dimension) consumes per-split feature embeddings
or raw images and produces one :class:`EvaluationResult` row per metric.
Image orchestration accepts
:class:`~geo_eval_harness.config.run.RunConfig`; see :doc:`cli` for loading,
preset resolution, and the separate standalone measurement schemas.

Result schema
-------------

.. currentmodule:: geo_eval_harness.main

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

.. currentmodule:: geo_eval_harness.knn
.. autoclass:: KNNClassifier

Linear probing
--------------

.. currentmodule:: geo_eval_harness.main
.. autofunction:: evaluate_logistic

.. currentmodule:: geo_eval_harness.linear
.. autoclass:: LogisticRegression

Segmentation
------------

.. currentmodule:: geo_eval_harness.main
.. autofunction:: evaluate_segmentation

.. currentmodule:: geo_eval_harness.segmentation_probe
.. autoclass:: SegmentationProbe

.. currentmodule:: geo_eval_harness.segmentation_task
.. autoclass:: SegmentationSolver
.. autoclass:: SegMetrics

Intrinsic dimension
-------------------

See :doc:`intrinsic_dim` for the standalone module API; the orchestration
function lives in :mod:`geo_eval_harness.main`:

.. currentmodule:: geo_eval_harness.main
.. autofunction:: evaluate_intrinsic_dim

Result I/O
----------

.. autofunction:: append_rows_atomic
