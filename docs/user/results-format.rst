Results format
==============

All evaluation runs append rows to per-model CSV files.  Each row is a
flattened :class:`~torchgeo_bench.main.EvaluationResult` describing a single
``(dataset, method, model, config)`` measurement.  Rows are split by *kind*
across three directories, keyed by ``<model name>.csv``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Directory
     - Row ``method``\ s
     - Contents
   * - ``results/models/``
     - ``knn5``, ``linear``, ``seg-*``
     - Classification/segmentation metrics -- appended on each metrics run.
   * - ``results/profiles/``
     - ``profile``
     - Throughput/latency/param-count measurements -- one-time per model+hardware.
   * - ``results/intrinsic_dim/``
     - ``intrinsic_dim``
     - Intrinsic-dimension estimates -- one-time per model.

Profile and intrinsic-dim rows are one-time model+hardware measurements, so
they are kept out of the metrics file: rerunning a classification sweep
(new dataset, fixed metric, etc.) only touches ``results/models/``, instead
of also rewriting/diffing the expensive one-off profile and intrinsic-dim
rows every time.

Sample rows
-----------

.. code-block:: text

   dataset,method,metric_name,metric_value,ci_lower,ci_upper,feature_dim,best_c,n_train,n_val,n_test,seed,model,name,normalization,image_size,interpolation,partition,bands,num_classes
   m-eurosat,knn5,accuracy,0.8234,0.8123,0.8345,512,,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,bandspec_zscore,224,bilinear,default,rgb,10
   m-eurosat,linear,accuracy,0.8567,0.8461,0.8673,512,0.1,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,bandspec_zscore,224,bilinear,default,rgb,10
   burn_scars,seg-fpn,mIoU,0.6234,0.0,0.0,768,,1000,200,300,0,torchgeo_bench.models.TimmPatchBenchModel,resnet50,bandspec_zscore,224,bilinear,default,rgb,3

Datasets emit unnormalized tensors; each model wrapper normalises inside
:meth:`~torchgeo_bench.models.BenchModel.normalize_inputs` according to
the strategy selected by ``input.normalization``. The CSV retains the internal
names below: ``dataset`` maps to ``bandspec_zscore``, ``model`` to ``model_native``,
and ``none`` to ``identity``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Behaviour
   * - ``bandspec_zscore``
     - Per-channel z-score using ``BandSpec`` mean/std (default).
   * - ``model_native``
     - Convert to the wrapper's ``expected_input_unit``, then apply any
       ``pretrain_mean`` / ``pretrain_std`` declared on the class.
   * - ``minmax``
     - Scale each channel to ``[0, 1]`` from BandSpec min/max.
   * - ``minmax_zscore``
     - ``minmax`` then z-score against assumed ``mean=0.5, std=0.25``.
   * - ``identity``
     - No rescaling (for models whose forward owns normalisation).

Older snapshots may carry legacy values such as ``raw`` / ``mean_stdev`` /
``percentile_2_98`` — they are kept verbatim for resume safety.

Method values
-------------

================== ==================================================================================
``method``         Meaning
================== ==================================================================================
``knn5``           KNN-5 classification (multilabel KNN for ``m-bigearthnet``). -> ``results/models/``
``linear``         L-BFGS logistic regression with C-sweep on the validation set. -> ``results/models/``
``intrinsic_dim``  Optional intrinsic-dimension metrics on extracted embeddings (requires
                   the ``[id]`` extra) plus dependency-free centered feature-spectrum
                   diagnostics. Both are emitted when
                   YAML ``intrinsic_dim: {enabled: true}``; set ``estimators: []`` for
                   spectrum-only output without ``torchid``. -> ``results/intrinsic_dim/``
``profile``        Optional throughput/latency/param-count measurement (requires
                   YAML ``profile: {enabled: true}``). -> ``results/profiles/``
``seg-<head>``     Segmentation probe with the configured head (``linear`` / ``conv_block`` /
                   ``fpn`` / ``dpt`` / ``patch_linear``). -> ``results/models/``
================== ==================================================================================

CSV schema
----------

==================== ============================================================
Column               Description
==================== ============================================================
``dataset``          Dataset CLI name (e.g. ``m-eurosat``).
``method``           ``knn5``, ``linear``, ``intrinsic_dim``, or ``seg-<head_type>``.
``metric_name``      Primary metric (``accuracy``, ``micro_mAP``, ``mIoU``,
                     ``id_<estimator>_<split>`` for intrinsic-dimension rows,
                     or ``spectrum_<metric>_<split>`` for feature-spectrum rows).
``metric_value``     Point estimate.
``ci_lower``         Bootstrap CI lower bound (0.0 when not applicable).
``ci_upper``         Bootstrap CI upper bound (0.0 when not applicable).
``feature_dim``      Embedding dimension produced by the backbone.
``best_c``           Best ``C`` from the logistic-regression sweep
                     (linear probe only, otherwise ``None``).
``best_lr``          Best learning rate (segmentation only).
``best_batch_size``  Best batch size (segmentation only).
``n_train``          Train-split sample count.
``n_val``            Validation-split sample count.
``n_test``           Test-split sample count.
``seed``             RNG seed used for the run.
``model``            Fully-qualified class from the resolved preset's ``target``.
``name``             Human-readable model name from the resolved preset.
``normalization``    Strategy applied by the model wrapper (see table above).
``image_size``       Input resize size (``None`` if no resizing).
``interpolation``    Resize interpolation mode.
``partition``        GeoBench V1 partition name (``default`` for V2).
``bands``            ``rgb`` / ``all`` / a sorted comma-joined list.
``num_classes``      Dataset label count. It is also part of the resume key so
                     label-schema changes cannot reuse stale rows.
``c_range_start``    ``classification.linear.c_log10_start``.
``c_range_stop``     ``classification.linear.c_log10_stop``.
``c_range_num``      ``classification.linear.c_count``.
``merge_val``        Whether ``train+val`` was merged before final logistic fit.
``bootstrap``        Number of bootstrap resamples used for CIs.
``fw_iou``           Frequency-weighted IoU (segmentation only).
``precision``        Macro precision (segmentation only).
``recall``           Macro recall (segmentation only).
``f1``               Macro F1 (segmentation only).
==================== ============================================================

Segmentation rows also include ``boundary_iou`` (mean Boundary IoU) and ``boundary_iou_width`` (the boundary tolerance in pixels, rounded from 2% of the image diagonal).

Atomic appends
--------------

Rows are appended via :func:`~torchgeo_bench.main.append_rows_atomic`,
which uses ``fcntl`` advisory file locking (available on Linux and
macOS).  This makes it safe to point multiple parallel jobs (e.g. one
per GPU or per dataset) at the same output file without corrupting it.

Resume mode
-----------

With ``--resume`` or YAML ``output: {resume: true}``, the runner reads existing CSVs and
skips any combination that already has a matching row.  Since profile and
intrinsic-dim rows may live in their own files (see above), resume reads
all three files -- ``results/models/<name>.csv``,
``results/profiles/<name>.csv``, and ``results/intrinsic_dim/<name>.csv``
-- and unions their completed-metric keys.  The de-dup key is:

.. code-block:: python

   (dataset, method, model, name, normalization, image_size, interpolation,
    partition, bands, num_classes, res, pool, config_hash)

Resume checks each method separately, so ``--methods linear`` never requires a KNN row. Profile and intrinsic-dimension passes do not change ``config_hash``; each requested metric must be present before a pass is skipped.

Resume requires an exact ``config_hash`` match. The hash covers the resolved model, preprocessing, and evaluation settings plus ``runtime.seed`` and ``runtime.batch_size``, since batch size can change floating-point results. It excludes ``runtime.device`` and ``runtime.workers``, so completed results and profile measurements are reused on another device or worker count; use ``--no-resume`` or a separate output file to remeasure performance.

Segmentation hashes also include the best-validation-mIoU selection protocol, determined from the resolved dataset task. A final-epoch result cannot satisfy a best-validation run's resume key. Classification hashes are unaffected by this protocol discriminator.

Method names and CSV columns are unchanged. The selection policy is defined in :ref:`segmentation-model-selection`.

Existing results are not relabeled or rewritten. Affected segmentation configurations need new runs to obtain best-validation results.

Rows written before version 0.5.0 do not have ``num_classes`` and are treated
as incomplete by resume mode. The checked-in SpaceNet2/7 rows produced under
the old three-class task were removed; they remain available from the 0.4.0 tag.

CoordBench uses a separate schema and output path. See :doc:`coordbench` and
:class:`~torchgeo_bench.coordbench.CoordResult`.
