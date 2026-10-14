Command-line interface and configuration
========================================

.. module:: torchgeo_bench.cli

The installed ``torchgeo-bench`` command, ``python -m torchgeo_bench``, and
``python -m torchgeo_bench.cli`` share one explicit-flags/YAML interface:

``run``
    Evaluate image backbones with classification and segmentation probes.
    Select a preset with ``--model`` and repeat ``--dataset`` for multiple
    datasets, or supply ``--config``. ``--methods linear`` runs only the
    linear classifier. ``--dry-run`` prints reusable YAML;
    ``--config-help`` prints the JSON schema.

``models [name]`` / ``datasets [name]``
    List packaged presets or registered image datasets, or show one entry.
    Discovery does not construct models or load samples.

``download <name> [<name> ...]``
    Download named datasets. Collection aliases ``geobench_v1`` and
    ``geobench_v2`` accept ``--datasets a,b,c``. ``--output-dir`` changes
    the download destination, not the benchmark's fixed ``./data/`` paths.

``profile``
    Measure a fixed real dataset batch and emit JSON to stdout by default. Accepts ``--config``, input/runtime flags, warmup/measurement counts, precision, and optional FLOP counting. Specify ``--dataset`` at most once; one CLI value can override the dataset in YAML. ``--output-dir`` writes ``profile.json`` beneath the given directory; ``--output`` selects an exact filename.

``flops``
    Measure backbone/probe compute using synthetic inputs and append to
    a CSV. Accepts its own ``--config``, ``--config-help``, band/head
    selections, timing, output, and resume flags.

``coord``
    Evaluate location encoders under random/spatial or official splits.
    Accepts coordinate YAML or explicit model, dataset, method, split,
    runtime, and output flags.

All measurement commands share ``--output-dir`` for an output root and ``--output`` for an explicit file that takes precedence. CSV commands default to the ``results`` root; profiling defaults to stdout. Image runs keep ``models/``, ``profiles/``, and ``intrinsic_dim/`` subdirectories under the root unless an explicit file combines them.

All measurement commands support ``--dry-run``. See
:doc:`/user/configuration` for exact YAML fields and :doc:`/user/coordbench`
for coordinate examples. Legacy ``key=value`` / ``+key=value`` syntax is
rejected; it is not available through an alternate module entry point.

Strict image configuration
--------------------------

Load and validate a YAML file without importing model implementations:

.. code-block:: python

   from torchgeo_bench.config.run import load_run_config
   from torchgeo_bench.config.presets import resolve_run_config

   config = load_run_config("docs/examples/image-run.yaml")
   effective, preset = resolve_run_config(config, "m-eurosat")

``resolve_run_config`` applies model and dataset defaults beneath explicitly
supplied values. ``RunConfig.model_dump_yaml()`` preserves omission by
excluding unset fields; dumping all schema defaults and reloading them would
turn those defaults into explicit overrides.

.. currentmodule:: torchgeo_bench.config.run

.. autoclass:: RunConfig
   :members: model_dump_yaml
   :inherited-members: BaseModel
   :no-show-inheritance:

.. autofunction:: load_run_config
.. autofunction:: validate_run_config

.. currentmodule:: torchgeo_bench.config.schema

.. autoclass:: ModelConfig
   :members:
   :no-show-inheritance:

.. autoclass:: SegmentationConfig
   :members:
   :no-show-inheritance:

.. autoclass:: EarlyStoppingConfig
   :members:
   :no-show-inheritance:

.. autofunction:: load_yaml

Preset resolution and construction
----------------------------------

Preset metadata is separate from constructor kwargs. Runtime objects such
as ``BandSpec`` instances or an empirical-RCF training dataset must be
passed explicitly to ``build_model``. The normal image runner handles
band selection, normalization-name mapping, and per-dataset construction.
``kwargs`` are ordinary constructor values, not recursively instantiated
target mappings.

.. currentmodule:: torchgeo_bench.config.presets

.. autoclass:: ModelPreset
   :members: for_dataset
   :no-show-inheritance:

.. autofunction:: load_model_preset
.. autofunction:: resolve_run_config
.. autofunction:: build_model

Benchmark entry point
---------------------

The image loop consumes a ``RunConfig`` directly:

.. code-block:: python

   from torchgeo_bench.config.run import load_run_config
   from torchgeo_bench.main import main

   config = load_run_config("docs/examples/image-run.yaml")
   main(config)

Unlike loading or resolving configuration, calling ``main`` executes the
benchmark and may load data or weights and append results.

.. currentmodule:: torchgeo_bench.main
.. autofunction:: main

Standalone measurement settings
--------------------------------

Standalone profiling uses singular ``dataset`` plus top-level timing fields;
synthetic compute measurements use band-source metadata, head selections,
and a ``timing`` block. Neither takes an image ``RunConfig``.

.. currentmodule:: torchgeo_bench.config.profile

.. autoclass:: ProfileConfig
   :members: model_dump_yaml
   :inherited-members: BaseModel
   :no-show-inheritance:

.. autofunction:: resolve_profile_config

.. currentmodule:: torchgeo_bench.config.flops

.. autoclass:: FlopsConfig
   :members: model_dump_yaml, resolve
   :inherited-members: BaseModel
   :no-show-inheritance:

.. currentmodule:: torchgeo_bench.flops_pipeline
.. autofunction:: main

See :doc:`coordbench` for the separate coordinate configuration and runner.

Download helpers
----------------

.. currentmodule:: torchgeo_bench.download
.. autofunction:: download_geobench_v1
.. autofunction:: download_geobench_v2
.. autofunction:: download_eurosat
