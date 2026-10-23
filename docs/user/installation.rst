Installation
============

``geo-eval-harness`` targets Python 3.12 or newer.  We recommend installing
inside a fresh virtual environment using `uv`_, which is the project's
canonical workflow.

.. _uv: https://docs.astral.sh/uv/

.. note::

   The install runs on **Linux**, **macOS**, and **Windows**. On Linux
   x86_64 it automatically includes GPU-accelerated FAISS KNN (CUDA 12,
   driver R525+); other platforms get CPU FAISS.

uv (recommended)
----------------

Clone the repository and let ``uv`` create the environment and install all
development dependencies:

.. code-block:: console

   $ git clone https://github.com/torchgeo/geo-eval-harness.git
   $ cd geo-eval-harness
   $ uv sync --extra dev

The install pulls in ``faissknn[cuda]`` (GPU FAISS) on Linux x86_64 and
``faissknn[cpu]`` everywhere else — no extra needed.

The ``geo-eval-harness`` console script is then available via ``uv run``. Use ``--help`` to list commands or ``--version`` to print the installed version:

.. code-block:: console

   $ uv run geo-eval-harness --help
   $ uv run geo-eval-harness --version

pip
---

If you prefer ``pip`` (e.g. inside a conda environment), install the
project in editable mode:

.. code-block:: console

   $ git clone https://github.com/torchgeo/geo-eval-harness.git
   $ cd geo-eval-harness
   $ pip install -e ".[dev]"

Optional extras
---------------

A number of dependencies are gated behind PEP 621 extras so that lean
benchmarking installations do not pull in heavyweight libraries by
default.  Combine extras with comma-separated lists:

.. code-block:: console

   $ uv sync --extra docs --extra dev
   $ # or
   $ pip install -e ".[docs,dev]"

.. note::

   ``uv sync`` installs *exactly* the extras you pass and removes anything not
   listed, so extras do **not** accumulate across calls.  Request them together
   in one command (``uv sync --extra docs --extra dev``) rather than running
   ``uv sync --extra docs`` and then ``uv sync --extra dev`` (the second drops
   the first).

================  ==============================================================
Extra             Pulls in
================  ==============================================================
``coordbench``    ``rshf`` for the SatCLIP, GeoCLIP, Climplicit, and SINR location encoders
``dev``           ruff, pytest, pytest-cov, pytest-xdist, pre-commit, mdformat, pyproject-fmt
``docs``          sphinx, pydata-sphinx-theme, myst-parser, sphinx-copybutton, sphinx-design
``id``            ``torchid`` for intrinsic-dimension metrics (Python ≥ 3.13 only)
``lora``          ``peft`` for LoRA segmentation fine-tuning (also included in ``dev``)
``olmoearth``     ``olmoearth-pretrain-minimal`` for the OlmoEarth backbone
``all``           every extra above
================  ==============================================================

Datasets are not installed by these extras — see :doc:`datasets` for how to
download GeoBench V1 / V2 with the bundled ``geo-eval-harness download`` command.
CoordBench tables stream from Hugging Face through ``geo-eval-harness coord``; see
:doc:`coordbench`.

TerraTorch models
-----------------

TerraTorch's PyPI release 1.2.13 requires ``torchgeo<0.10``, while this package requires ``torchgeo>=0.10,<0.11``. TerraTorch is installed separately from a compatible source revision and is not part of ``[all]``:

.. code-block:: console

   $ uv pip install "terratorch @ git+https://github.com/torchgeo/terratorch.git@89a3ed13d7a6272c009c4c5fbabf62461fcfb16c"

Run these models with ``uv run --no-sync geo-eval-harness ...`` so project synchronization does not remove the separately installed dependency. With pip, use the same dependency specification after ``pip install geo-eval-harness``.

Package name
------------

Install and invoke ``geo-eval-harness``; Python imports and module entry points use ``geo_eval_harness``. Update imports and fully qualified ``model.target`` values in custom YAML files to that namespace. Stored CSVs retain their recorded model paths; resume and result comparisons recognize equivalent paths from ``torchgeo_bench`` without changing the recorded measurements or configuration hashes.
