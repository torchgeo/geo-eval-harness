Installation
============

``torchgeo-bench`` targets Python 3.12 or newer.  We recommend installing
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

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ uv sync --extra dev

The install pulls in ``faissknn[cuda]`` (GPU FAISS) on Linux x86_64 and
``faissknn[cpu]`` everywhere else — no extra needed.

The ``torchgeo-bench`` console script is then available via ``uv run``. Use ``--help`` to list commands or ``--version`` to print the installed version:

.. code-block:: console

   $ uv run torchgeo-bench --help
   $ uv run torchgeo-bench --version

pip
---

If you prefer ``pip`` (e.g. inside a conda environment), install the
project in editable mode:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
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
``cuda``          ``faissknn[cuda]`` → ``faiss-cuda-cu128`` for GPU KNN (Linux only; shares the ``faiss`` namespace with the default ``faissknn[cpu]`` — install in a fresh env)
``dev``           ruff, pytest, pytest-cov, pytest-xdist, pre-commit, mdformat, pyproject-fmt
``docs``          sphinx, pydata-sphinx-theme, myst-parser, sphinx-copybutton, sphinx-design
``id``            ``torchid`` for intrinsic-dimension metrics (Python ≥ 3.13 only)
``lora``          ``peft`` for LoRA segmentation fine-tuning (also included in ``dev``)
``olmoearth``     ``olmoearth-pretrain-minimal`` for the OlmoEarth backbone
``terratorch``    ``terratorch`` for TerraTorch backbones
``all``           every extra above **except** ``cuda`` (FAISS conflict)
================  ==============================================================

Datasets are not installed by these extras — see :doc:`datasets` for how to
download GeoBench V1 / V2 with the bundled ``torchgeo-bench download`` command.
CoordBench tables stream from Hugging Face through ``torchgeo-bench coord``; see
:doc:`coordbench`.
