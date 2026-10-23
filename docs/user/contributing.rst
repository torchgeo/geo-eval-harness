Contributing
============

We welcome contributions!  This page summarises the local development
workflow.  See `AGENTS.md <https://github.com/torchgeo/geo-eval-harness/blob/main/AGENTS.md>`__
for the longer-form guide aimed at AI coding agents (which captures the
same conventions).

Environment
-----------

The canonical workflow uses `uv <https://docs.astral.sh/uv/>`_, which
resolves dependencies from :file:`pyproject.toml` and manages the virtual
environment for you:

.. code-block:: console

   $ git clone https://github.com/torchgeo/geo-eval-harness.git
   $ cd geo-eval-harness
   $ uv sync --extra dev

If you prefer `conda <https://docs.conda.io>`_, install into that environment with pip. ``uv sync`` manages a separate :file:`.venv`:

.. code-block:: console

   $ conda create -n geo-eval-harness 'python>=3.12,<3.13'
   $ conda activate geo-eval-harness
   $ pip install -e ".[dev]"

Makefile shortcuts
------------------

The top-level :file:`Makefile` provides convenient wrappers around the
above:

=================== ===================================================
Target              What it does
=================== ===================================================
``make install``    Create / update the conda env and install ``[dev]``.
``make sync``       Alias for ``install``.
``make tests``      ``pytest`` with coverage and offline integrations.
``make lint``       ``pre-commit run --all-files``.
``make format``     ``ruff format`` then ``ruff check --fix --select I``.
``make docs``       Build HTML documentation into ``docs/_build/html``.
``make docs-clean`` Remove the ``docs/_build`` directory.
``make clean``      Removes coverage reports and the pytest cache.
=================== ===================================================

Linting and formatting
----------------------

We use `ruff <https://docs.astral.sh/ruff/>`_ for both linting and
formatting:

.. code-block:: console

   $ uv run ruff check .
   $ uv run ruff format .

The project's ruff configuration enables ``E``, ``W``, ``F``, ``I``,
``B``, ``C4``, ``UP``, ``ARG``, ``SIM``, and ``D`` (pydocstyle, Google
convention) checks.  Line length is 100.

Tests
-----

.. code-block:: console

   $ uv run pytest                                  # unit + offline integration tests, with coverage
   $ uv run pytest -m integration                   # toy-data workflows only
   $ uv run pytest -m slow                          # optional downloaded-data/weight tests
   $ uv run pytest -m accuracy_check                # optional model accuracy baselines
   $ uv run pytest tests/test_intrinsic_dim.py -v   # one file
   $ uv run pytest -k "m-eurosat" -v                # by keyword
   $ uv run pytest --no-cov                         # disable coverage for speed

All test cases live under :file:`tests/`. Shared inputs and subprocess helpers belong in :file:`tests/support/`, not in another test module. Use the same Ruff profile for every test.

The default suite needs no downloaded datasets or pretrained weights and can run on CPU; GPU-specific tests skip when CUDA is unavailable. The :file:`tests/integration/` suite creates small on-disk datasets with separate training, validation, and test samples. It exercises CLI parsing, data loading, feature extraction, fitting, output files, and resume behavior together. External download transport is replaced with local fixture data, and a temporary random-weight preset lets the public CLI exercise segmentation offline; numerical algorithms and result writers run normally. Integration tests must assert meaningful outputs, not just a zero exit status.

``pytest-cov`` measures both lines and branches across :mod:`geo_eval_harness`, including Python subprocesses. Every normal run displays missing coverage and writes :file:`coverage.xml`; CI saves that report as an artifact and uploads it to Codecov. Generate a navigable local report with ``uv run pytest --cov-report=html`` and open :file:`htmlcov/index.html`. Optional model packages remain in the coverage denominator even when they are not installed, so compare results using the same dependency extras.

Only the explicitly selected ``slow`` and ``accuracy_check`` suites need real data or cached weights. Missing datasets skip individually; present but malformed data must fail. They use the canonical subdirectories documented in :doc:`datasets`.

Accuracy checks default to CPU with a 600-second timeout per model/dataset combination. Override these explicitly for larger models, for example ``uv run pytest -m accuracy_check --accuracy-device cuda:0 --accuracy-timeout 3600``. The absolute accuracy tolerance remains 0.02 for both KNN and linear probes.

Refresh :file:`tests/fixtures/accuracy_baselines.csv` with ``uv run python scripts/update_baselines.py``. The script reads stored results without running models or downloading weights. Its explicit case list preserves the small regression matrix; selection matches the resolved model target, normalization, bands, partition, seed, image settings, class count, resolution, pooling, and linear-probe settings before choosing the latest hashed result. Missing comparable results are an error. The fixture records the source hash for provenance, but does not require a rerun to share it: resume hashes also include bootstrap and batch-size settings. The default offline suite checks that the fixture still matches the stored results and current presets. Terratorch execution compatibility is tracked separately in `issue #397 <https://github.com/torchgeo/geo-eval-harness/issues/397>`__; refreshing its stored baselines does not verify that optional backend.

Code style
----------

* Python 3.12+ throughout. Use modern type hints (``list[str]``,
  ``X | None``) — do **not** import from ``typing.List`` / ``Optional`` /
  ``Union``.
* Avoid ``from __future__ import annotations``; prefer ``Self``, quoted
  annotations, or explicit imports for forward references.
* Google-style docstrings (configured via ``ruff.lint.pydocstyle.convention``).
* Use the ``logging`` module — no bare ``print`` calls.
* No defensive ``try/except ImportError`` for hard dependencies — every
  package in ``[project.dependencies]`` is guaranteed to be installed.

Documentation
-------------

This very site is built with Sphinx.  The quickest way to build it locally
is via the Makefile shortcut:

.. code-block:: console

   $ make docs
   $ open docs/_build/html/index.html

This assumes ``sphinx-build`` is on your ``PATH`` (install with
``uv sync --extra docs``).  To rebuild from scratch:

.. code-block:: console

   $ make docs-clean && make docs

Releasing to PyPI
-----------------

PyPI publishing uses the ``pypi`` GitHub environment and a
`Trusted Publisher <https://docs.pypi.org/trusted-publishers/>`_. Prepare every
release through a pull request:

Configure the ``geo-eval-harness`` project on PyPI with a Trusted Publisher for GitHub owner ``torchgeo``, repository ``geo-eval-harness``, workflow ``release.yml``, and environment ``pypi``. For the first publication, create a `pending publisher <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>`_. A publisher configured for another repository name does not authorize this repository; repository settings cannot configure the PyPI account.

1. Update :file:`docs/user/changelog.rst`, ``project.version`` in
   :file:`pyproject.toml`, the fallback ``__version__`` in
   :file:`src/geo_eval_harness/__init__.py`, and :file:`uv.lock`.
2. Run the release gate from a clean checkout:

   .. code-block:: console

      $ uv run pre-commit run --all-files
      $ uv run pytest
      $ uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
      $ uv build --no-sources
      $ uv run --no-project python scripts/check_distributions.py --tag vX.Y.Z
      $ uvx twine check --strict dist/*

3. Merge the release PR only after CI and the docs preview pass. Pull the merge
   commit onto ``main`` and confirm that the worktree is clean.
4. Create and push a tag that exactly matches ``project.version``:

   .. code-block:: console

      $ git tag vX.Y.Z
      $ git push origin vX.Y.Z

5. Create the matching GitHub release with curated notes (or generated notes
   reviewed against the changelog):

   .. code-block:: console

      $ gh release create vX.Y.Z --verify-tag --title vX.Y.Z --generate-notes

6. Watch the ``Publish to PyPI`` workflow, then verify the project page and a
   fresh install:

   .. code-block:: console

      $ gh run list --workflow release.yml
      $ uvx --from "geo-eval-harness==X.Y.Z" geo-eval-harness --help

The ``Publish to PyPI`` workflow (:file:`.github/workflows/release.yml`)
builds the source distribution and wheel in an unprivileged job, then uploads
them from a separate OIDC-authenticated job. The tag is the publication trigger;
do not move or reuse a published version tag.

CI installs the built wheel into a fresh environment and exercises the CLI outside the checkout. The release job checks that the tag matches ``project.version``, both archives contain the source package and its resources, and package metadata contains no direct URL dependencies.
