Contribute a model (Stage 2)
============================

This guide covers everything a pull request needs to add a new frozen
pretrained model to ``geo-eval-harness``.  Before continuing, complete
:doc:`eval_own_model` which demonstrates how to implement your new model to work with the geo-eval-harness pipeline and verify your model produces sensible results
on the applicable datasets.

.. _contrib-prerequisites:

Prerequisites
-------------

The setup steps are the same as Stage 1:

.. code-block:: console

   $ git clone https://github.com/torchgeo/geo-eval-harness.git
   $ cd geo-eval-harness
   $ uv sync --extra dev

Then fork the repository on GitHub and create a feature branch:

.. code-block:: console

   $ git remote add fork https://github.com/<your-username>/geo-eval-harness.git
   $ git checkout -b add-<model-name>

For the model implementation and YAML config steps, follow
:doc:`eval_own_model` (Sections :ref:`eval-implement` and
:ref:`eval-model-config`).  The remainder of this page covers integration,
tests, and the PR submission.

.. _contrib-integrate:

Integrate into the package
--------------------------

Once your model class is working locally, move it into geo-eval-harness and expose its import path to the model preset loader.

**1. Place the module** under :file:`src/geo_eval_harness/models/`:

.. code-block:: console

   $ mv new_model.py src/geo_eval_harness/models/new_model.py

**2. Export the class** from :file:`src/geo_eval_harness/models/__init__.py`:

.. code-block:: python

   from .new_model import NewModel

   __all__: list[str] = [
       # Keep the existing entries in alphabetical order.
       "NewModel",
   ]

**3. Update the model preset** ``target`` to the package path and put constructor options under ``kwargs``:

.. code-block:: yaml

   name: new_model
   target: geo_eval_harness.models.NewModel
   track: image
   kwargs:
     pretrained: true

**4. Declare optional dependencies** in :file:`pyproject.toml` if your model
requires packages beyond ``[project.dependencies]``:

.. code-block:: toml

   [project.optional-dependencies]
   newmodel = ["newpackage>=1.0"]

Install the extra locally to confirm it resolves:

.. code-block:: console

   $ uv sync --extra dev --extra <newmodel>
   $ uv lock

.. note::

   Document required extras in the constructor docstring and PR description. Import optional packages inside the constructor or weight-loading method, not at module scope, so the model class remains importable without the extra. Do not add fallback imports for hard dependencies or silently use random weights when pretrained loading fails.

.. _contrib-weights:

Weights
-------

* Pretrained weights must be **publicly accessible without authentication**.
  `HuggingFace Hub <https://huggingface.co/models>`_ is the preferred host.
* The model must load after a fresh ``pip install 'geo-eval-harness[<newextra>]'``
  with no manual file placement.  Use
  `huggingface_hub.hf_hub_download <https://huggingface.co/docs/huggingface_hub/>`_
  or an equivalent auto-download call inside your ``__init__``.
* The weights URL must appear in the PR description so reviewers can verify
  provenance.

.. _contrib-tests:

Write tests
-----------

Create :file:`tests/test_<model>.py`.  Every added code path must be covered.

**Fast tests (run in CI) — no network I/O:**

.. code-block:: python

   import pytest
   import torch

   from geo_eval_harness.datasets.base import BandSpec
   from geo_eval_harness.models.new_model import NewModel


   def _bands(n: int = 3) -> list[BandSpec]:
       return [
           BandSpec(sensor="s2", name=f"b{i}", source_name=f"B{i}",
                    mean=500.0, std=100.0, min=0.0, max=10000.0)
           for i in range(n)
       ]


   def test_new_model_output_shape() -> None:
       """Model returns (B, K) with random weights."""
       model = NewModel(bands=_bands(), pretrained=False).eval()
       x = torch.randn(2, 3, 64, 64)
       with torch.no_grad():
           out = model.forward_patch_features(x)
       assert out.ndim == 2
       assert out.shape[0] == 2


   def test_new_model_num_channels() -> None:
       """`num_channels` matches the input BandSpec list length."""
       model = NewModel(bands=_bands(5), pretrained=False)
       assert model.num_channels == 5

**Weight-download tests (slow, run locally before PR) — mark with** ``@pytest.mark.slow``:

Replace the template's Identity backbone and implement weight loading before using this test as evidence that a pretrained model works. The unmodified scaffold loads no weights, even when ``pretrained=True``.

.. code-block:: python

   @pytest.mark.slow
   def test_new_model_pretrained_loads() -> None:
       """Pretrained weights download and load without error."""
       model = NewModel(bands=_bands(), pretrained=True).eval()
       x = torch.randn(1, 3, 64, 64)
       with torch.no_grad():
           out = model.forward_patch_features(x)
       assert out.ndim == 2

Run the fast tests before opening the PR:

.. code-block:: console

   $ uv run pytest --no-cov tests/test_new_model.py

Also cover the normalization strategies your wrapper supports, unsupported inputs, and construction through its packaged preset. For a segmentation-capable backbone, exercise its configured feature layers with a compatible head. Fast tests must not download weights or datasets.

Slow tests must pass locally but are excluded from the default CI run
(``pytest`` without ``-m slow`` skips them automatically):

.. code-block:: console

   $ uv run pytest --no-cov -m slow tests/test_new_model.py

.. _contrib-results:

Submit results
--------------

Run the benchmark on datasets applicable to your model's sensor coverage. The default output is :file:`results/models/<model_name>.csv`, matching the model PR template. For example:

.. code-block:: console

   $ uv run geo-eval-harness run --model new_model --device cuda:0 \
       --dataset m-eurosat --dataset m-so2sat --dataset m-bigearthnet \
       --dataset m-brick-kiln --dataset m-forestnet --dataset m-pv4ger

For V2 datasets:

.. code-block:: console

   $ uv run geo-eval-harness run --model new_model --device cuda:0 \
       --dataset benv2 --dataset treesatai --dataset so2sat --dataset forestnet --resume

Use ``--device cpu`` without a CUDA device. Record the selected bands and normalization, and list unsupported datasets with a reason rather than silently skipping them. For a longer dataset list or segmentation options, submit the run YAML and invoke it with ``--config`` as shown in :doc:`eval_own_model`. Commit only the new model's result file.

The CSV schema is identical to the per-model results files — see
:doc:`results-format` for the full column reference.

.. _contrib-lint:

Lint and full test suite
------------------------

Before opening the PR, apply auto-fixes and verify the full test suite passes:

.. code-block:: console

   $ uv run ruff check . --fix && uv run ruff format .
   $ uv run pytest --no-cov

.. _contrib-pr:

Open the PR
-----------

When all checklist items are satisfied, open a pull request against ``main``
using the **"Add model"** template:

:file:`.github/PULL_REQUEST_TEMPLATE/add_model.md`

All checklist items must be checked before requesting a review.  The
template prompts you for:

* A model summary table (name, pretraining data, sensor coverage, weights URL,
  and paper/project page if available).
* Confirmation that each technical requirement is satisfied (class exported,
  config present, weights public, tests written and passing, results
  submitted, lint clean).

.. seealso::

   :doc:`eval_own_model` — Stage 1: implement and benchmark your model.
