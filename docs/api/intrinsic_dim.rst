geo_eval_harness.intrinsic_dim
==============================

.. module:: geo_eval_harness.intrinsic_dim

Wrapper around the `torchid <https://github.com/jacobpennington/torchid>`__
intrinsic-dimension estimators plus dependency-free centered feature-spectrum
diagnostics.  Used by
:func:`geo_eval_harness.main.evaluate_intrinsic_dim` to attach
``method="intrinsic_dim"`` rows to
``output.directory/intrinsic_dim/<model name>.csv``. An explicit
``output.file`` combines them with the image-run metrics instead.

Enable the pass with ``intrinsic_dim.enabled: true`` in image YAML.
``intrinsic_dim.estimators: []`` requests only feature-spectrum diagnostics.

The ``torchid`` dependency is optional and gated behind the ``[id]`` extra
in :file:`pyproject.toml` (it requires Python ≥ 3.13).  When the dependency
is not installed an :class:`ImportError` is raised on first use.

Public API
----------

.. autoexception:: DegenerateSpectrumError

.. autofunction:: compute_intrinsic_dim

.. autofunction:: compute_feature_spectrum
