geo_eval_harness.models
=======================

.. module:: geo_eval_harness.models

This module provides the abstract :class:`BenchModel` interface and a
collection of concrete backbones that can be benchmarked across the
:mod:`geo_eval_harness.datasets` registry.

Model construction is exposed through
:func:`~geo_eval_harness.config.presets.load_model_preset` and
:func:`~geo_eval_harness.config.presets.build_model`. Preset ``kwargs`` contain
constructor options; the runner supplies selected ``BandSpec`` objects
and normalization explicitly. See :doc:`cli` for typed preset resolution.

Interface
---------

.. autoclass:: BenchModel

Backbones
---------

Random Convolutional Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RCFBench
.. autoclass:: RCFModelSettings

Image statistics baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ImageStatsBench

Handcrafted baseline
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: HandcraftedBench
   :members: feature_names, num_features, feature_metadata

timm encoders
^^^^^^^^^^^^^

.. autoclass:: TimmPatchBenchModel
.. autoclass:: TimmModelSettings

torchgeo encoders
^^^^^^^^^^^^^^^^^

.. autoclass:: TorchGeoResNetBench
.. autoclass:: TorchGeoSwinBench
.. autoclass:: TorchGeoScaleMAEBench
.. autoclass:: TorchGeoDOFABench
.. autoclass:: TorchGeoEarthLocBench

OlmoEarth
^^^^^^^^^

.. autoclass:: OlmoEarthBenchModel

SAM 3
^^^^^

.. autoclass:: SAM3Encoder

Segmentation heads
------------------

These heads attach to a frozen :class:`BenchModel` backbone to produce
dense per-pixel predictions.  See :class:`~geo_eval_harness.segmentation_probe.SegmentationProbe`
for the wiring layer, and :doc:`/user/segmentation-layers` for the
verified ``segmentation.layers`` values for each supported timm
backbone family.

.. autoclass:: LinearHead
.. autoclass:: ConvBlockHead
.. autoclass:: FPNHead
.. autoclass:: DPTHead
.. autoclass:: UPerNetHead
