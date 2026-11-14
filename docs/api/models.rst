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
.. autoclass:: InputUnit
.. autoclass:: NormalizationStrategy

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
.. autoclass:: TorchGeoCromaBench
.. autoclass:: TorchGeoDEOBench
.. autoclass:: TorchGeoPanopticonBench

OlmoEarth
^^^^^^^^^

.. autoclass:: OlmoEarthBenchModel

AnySat
^^^^^^

.. autoclass:: AnySatBenchModel

THOR
^^^^

.. autoclass:: ThorBenchModel

SMARTIES
^^^^^^^^

.. autoclass:: SmartiesBench

EoS-FM
^^^^^^

.. autoclass:: EoSFMBenchModel

SAM 3
^^^^^

.. autoclass:: SAM3Encoder

Tessera V2
^^^^^^^^^^

.. autoclass:: TesseraBenchModel
   :members: forward_batch

TerraFM
^^^^^^^

.. autoclass:: TerraFMBench

TerraTorch
^^^^^^^^^^

.. autoclass:: TerraTorchClayBench
   :no-show-inheritance:
.. autoclass:: TerraTorchPrithviBench
   :no-show-inheritance:
.. autoclass:: TerraTorchTerraMindBench
   :no-show-inheritance:

UniverSat
^^^^^^^^^

.. autoclass:: UniverSatBenchModel

Segmentation heads
------------------

These heads attach to a :class:`BenchModel` backbone to produce dense per-pixel predictions with frozen features, LoRA, or full fine-tuning.  See :class:`~geo_eval_harness.segmentation_probe.SegmentationProbe`
for feature extraction and head execution, and :doc:`/user/segmentation-layers` for the
verified ``segmentation.layers`` values for each supported timm
backbone family.

.. autoclass:: LinearHead
.. autoclass:: PatchLinearHead
.. autoclass:: ConvBlockHead
.. autoclass:: FPNHead
.. autoclass:: DPTHead
.. autoclass:: UPerNetHead
.. autoclass:: UNetHead
.. autoclass:: ViTNeck
.. autoclass:: NeckedHead
.. autoclass:: geo_eval_harness.models.anyup.AnyUpClassifier
