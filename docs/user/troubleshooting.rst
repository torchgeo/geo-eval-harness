Troubleshooting
===============

Missing dataset files
---------------------

Run benchmarks from a directory with the expected ``data/`` layout:

* V1: :file:`data/classification_v1.0_wds/<name>/`
* V2: :file:`data/geobenchv2/<name>/`
* EuroSAT: :file:`data/eurosat/`

Use ``geo-eval-harness datasets <name>`` to inspect a dataset and ``geo-eval-harness download <name>`` to download it. If data lives elsewhere, symlink ``data/`` to that location. See :doc:`datasets` for all supported paths.

CUDA out of memory
------------------

Reduce the loader batch size or use CPU:

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat --batch-size 32
   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu

For segmentation, ``runtime.batch_size`` controls image loading and uncached training; ``segmentation.batch_size`` controls training on cached features. Reduce the setting for the path you use.

If the feature cache exceeds system RAM, disable it in a file passed through ``--config``:

.. code-block:: yaml

   model: {name: timm/resnet18}
   datasets: [caffe]
   segmentation:
     cache_features: false

GPU startup errors
------------------

Image runs default to ``runtime.device: cuda:0``. An explicit CUDA request fails when CUDA is unavailable or the device index is invalid. Use ``--device auto`` to select CUDA when available and CPU otherwise, or ``--device cpu`` to require CPU.

``RuntimeError: The NVIDIA driver on your system is too old`` means the installed PyTorch build requires a newer driver. ``CUDA error: no kernel image is available for execution on the device`` means that build lacks kernels for the GPU architecture. Install a PyTorch build supported by both the GPU and driver, keeping the project's dependency requirements satisfied. ``uv sync --frozen`` installs the versions in :file:`uv.lock`.

Missing EuroSAT spatial splits
------------------------------

``geo-eval-harness download eurosat`` downloads the shared EuroSAT imagery and both standard and spatial train/val/test split files. Rerun the command if ``eurosat-spatial`` reports missing files.

Missing V1 JSON metadata
------------------------

V1 tar shards require ``.meta.json`` metadata; custom HDF5 samples require the ``metadata_json`` attribute. Download a verified subset with:

.. code-block:: console

   $ geo-eval-harness download geobench_v1 --datasets m-eurosat

Omit ``--datasets`` to download the full V1 collection. Downloads use :file:`data/classification_v1.0_wds/`. See :doc:`datasets` for the JSON schema.

Documentation network warnings
------------------------------

Sphinx's ``WARNING: failed to reach any of the inventories`` indicates that an external API inventory could not be downloaded. Check network access and retry the build before treating its cross-references as verified.
