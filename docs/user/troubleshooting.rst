Troubleshooting
===============

``Dataset directory not found`` / files missing
-----------------------------------------------

Datasets must live under ``./data/`` from the directory where you run
``geo-eval-harness``.  The runner does **not** honour ``GEOBENCH_ROOT``
or ``GEOBENCH_V2_ROOT`` environment variables — paths are fixed:

* V1: :file:`data/classification_v1.0/<name>/`
* V2: :file:`data/geobenchv2/<name>/`
* EuroSAT: :file:`data/eurosat/`

Re-run ``geo-eval-harness download …`` to fetch missing data.  If your
data lives elsewhere, symlink ``data/`` to the real location.

``ModuleNotFoundError: geobench``
---------------------------------

The legacy ``geobench`` package is no longer a dependency.  V1 datasets
are read directly from HDF5 (the internal ``GeoBenchv1`` loader in
:file:`src/geo_eval_harness/datasets/geobench_v1.py`); V2 dispatches to
upstream ``geobench_v2.datasets.GeoBench<X>``.  Make sure your
environment matches the pinned ``geobenchv2`` version in
:file:`pyproject.toml`.

CUDA out of memory
------------------

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat --batch-size 32
   $ # or run on CPU
   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu

For segmentation, also try

.. code-block:: yaml

   model: {name: timm/resnet18}
   datasets: [caffe]
   segmentation:
     cache_features: false

in a file passed through ``--config`` if RAM (rather than GPU memory) is
the bottleneck.

GPU run crashes immediately
---------------------------

The image default is ``runtime.device: cuda:0``, so an unmodified image run uses the
GPU.  ``uv sync`` installs the latest ``torch``, whose bundled CUDA and kernel
architectures may not match your GPU or driver.  Two distinct failures:

* ``RuntimeError: The NVIDIA driver on your system is too old`` — the installed
  ``torch`` was built against a newer CUDA than your driver supports.
* ``CUDA error: no kernel image is available for execution on the device``
  (``cudaErrorNoKernelImageForDevice``) — torch's CUDA 12.8 wheels dropped
  Volta (``sm_70``) kernels, so they fail on a V100 *even though* CUDA
  initialises.  Older GPUs need a **cu126** (or earlier) build, e.g.
  ``torch==2.7.1+cu126`` + ``torchvision==0.22.1+cu126`` from
  ``https://download.pytorch.org``.

Either way you can fall back to CPU (slower, but always works):

.. code-block:: console

   $ geo-eval-harness run --model rcf --dataset m-eurosat --device cpu

CPU is fine for the small V1 splits, but large V2 datasets (e.g. ``benv2`` /
BigEarthNet) can take far longer — prefer a working GPU for those.

``KeyError: 's2'`` on a V2 dataset
----------------------------------

A known V2 issue: ``geobench_v2.rearrange_bands`` expects modality keys
(``'s2'``, ``'s1'``, …) that aren't present when a flat band list is
requested.  Workaround: use ``--bands all`` for affected V2
datasets.

``eurosat-spatial`` reports ``Dataset not found``
-------------------------------------------------

``geo-eval-harness download eurosat`` fetches the shared EuroSAT images and
both the standard and spatial train/val/test splits. If an older download
is missing the spatial split files, rerun this command before benchmarking
``eurosat-spatial``.

V1 reports missing JSON metadata
--------------------------------

Old V1 downloads contain ``.meta.pkl`` shard members or HDF5 ``pickle`` attributes. These are no longer loaded. Replace the cached data with the pickle-free mirror:

.. code-block:: console

   $ geo-eval-harness download geobench_v1 --datasets m-eurosat

Omit ``--datasets`` to replace the full V1 suite. Downloads and V1 slow tests both use ``data/classification_v1.0_wds/``. Tests skip absent datasets, but present legacy or malformed data fails visibly.

Build / docs warnings
---------------------

If you build the docs locally without internet access, expect ~9
``WARNING: failed to reach any of the inventories`` messages from
``sphinx.ext.intersphinx``.  These are network reachability errors, not
real issues — the GitHub Pages build runner has network access and
resolves these inventories cleanly.
