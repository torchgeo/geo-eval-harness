Results explorer
================

.. raw:: html

   <p>
     <a href="../_static/results-explorer.html" target="_blank" rel="noopener"
        style="display:inline-block;padding:10px 18px;background:#990F3D;
               color:#fff1e5;text-decoration:none;border-radius:2px;
               font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.04em;">
       Open the Results Explorer →
     </a>
   </p>

A self-contained HTML page for browsing a snapshot of ``geo-eval-harness``
benchmark results. The committed page is regenerated from
``results/models/*.csv``, ``results/profiles/*.csv``,
``results/intrinsic_dim/*.csv``, ``results/compute_cost.csv``, and the
archived JSON snapshots with
:file:`scripts/regen_results_explorer.py`.

It covers classification (``knn5`` / ``linear``), segmentation (``seg-<head>``
rows with their setup columns, reported as mIoU; see
:ref:`segmentation-setup-columns`), intrinsic dimension, and the cost/throughput
measurements. New snapshots require setup columns for segmentation rows. The segmentation method menus are
``seg-<head>`` split by ``adaptation`` (frozen, LoRA, full fine-tuning).  ``compute_cost.csv`` is a
wide table that the script melts into the long ``method="profile"`` rows the
page expects.

The HTML stores repeated rows and column values once, then expands them in the browser. The generator limits the HTML and snapshot files to 50 MB each.

The page is regenerated automatically: whenever a results CSV lands on
``main``, the ``Results explorer`` GitHub Actions workflow reruns the script
and opens a pull request with the refreshed page for a maintainer to merge.  On a pull request the same
workflow regenerates the page and uploads it as a build artifact for preview
instead of requiring it to be committed.

.. note::

   The table preserves compute measurements for every image size and segmentation head. The Compute & efficiency figure uses 224-pixel classification inputs and matches their accuracy rows by model and band selection. Identified measurements take precedence over rows without provenance; comparisons use the newest single GPU/precision/batch cohort and its latest measurement per model and band choice. Explicit band lists have no matching measurement. GPU names come from the saved measurements; dollar estimates are disabled when the price table has no matching GPU. ``compute_cost.csv`` records throughput and memory but no power draw, so CO2 estimates require separate energy measurements.
