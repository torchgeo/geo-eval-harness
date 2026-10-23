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

It covers classification (``knn5`` / ``linear``), segmentation (``seg-linear``,
``seg-conv_block``, ``seg-fpn``, ``seg-dpt``, reported as mIoU), intrinsic
dimension, and the cost/throughput measurements.  ``compute_cost.csv`` is a
wide table that the script melts into the long ``method="profile"`` rows the
page expects.

The page is regenerated automatically: whenever a results CSV lands on
``main``, the ``Results explorer`` GitHub Actions workflow reruns the script
and opens a pull request with the refreshed page for a maintainer to merge, so
the explorer never lags behind ``results/``.  On a pull request the same
workflow regenerates the page and uploads it as a build artifact for preview
instead of requiring it to be committed.

.. note::

   Cost is measured per (model, band config, task, head) at a fixed 224px
   input and carries no dataset, so the Compute & efficiency figure joins it
   to accuracy on ``(name, bands)``: a frozen backbone costs the same
   whichever dataset is probed.  Runs configured with an explicit band list
   rather than ``rgb`` or ``all`` have no matching measurement and are absent
   from that figure.  ``compute_cost.csv`` records throughput and memory but
   no power draw, so the CO2 panel is hidden until energy data returns.
