Ranking explorer
================

.. raw:: html

   <p>
     <a href="../_static/ranking_explorer.html" target="_blank" rel="noopener"
        style="display:inline-block;padding:10px 18px;background:#990F3D;
               color:#fff1e5;text-decoration:none;border-radius:2px;
               font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.04em;">
       Open the Ranking Explorer →
     </a>
   </p>

Benchmark results are spread across papers, repositories, and follow-up runs.
The Ranking Explorer is a static comparison of frozen geospatial foundation
models under one evaluation protocol, covering both classification and
segmentation tasks. It is intended as a shared, inspectable record that the
community can use to converge on reliable performance information—not a
declaration of a final winner.

Rankings are summaries: choose the task (classification or segmentation), how
datasets are combined, then the probe and input bands whose results are being
summarized. If a result is wrong, missing, or implausible, please `open an
issue <https://github.com/torchgeo/geo-eval-harness/issues>`__ with the model,
view, and any supporting evidence. General feedback is welcome too.

The aggregation choices answer different questions:

* **Average rank** ranks models within every dataset and averages those
  positions. Score gaps do not matter, so different score ranges do not
  dominate; 1 is best.
* **ELO** summarizes head-to-head wins from every dataset in one rating. Its
  displayed interval is a bootstrapped 95% confidence interval; higher is
  better.
* **Improvability** is the share of remaining error a model must remove to tie
  the best model on each dataset, averaged across datasets; 0 is best.

The fixed methodology and coverage rules are:

* All rows use patch-mean features.
* A model is ranked only with a complete row across the eligible datasets in
  its ``(task, probe, bands)`` view.
* The generator uses complete ``model_native`` results when available and
  otherwise uses complete ``bandspec_zscore`` results; it never combines the
  two.
* Models without a complete row are named below the table.

The GFLOPs chart directly below the table pairs positive, measured 224-pixel
backbone GFLOPs with the selected aggregation. It labels the displayed models
and places their names to avoid collisions where possible. Both axis titles
carry their own direction, so the vertical axis reads ``Worse ←`` to
``→ Better`` and the horizontal axis reads ``Less compute ←`` to
``→ More compute``. Zero or unmeasured values are omitted from the log-scale
plot and listed below it.

The final rank-sensitivity heatmap is an Average-rank comparison across
whichever views you select — preset buttons cover "classification only",
"segmentation only", and "everything" (including cross-task pairs), and a
"Customize which views to compare" panel exposes a checkbox per view for a
fully custom selection. Every cell first finds the models present in both
views, then re-ranks those shared models using each view's *full* dataset set
before calculating tie-aware Kendall τ-b. Thus, values near +1 mean that two
views produce the same ordering, 0 means little ordering agreement, and -1
means a reversal. The row and column labels identify the compared views; each
cell also reports its shared-model count and both dataset counts, and a
dagger marks a small shared roster — cross-task cells are usually small
because classification and segmentation rosters mostly don't overlap, so read
those with extra caution.

The hand-edited source is
:file:`scripts/ranking_explorer.template.html`. The generator
reads its empty JSON anchors and writes the self-contained documentation asset:

.. code-block:: console

   python scripts/regen_leaderboard.py

Use ``--template`` and ``--html`` to render a synthetic template/output pair
in tests. Do not hand-edit :file:`docs/_static/ranking_explorer.html`.
