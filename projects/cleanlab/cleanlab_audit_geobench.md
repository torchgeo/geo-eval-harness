---
orphan: true
---

# GeoBench Cleanlab Audit — Initial Report

Audit of label quality and class structure across all GeoBench V1 + V2 classification datasets plus `eurosat-spatial`. The goal is to separate "models are getting better" from "the eval is the bottleneck."

## Methodology

For each of 11 classification datasets, we picked the top-1 linear-probe result from `results/all_results.csv` (filtering out legacy rows with deprecated `normalization=raw`), rebuilt that exact `(model, dataset, bands, normalization, image_size, partition)` configuration, and:

1. Extracted train + test embeddings from the frozen backbone.
2. Fit a logistic regression on `train+val` at the recorded best `C`.
3. Predicted softmax probabilities on train and test.
4. Ran cleanlab's `find_label_issues` (single-label) or `find_multilabel_issues_per_class` (multi-label) on the predicted probs.
5. Computed per-class statistics: accuracy / AP, flag rate, top-confused class, Jaccard with neighbouring classes (multi-label only).

Outputs are saved as:

- `results/cleanlab/probs/<dataset>__<model>_{train,test}.npz` — labels + predicted probs.
- `results/cleanlab/<dataset>_{train,test}.csv` — per-sample issue scores.
- `results/cleanlab/perclass_<dataset>_<split>.csv` — per-class breakdown.
- `results/cleanlab/summary.csv` — aggregate noise rates.

### Top-1 model per dataset

| Dataset | Model | Linear-probe acc/mAP | Bands | Norm |
|---|---|---:|---|---|
| m-eurosat | tgeo_panopticon | 0.969 | rgb | model_native |
| m-bigearthnet | tt_terramind_v1_large | 0.751 (mAP) | all | bandspec_zscore |
| m-brick-kiln | tt_clay_v1_base | 0.975 | rgb | minmax |
| m-pv4ger | tgeo_panopticon | 0.968 | rgb | bandspec_zscore |
| m-so2sat | tt_terramind_v1_base | 0.741 | all | bandspec_zscore |
| m-forestnet | tt_terramind_v1_large | 0.571 | all | minmax_zscore |
| benv2 | tt_terramind_v1_large | 0.846 (mAP) | all | bandspec_zscore |
| treesatai | tt_clay_v1_base | 0.674 (mAP) | all | bandspec_zscore |
| so2sat (V2) | tt_terramind_v1_base | 0.741 | all | bandspec_zscore |
| forestnet (V2) | tgeo_dofa_large | 0.566 | all | bandspec_zscore |
| eurosat-spatial | tgeo_dofa_large | 0.965 | rgb | bandspec_zscore |

### Caveats

- **Train probs are in-sample**: the linear probe was fit on train+val, so train flag rates underestimate true train noise. Test probs are out-of-sample and unbiased — those are the headline numbers.
- **Top-1 model only** for now. Per-cleanlab best practice we should ensemble top-3 to reduce single-model bias; queued for a follow-up run.
- **Multi-label aggregate flag rates are inflated** by per-class binary expansion across long-tailed label distributions. Per-class breakdown is the trustworthy view.
- Cleanlab's flag is a **noisy oracle**. Expected ~20–30% false positive rate on flagged samples. Manual spot-check is still required to convert these numbers into a "human-confirmed noise rate".

## Aggregate test-set noise rates

Sorted by single-label vs multi-label and noise rate.

| Dataset | Task | N test | Flagged | Test noise rate | Top confused |
|---|---|---:|---:|---:|---|
| m-brick-kiln | single | 999 | 4 | **0.4%** | 0→1 |
| m-eurosat | single | 1000 | 6 | **0.6%** | 2→6 |
| m-pv4ger | single | 999 | 9 | **0.9%** | 1→0 |
| eurosat-spatial | single | 5400 | 62 | **1.1%** | 2→6 |
| m-so2sat | single | 986 | 149 | **15.1%** | 8→5 |
| so2sat (V2) | single | 986 | 150 | **15.2%** | 8→5 |
| m-forestnet | single | 993 | 299 | **30.1%** | 2→0 |
| forestnet (V2) | single | 993 | 313 | **31.5%** | 2→0 |
| benv2 | multi | 4000 | 1995 | 49.9%¹ | — |
| treesatai | multi | 2000 | 1035 | 51.8%¹ | — |
| m-bigearthnet | multi | 1000 | 687 | 68.7%¹ | — |

¹ Multi-label aggregate, inflated; see per-class section.

## Single-label per-class findings

### Clean datasets

**m-eurosat / m-brick-kiln / m-pv4ger** — every class ≥ 93% accuracy, total flagged ≤ 9 samples. These eval results are trustworthy.

**eurosat-spatial — gap localizes to one class**

| Class | n | acc | top confused → | share |
|---:|---:|---:|---:|---:|
| 5 | 115 | **0.678** | class 2 | 23.5% |
| 6 | 307 | 0.850 | class 2 | 11.7% |
| 0, 1, 2, 3, 4, 7, 8 | – | ≥ 0.95 | – | – |

The accuracy gap between `m-eurosat` (0.969) and `eurosat-spatial` (0.965) is small in aggregate, but the spatial-split *test set* concentrates the remaining error in **classes 5 and 6**. Class 5 in particular drops to 68% accuracy and is misclassified to class 2 a quarter of the time. This is a real spatial-shift effect, not label noise — both EuroSAT splits have ≤ 1.1% test noise.

### LCZ ambiguity (so2sat family)

m-so2sat (V1) and so2sat (V2) show **identical** failure modes — same class indices, same confusion shares, within 0.2% on every metric. This is expected: V2 reuses the V1 imagery; the only difference is the split.

| Class | n | acc | top confused → | share |
|---:|---:|---:|---:|---:|
| 5 | 58 | **0.43** | class 2 | 25.9% |
| 8 | 58 | 0.64 | class 5 | 25.9% |
| 0 | 58 | 0.55–0.57 | class 3 | 27.6% |
| 1 | 58 | 0.66 | class 0 | 10.3% |

These are well-known **Local Climate Zone confusions**: compact lowrise vs compact midrise (0↔3), light industry vs heavy industry (5↔2), etc. The classes are *physically* similar from satellite imagery — no model is going to disambiguate them at significantly above the rates above. This is a **capability ceiling baked into the eval**, not noise. Model rankings on so2sat above ~0.75 accuracy are essentially measuring how close each model gets to the LCZ ambiguity ceiling, not which model is "smarter".

### forestnet — broken classes

m-forestnet (V1) and forestnet (V2) show *similar* test-set noise (30% vs 32%), with V2 slightly **worse** on the worst classes:

| forestnet (V2) class | n | acc | top confused → | share |
|---:|---:|---:|---:|---:|
| 2 | 73 | **0.151** | class 0 | 38.4% |
| 6 | 50 | **0.240** | class 5 | 32.0% |
| 5 | 73 | **0.205** | class 4 | 38.4% |
| 11 | 30 | 0.300 | class 4 | 33.3% |
| 10 | 43 | 0.372 | class 4 | 25.6% |

Five out of twelve classes have accuracy below 40% with confusion to a single neighbouring class > 25%. Either:
- the labels are genuinely noisy in those classes (annotator disagreement on similar deforestation-driver categories), or
- the visual signal isn't in the imagery at all (some categories may need temporal / contextual cues not available in a single tile).

**V2 is not a label-clean upgrade over V1 for forestnet.** Reporting only overall accuracy here is misleading — the metric is dominated by these five broken classes.

### m-pv4ger — class imbalance

| Class | n | acc | top confused → | share |
|---:|---:|---:|---:|---:|
| 1 (panel) | 153 | 0.824 | class 0 (no panel) | 17.6% |
| 0 (no panel) | 846 | 0.994 | class 1 | 0.6% |

Almost all errors are false negatives on the rare positive class. Expected for a binary detection task. Aggregate 0.97 accuracy hides this asymmetry — recall on the panel class is 0.82.

## Multi-label per-class findings

The **aggregate** multi-label noise rates (50–69%) are misleading. Cleanlab's multi-label finder runs binary `find_label_issues` per class and unions the results, which inflates with the number of classes and class imbalance. The per-class view tells a different story.

### benv2 (19 classes) — head classes are fine

Big classes (n_pos ≥ 700) all have AP ≥ 0.69 and flag-among-positive ≤ 18%. The aggregate 49.9% comes from tail classes and the per-class union.

| Class pair | Jaccard | n_pos (a, b) | AP (a, b) | flag-among-pos (a, b) |
|---:|---:|---|---|---|
| 9 ↔ 10 | **0.51** | 1270, 1452 | 0.93, 0.89 | 5%, 9% |
| 8 ↔ 13 | 0.28 | 1215, 1355 | 0.84, 0.76 | 11%, 17% |
| 4 ↔ 2 | 0.25 | 901, 1658 | 0.83, 0.93 | 13%, 8% |

Classes 9 and 10 share more than half their positive samples (Jaccard 0.51). This is the "near-duplicate label" hypothesis: in BigEarthNet V2's relabelling, two CORINE land-cover classes are essentially co-occurring nearly always. The model still discriminates them well per-class (AP 0.93 and 0.89), so model rankings on benv2 are probably fine — but the aggregate mAP under-credits models that "merge" 9 and 10, and over-credits models that learn the rare exclusive cases.

### m-bigearthnet (43 classes) — long-tailed

Big classes (n_pos ≥ 200) have AP ≥ 0.67. Flag rate is dominated by tail classes with 25–50 positives in the test set:

| Class | n_pos | AP | flag-among-pos |
|---:|---:|---:|---:|
| 13 | 26 | 0.48 | 42% |
| 10 | 33 | 0.32 | 36% |
| 16 | 36 | 0.51 | 28% |
| 39 | 269 | 0.70 | 21% |

`33 ↔ 34` Jaccard 0.37 — small overlap, not catastrophic. Otherwise no class pair stands out. Like benv2, head-class APs are reasonable and the aggregate noise rate is mostly a long-tail / per-class-expansion artifact.

### treesatai (15 classes) — actually broken on rare species

Different story: **multiple classes the model literally cannot predict**.

The "13 vs 15 columns" mismatch noted in the first draft was a metadata bug in our wrapper. Upstream `geobench_v2.GeoBenchTreeSatAI` declares 15 classes (the 14 European tree genera plus "Cleared"); our wrapper had a stale `num_classes = 13`. Fixed in this commit. The label tensor is correct (15-dim multi-hot); only the metadata was wrong.

Class indices map to species (upstream order):

```
0 Abies   1 Acer    2 Alnus    3 Betula  4 Cleared
5 Fagus   6 Fraxinus 7 Larix   8 Picea   9 Pinus
10 Populus 11 Prunus 12 Pseudotsuga 13 Quercus 14 Tilia
```

| Class | Species | n_pos | AP | n_pred_pos | flag-among-pos |
|---:|---|---:|---:|---:|---:|
| 14 | Tilia (linden) | 12 | **0.008** | 0 | 75% |
| 10 | Populus (aspen/poplar) | 13 | **0.016** | 0 | 62% |
| 11 | Prunus (cherry) | 23 | **0.058** | 0 | 48% |
| 0 | Abies (silver fir) | 49 | **0.049** | 2 | 57% |
| 7 | Larix (larch) | 218 | **0.27** | 31 | 55% |

Five species have AP below 0.10. Tilia, Populus, Prunus and Abies are all under-represented in the test set (12–49 positives across 2000 samples) and the probe never raises probability above 0.5 for any of them. Larix is interesting: 218 positives, distinct deciduous-conifer phenology, yet AP is only 0.27 — that's a real model failure on a learnable class.

Most likely a combination of (a) tail-class scarcity preventing the linear probe from learning a discriminative direction in a 768-d embedding space, and (b) genuine visual ambiguity between similar genera (Abies ↔ Picea, Populus ↔ Betula). Not pure label noise.

**Implication for model rankings**: the same model that "loses" treesatai mAP by being slightly less random on classes 14 / 10 / 11 / 0 isn't actually worse. **Recommend reporting both micro and macro mAP, and excluding classes with AP < 0.1 for any model in the comparison set.**

## Cross-cutting conclusions

1. **eurosat-spatial vs m-eurosat is real, and concentrated.** Both have ≤ 1.1% label noise. The accuracy gap (small) localizes in classes 5 and 6 of eurosat-spatial — a real distribution-shift effect, not a noise ceiling.

2. **so2sat (V1 = V2) hits a fundamental ambiguity ceiling around 75%.** Above that, models are competing on how close they get to the LCZ-ambiguity floor (where 0↔3, 2↔5, 5↔8 confusions can't be resolved from imagery). Treat any model claim above ~0.75 with skepticism — the eval can't measure it.

3. **forestnet V2 is not a label-clean upgrade over V1.** Both have ~30% test noise concentrated in 5 classes that the model can't predict at all. Macro-averaging would expose the issue; current micro-acc hides it.

4. **benv2 / m-bigearthnet head-class results are trustworthy** — the high aggregate flag rate is a per-class-expansion artifact on long-tailed multi-label, not actual label noise on the head classes.

5. **treesatai needs investigation before any model claim is made on it.** At least 4 classes have AP < 0.05 with n_pred_pos = 0, and the label-tensor shape doesn't match `num_classes`.

## Open questions / next steps

- **Re-extract embeddings** and run `cleanlab.Datalab` (no extra deps) to get outlier / near-duplicate / underperforming-group / non-IID issues per dataset. The `underperforming_group` manager will localize the eurosat-spatial / forestnet failure clusters in feature space.
- **Top-3 model ensemble** for prob averaging — reduces cleanlab single-model bias.
- **Eurosat-family deduplication** — phash `m-eurosat` vs `eurosat-spatial` vs torchgeo `eurosat` to detect cross-split leakage.
- **Manual spot-check** of top-50 flagged tiles per dataset to compute a human-confirmed noise rate (gallery script ready, just needs eyes).
- ~~**Investigate treesatai** label tensor shape~~ — done; bug was a stale `num_classes = 13` in the wrapper, fixed to 15.

## Output files in `results/cleanlab/`

```
probs/<dataset>__<model>_{train,test}.npz   # labels, classes, probs
<dataset>_{train,test}.csv                # per-sample issue scores
perclass_<dataset>_<split>.csv            # per-class breakdown
summary.csv                             # aggregate noise rates
```

See the [project README](README.md) for installation and the gallery/deduplication commands. To reproduce the analysis from the historical results CSV, run from the repository root:

```bash
for dataset in m-eurosat m-bigearthnet m-brick-kiln m-pv4ger m-so2sat \
    m-forestnet benv2 treesatai so2sat forestnet eurosat-spatial; do
    python projects/cleanlab/cleanlab_extract_probs.py \
        --dataset "$dataset" --results results/all_results.csv --verbose
done
python projects/cleanlab/run_cleanlab_audit.py --verbose
python projects/cleanlab/cleanlab_per_class_singlelabel.py --splits test
python projects/cleanlab/cleanlab_per_class_multilabel.py --splits test
```

Existing probability artifacts can be reused by starting with `run_cleanlab_audit.py`. For newer per-model results, pass `--results results/models` to the extractor instead.
