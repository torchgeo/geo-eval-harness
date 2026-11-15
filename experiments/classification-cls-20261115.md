# Classification CLS companion

This companion adds CLS-pooling rows to the mean-pooling refresh in PR #927. Existing mean and segmentation rows remain unchanged. Evaluation source is `4c8693320629fec9df030bbb04062ebdde6a6f3c`, using the refresh's installed AWS environment, dataset copies, and checkpoint checksums.

Generate configurations from the immutable refresh manifest and its coverage ledger:

```sh
uv run python experiments/plan_classification_cls.py --refresh-manifest outputs/classification-refresh-20261114/manifest.json --coverage results/coverage/classification-refresh-20261114.csv --batch 1 --output outputs/classification-cls-20261115/batch1.json --output-coverage results/coverage/classification-cls-20261115-batch1.csv
```

Use `--batch 2` and corresponding output paths for the Prithvi batch. Batch 2 is launched only after Batch 1 is collected. The manifest stores the paired mean job ID; the coverage CSV uses the refresh ledger schema. Excluded mean inputs remain explicit exclusions, with the original reason. Both manifests retain the evaluation-source commit independently of the planner commit.

Batch 1 contains 262 runnable configurations: Clay v1 56, Clay v1.5 56, Scale-MAE 36, SMARTIES base 50, SMARTIES large 50, and TerraFM S2 14. Its 98 exclusions comprise 8 Clay mixed optical/SAR native-normalization requests, 24 Scale-MAE non-RGB channel requests, 20 SMARTIES mixed-sensor/unsupported-native-normalization requests, and 46 TerraFM missing-pretrained-band requests. Batch 2 contains 224 runnable configurations (56 for each of four Prithvi presets), with 16 mixed optical/SAR native-normalization exclusions.

Each CLS job freezes the mean job's resolved inputs, model kwargs, and evaluation settings before changing pooling and, for named CLS presets, the model name. Scale-MAE retains the mean run's dataset-specific resolution and image size. The unused segmentation settings are copied unchanged. TerraFM retains `checkpoint_path` and `checkpoint_md5`. The planner rejects mean configuration-hash drift and unexpected classification preset differences.

The protocol uses seed 0, 200 bootstrap samples, KNN-5, and linear probing with 40 regularization candidates from log10(C) -6 to 4 and a train-plus-validation refit. Jobs start at batch size 64, use fresh attempt output paths with resume disabled, and halve the batch on recognized OOM errors down to 1. Each effective hash and changed batch size is retained. Execution uses the existing 20-hour attempt limit and explicit failure review. Collection audits source provenance, effective hashes, metadata, dataset split counts, label semantics, metrics, confidence intervals, and duplicate identities before appending rows.

The ranking explorer treats CLS as a distinct model identity, including SMARTIES and TerraFM rows that share a CSV/name with mean pooling. Each pooling independently meets the existing coverage and normalization-selection requirements. Neither pooling fills gaps in the other.

Final comparisons pair methods using the manifest's mean job IDs and audited effective hashes, with equal weight per matched configuration. Report means and counts of disjoint 95% confidence intervals in both directions; disjoint intervals are descriptive comparisons, not a multiple-comparison-adjusted significance test.

Generate the paired summary with `python experiments/summarize_classification_cls.py --refresh-manifest <mean-manifest.json> --refresh-coverage results/coverage/classification-refresh-20261114.csv --manifest <batch-manifest.json> --coverage <batch-coverage.csv> --output <comparison.json>`. It joins audited effective hashes, excludes unpaired methods, and retains separate model/method counts.
