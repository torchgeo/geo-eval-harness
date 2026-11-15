# So2Sat classification reruns

Two unchanged refresh jobs were rerun on evaluation source `4c8693320629fec9df030bbb04062ebdde6a6f3c`, using the existing AWS environment, seed 0, batch size 64, both probes, and fresh output paths with resume disabled. Both reproduced the original KNN score exactly at full precision and retained their original configuration hash. Neither configuration is replaced; the result files and refresh coverage entries remain unchanged.

The replacement rule was fixed before execution: replace only when KNN/hash match and rerun linear accuracy is within 0.01 of the m-so2sat twin. Preserve a repeated low score or any other outcome.

Accuracy and confidence intervals below are percentages; ECE is on the 0–1 scale.

## torchgeo/resnet50_s2rgb_satlas_mi

Job `854a7467f4c65e52c7f5`, hash `b27188873a211a78`. keep: rerun remains in the original low-score range.

| Attempt | KNN (95% CI) | KNN ECE | Linear (95% CI) | Best C | Linear ECE |
|---|---:|---:|---:|---:|---:|
| original | 25.2535 [22.6141, 27.9919] | 0.341989 | 36.3083 [33.3646, 39.1506] | 1701.25428 | 0.563318 |
| rerun | 25.2535 [22.6141, 27.9919] | 0.341989 | 36.3083 [33.3646, 39.1506] | 1701.25428 | 0.563318 |
| twin | 25.2535 [22.7155, 27.8981] | 0.341989 | 40.8722 [38.2353, 43.7145] | 0.0744380301 | 0.143923 |

## smarties/large

Job `6a95b8ccdaa884e08dc0`, hash `ca7642b6d21bb60c`. keep: rerun landed elsewhere.

| Attempt | KNN (95% CI) | KNN ECE | Linear (95% CI) | Best C | Linear ECE |
|---|---:|---:|---:|---:|---:|
| original | 24.7465 [22.6090, 27.0816] | 0.391888 | 50.7099 [47.4620, 54.2647] | 942.668455 | 0.361342 |
| rerun | 24.7465 [22.6090, 27.0816] | 0.391888 | 56.6937 [53.5396, 59.8403] | 4.64158883 | 0.116403 |
| twin | 24.7465 [22.6166, 27.5862] | 0.391888 | 54.0568 [50.8063, 57.0005] | 49.2388263 | 0.237229 |

The original and rerun logs contain no per-candidate validation scores (`runtime.verbose` is false). The retained evidence cannot distinguish a flat validation curve from a second optimum at high C. No logging, optimization, seed, or search settings were changed to obtain another answer.
