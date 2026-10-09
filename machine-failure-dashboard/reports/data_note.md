# Step 1: dataset findings

Verified official CSV SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

Source: [UCI AI4I](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset), CC BY 4.0.

10,000 records, 14 columns. No missing values were found. Duplicate input rows: 0; conflicting input-label groups: 0.

| Partition | Records | No failure | Failure | Failure rate |
|---|---:|---:|---:|---:|---:|
| train | 6,000 | 5,745 | 255 | 4.25% |
| validation | 2,000 | 1,955 | 45 | 2.25% |
| test | 2,000 | 1,961 | 39 | 1.95% |

The training partition has a different failure prevalence from the later partitions. Use AP, failure recall/precision, and threshold-specific counts when comparing models. A high accuracy alone can conceal missed failures.

The union of outcome flags disagrees with the official target for 27 records (9 failures without any flag and 18 flagged records with a non-failure target). Preserve the official target. Outcome flags are excluded from model inputs; their inconsistencies are documented rather than used to relabel records.

Six input fields are explicitly allowed. IDs, the target, and the five failure-mode columns never enter the feature matrix. All exploratory plots and numerical ranges use training records only. Original row order is preserved in the 60/20/20 split. This is an ordered synthetic holdout, with no verified real timestamps or machine trajectories. The task is classification of the current snapshot.

Generated plots: class balance, six input distributions, and numeric correlations. Model training is the next milestone.
