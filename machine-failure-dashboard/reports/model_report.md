# Step 3: frozen model evaluation

Version: `ai4i-v1-6ba56612ca`. Selected by training-only forward-CV AP.

Validation-selected threshold: **0.305858**, with `score >= threshold`.

| Final test measurement | Result |
|---|---:|
| Average precision | 0.7923 |
| Failure recall | 71.79% |
| Failure precision | 82.35% |
| Failure F1 | 0.7671 |
| F2 | 0.7368 |
| ROC-AUC | 0.9658 |
| Accuracy | 99.15% |
| False alerts per 1,000 non-failure records | 3.06 |
| Dummy AP | 0.0195 |

Test records: 2,000; failures: 39. Performance estimates are limited by this small positive sample and synthetic source.

Confusion counts: `{'true_negative': 1955, 'false_positive': 6, 'false_negative': 11, 'true_positive': 28}`.

## Final test classification report

Precision, recall, and F1 are shown for both classes. Support is the number of actual test records in each class. Macro averages give each class equal weight; weighted averages use class support.

| Class / average | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No failure (0) | 0.9944 | 0.9969 | 0.9957 | 1,961 |
| Failure (1) | 0.8235 | 0.7179 | 0.7671 | 39 |
| Macro average | 0.9090 | 0.8574 | 0.8814 | 2,000 |
| Weighted average | 0.9911 | 0.9915 | 0.9912 | 2,000 |

Overall accuracy: **99.15%** on 2,000 test records. Failure-class metrics matter because the dataset contains far more non-failure than failure records.

The frozen report also compares the same model at threshold 0.5. Test results are reported without modifying the selected model or threshold.

Errors: 17; see `final_test_errors.csv`. Scores are model outputs; probability calibration and real-machine validation are separate extensions.

## Error review

Inspect the saved rows individually. A false alert triggers an unnecessary demo inspection; a missed failure is an official positive below the alert threshold. Temperature, power, and wear features can suggest follow-up investigations, but their values alone do not establish a causal explanation.

| Original row | Error | Score | Torque (Nm) | Wear (min) |
|---|---|---:|---:|---:|
| 8004 | false_alert | 0.3707 | 67.3 | 159 |
| 8026 | false_alert | 0.3576 | 50.7 | 219 |
| 8112 | missed_failure | 0.0190 | 33.1 | 209 |
| 8200 | missed_failure | 0.0900 | 27.0 | 225 |
| 8278 | false_alert | 0.4381 | 55.8 | 202 |
