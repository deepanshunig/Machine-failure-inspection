# Step 2: offline baseline

Models are compared using three forward folds within the 6,000 training records. Each fold fits preprocessing only on its own training records.

| Model | Mean forward-CV AP |
|---|---:|
| Dummy prior | 0.0473 |
| Selected logistic regression | 0.4202 |

Selected configuration: `{'C': 1.0, 'class_weight': None}`. The selected model is refitted on the training partition and saved with its preprocessing. Serialization round-trip predictions matched.

These are training-CV results used for selection, not final held-out performance. The pooled OOF precision-recall plot is descriptive; the ranking uses the mean of fold AP. Validation/test models have not been evaluated.

There are 200 OOF classification errors at the demonstration threshold of 0.5. The error CSV identifies rows for investigation; a final alert threshold will be selected during the model-evaluation milestone.

Next: compare random forests and the engineered-feature configuration using the same training protocol, freeze the winner, select the threshold on validation, and run the final test evaluation.
