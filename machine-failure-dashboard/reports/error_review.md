# Five frozen-test errors reviewed

The observations below come from the frozen predictions, official labels, and source rows. They explain limitations and possible future experiments. They do not change this model, threshold, or test report.

| UDI | Result and score | Observation | Consequence or future investigation |
|---|---|---|---|
| 8004 | False alert, 0.3707 | Torque is 67.3 Nm, speed 1,274 rpm, wear 159 min. The official target and all outcome flags are zero. | An inspection would be unnecessary under the benchmark label. Investigate precision near the threshold in a new training-only study. High torque alone does not establish a failure. |
| 8026 | False alert, 0.3576 | Wear is 219 min and torque 50.7 Nm. Official target and outcome flags are zero. Its score is only 0.0517 above the threshold. | Nearby scores can be sensitive to the alert threshold. A future cost-based threshold study must use validation data and report the additional missed failures. |
| 8112 | Missed failure, 0.0190 | Wear is 209 min and torque 33.1 Nm. The official TWF outcome flag is one. | A very low score misses a labelled tool-wear failure. Diagnose recall by wear range in training folds; this snapshot may not supply enough information to distinguish wear failure. The outcome flag cannot become an input. |
| 8200 | Missed failure, 0.0900 | Wear is 225 min and torque 27.0 Nm. The official TWF flag is one. | Another missed tool-wear-labelled case. Row 8026 has similar wear but a negative label, illustrating observed overlap. Maintenance history could be a future input if a different dataset provides it. |
| 8278 | False alert, 0.4381 | Product type H, torque 55.8 Nm, speed 1,294 rpm, wear 202 min. The target and all outcome flags are zero. | This score is 0.1323 above the threshold. Examine training-fold errors by product type and power/wear ranges before proposing a new model. These values do not prove why the forest scored it highly. |

These cases include three unnecessary alerts and two missed failures. Across the entire test set there are six false alerts and eleven misses. Suggested studies require a new declared evaluation protocol; the current test set must not become a tuning set. Probability calibration and real-machine validation remain separate extensions.
