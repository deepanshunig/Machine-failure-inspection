"""Freeze a validation-selected decision threshold, then evaluate the final holdout."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from machine_failure.data import DATASET_SHA256, FEATURE_COLUMNS, checksum, load_partition
from machine_failure.train import failure_scores


def select_threshold(target, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(target, scores)
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(
        5 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    choices = np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))
    selected = max(choices, key=lambda i: (precision[i], thresholds[i]))
    return float(thresholds[selected])


def classification_metrics(target, scores, threshold: float) -> dict:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "records": len(target),
        "failures": int(np.asarray(target).sum()),
        "threshold": threshold,
        "average_precision": float(average_precision_score(target, scores)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "f2": float(fbeta_score(target, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "accuracy": (tn + tp) / len(target),
        "false_alerts_per_1000_non_failures": 1000 * fp / (tn + fp),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "classification_report": classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["no_failure", "failure"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_evaluation_plots(root: Path, target, scores, metrics: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "reports/figures"
    precision, recall, _ = precision_recall_curve(target, scores)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(recall, precision, color="#2166ac", label="Frozen selected model")
    ax.axhline(np.mean(target), color="#8b97a3", linestyle="--", label="Test failure prevalence")
    ax.set(xlabel="Failure recall", ylabel="Failure precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Final ordered holdout | precision-recall curve", loc="left", pad=15)
    ax.legend()
    fig.savefig(output / "06_test_precision_recall.png", dpi=160)
    plt.close(fig)

    counts = metrics["confusion_matrix"]
    matrix = np.array(
        [
            [counts["true_negative"], counts["false_positive"]],
            [counts["false_negative"], counts["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=20,
                color="white" if matrix[i, j] > matrix.max() / 2 else "#17212b",
            )
    ax.set_xticks([0, 1], ["No alert", "Inspection suggested"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set(xlabel="Decision at frozen threshold", ylabel="Official target")
    ax.set_title("Final ordered holdout | actual record counts", loc="left", pad=15)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Records")
    fig.savefig(output / "07_test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def run_evaluation(root: Path, *, reproduce: bool = False) -> dict:
    destination = root / "reports/final_evaluation.json"
    if destination.exists() and not reproduce:
        raise ValueError("Final evaluation already exists; use --reproduce for the frozen version.")
    selection = json.loads((root / "reports/model_selection.json").read_text(encoding="utf-8"))
    artifact = root / "artifacts/selected_pipeline.joblib"
    if selection["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("Selection source differs from the verified dataset.")
    if checksum(artifact) != selection["artifact_sha256"]:
        raise ValueError("Selected model changed after selection was frozen.")
    model = joblib.load(artifact)
    validation_x, validation_y = load_partition(root, "validation")
    validation_scores = failure_scores(model, validation_x)
    metadata_path = root / "artifacts/model_metadata.json"
    if reproduce:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["artifact_sha256"] != selection["artifact_sha256"]:
            raise ValueError("The requested reproduction uses a different artifact.")
        threshold = metadata["threshold"]
    else:
        threshold = select_threshold(validation_y, validation_scores)
        quality = json.loads((root / "reports/data_quality.json").read_text(encoding="utf-8"))
        metadata = {
            "model_version": selection["model_version"],
            "artifact_sha256": selection["artifact_sha256"],
            "dataset_sha256": DATASET_SHA256,
            "algorithm": selection["selected"]["algorithm"],
            "params": selection["selected"]["params"],
            "feature_columns": list(FEATURE_COLUMNS),
            "training_ranges": quality["training_ranges"],
            "threshold": threshold,
            "decision_rule": "score >= threshold",
            "threshold_objective": "maximum validation F2; ties favour precision",
            "score_label": "Model failure score",
            "prediction_scope": "current sensor snapshot, synthetic-data prototype",
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "versions": selection["versions"],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Model bytes, features, and the decision threshold are frozen before test scoring.
    test_x, test_y = load_partition(root, "test")
    test_scores = failure_scores(model, test_x)
    training_x, training_y = load_partition(root, "train")
    dummy = DummyClassifier(strategy="prior").fit(training_x, training_y)
    dummy_scores = failure_scores(dummy, test_x)
    report = {
        "model_version": metadata["model_version"],
        "artifact_sha256": metadata["artifact_sha256"],
        "dataset_sha256": DATASET_SHA256,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "original_row_order_60_20_20",
        "validation": classification_metrics(validation_y, validation_scores, threshold),
        "test": classification_metrics(test_y, test_scores, threshold),
        "test_at_default_threshold": classification_metrics(test_y, test_scores, 0.5),
        "dummy_test": classification_metrics(test_y, dummy_scores, 0.5),
    }
    report["test_ap_exceeds_dummy"] = (
        report["test"]["average_precision"] > report["dummy_test"]["average_precision"]
    )
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metadata["evaluation"] = report["test"]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_evaluation_plots(root, test_y, test_scores, report["test"])
    errors = test_x.copy()
    errors.insert(0, "raw_row_index", test_x.index)
    errors["target"] = test_y
    errors["failure_score"] = test_scores
    errors["alert"] = (test_scores >= threshold).astype(int)
    errors = errors[errors["target"] != errors["alert"]]
    errors["error_kind"] = np.where(errors["target"] == 1, "missed_failure", "false_alert")
    errors.to_csv(root / "reports/final_test_errors.csv", index=False)
    metrics = report["test"]
    lines = [
        "# Step 3: frozen model evaluation",
        "",
        f"Version: `{metadata['model_version']}`. Selected by training-only forward-CV AP.",
        "",
        f"Validation-selected threshold: **{threshold:.6f}**, with `score >= threshold`.",
        "",
        "| Final test measurement | Result |",
        "|---|---:|",
        f"| Average precision | {metrics['average_precision']:.4f} |",
        f"| Failure recall | {metrics['recall']:.2%} |",
        f"| Failure precision | {metrics['precision']:.2%} |",
        f"| Failure F1 | {metrics['f1']:.4f} |",
        f"| F2 | {metrics['f2']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| Accuracy | {metrics['accuracy']:.2%} |",
        f"| False alerts per 1,000 non-failure records | "
        f"{metrics['false_alerts_per_1000_non_failures']:.2f} |",
        f"| Dummy AP | {report['dummy_test']['average_precision']:.4f} |",
        "",
        f"Test records: {metrics['records']:,}; failures: {metrics['failures']}. "
        "Performance estimates are limited by this small positive sample and synthetic source.",
        "",
        f"Confusion counts: `{metrics['confusion_matrix']}`.",
        "",
        "## Final test classification report",
        "",
        "Precision, recall, and F1 are shown for both classes. Support is the number of "
        "actual test records in each class. Macro averages give each class equal weight; "
        "weighted averages use class support.",
        "",
        "| Class / average | Precision | Recall | F1-score | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("no_failure", "No failure (0)"),
        ("failure", "Failure (1)"),
        ("macro avg", "Macro average"),
        ("weighted avg", "Weighted average"),
    ):
        row = metrics["classification_report"][key]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1-score']:.4f} | {int(row['support']):,} |"
        )
    lines.extend(
        [
            "",
            f"Overall accuracy: **{metrics['classification_report']['accuracy']:.2%}** "
            f"on {metrics['records']:,} test records. Failure-class metrics matter because "
            "the dataset contains far more non-failure than failure records.",
            "",
            "The frozen report also compares the same model at threshold 0.5. Test results "
            "are reported without modifying the selected model or threshold.",
            "",
            f"Errors: {len(errors)}; see `final_test_errors.csv`. Scores are model outputs; "
            "probability calibration and real-machine validation are separate extensions.",
            "",
            "## Error review",
            "",
            "Inspect the saved rows individually. A false alert triggers an unnecessary demo "
            "inspection; a missed failure is an official positive below the alert threshold. "
            "Temperature, power, and wear features can suggest follow-up investigations, but "
            "their values alone do not establish a causal explanation.",
            "",
            "| Original row | Error | Score | Torque (Nm) | Wear (min) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in errors.head(5).iterrows():
        lines.append(
            f"| {int(row['raw_row_index']) + 1} | {row['error_kind']} | "
            f"{row['failure_score']:.4f} | {row['Torque [Nm]']:.1f} | "
            f"{row['Tool wear [min]']:.0f} |"
        )
    (root / "reports/model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    report = run_evaluation(args.root.resolve(), reproduce=args.reproduce)
    print(
        json.dumps(
            {
                "status": "Frozen holdout evaluation completed",
                "test": report["test"],
                "test_ap_exceeds_dummy": report["test_ap_exceeds_dummy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
