"""Compare baseline models using forward folds inside training data only."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from machine_failure.data import (
    DATASET_SHA256,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    checksum,
    load_partition,
)
from machine_failure.features import PHYSICAL_COLUMNS, PhysicalFeatures


def logistic_pipeline(c: float, class_weight: str | None) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("type", OneHotEncoder(categories=[["L", "M", "H"]], sparse_output=False), ["Type"]),
            ("numeric", StandardScaler(), list(NUMERIC_COLUMNS)),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(C=c, class_weight=class_weight, max_iter=3000, random_state=42),
            ),
        ]
    )


def forest_pipeline(*, physical: bool, max_depth: int | None, class_weight: str | None) -> Pipeline:
    numeric = [*NUMERIC_COLUMNS, *(PHYSICAL_COLUMNS if physical else ())]
    preprocess = ColumnTransformer(
        [
            ("type", OneHotEncoder(categories=[["L", "M", "H"]], sparse_output=False), ["Type"]),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", PhysicalFeatures() if physical else "passthrough"),
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=max_depth,
                    min_samples_leaf=2,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def failure_scores(model, features: pd.DataFrame) -> np.ndarray:
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, positive_column]


def forward_evaluate(
    model, features: pd.DataFrame, target: pd.Series
) -> tuple[list[dict], pd.DataFrame]:
    folds = []
    predictions = []
    for fold, (train, evaluate) in enumerate(TimeSeriesSplit(n_splits=3).split(features), 1):
        if train.max() >= evaluate.min():
            raise ValueError("Forward fold contains overlapping or reversed indices.")
        for indices in (train, evaluate):
            if set(target.iloc[indices].unique()) != {0, 1}:
                raise ValueError("Both classes are required in every forward fold.")
        fitted = clone(model).fit(features.iloc[train], target.iloc[train])
        scores = failure_scores(fitted, features.iloc[evaluate])
        folds.append(
            {
                "fold": fold,
                "fit_records": len(train),
                "fit_first_row": int(train.min()) + 1,
                "fit_last_row": int(train.max()) + 1,
                "evaluation_first_row": int(evaluate.min()) + 1,
                "evaluation_last_row": int(evaluate.max()) + 1,
                "evaluation_records": len(evaluate),
                "evaluation_failures": int(target.iloc[evaluate].sum()),
                "ap": float(average_precision_score(target.iloc[evaluate], scores)),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "raw_row_index": features.iloc[evaluate].index.to_numpy(),
                    "fold": fold,
                    "target": target.iloc[evaluate].to_numpy(),
                    "failure_score": scores,
                }
            )
        )
    return folds, pd.concat(predictions, ignore_index=True)


def plot_baselines(root: Path, dummy: dict, selected: dict, oof: pd.DataFrame) -> None:
    cache = root / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "reports/figures"
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    scores = [dummy["mean_ap"], selected["mean_ap"]]
    bars = ax.bar(
        ["Dummy baseline", "Selected logistic regression"],
        scores,
        color=["#8b97a3", "#2166ac"],
        width=0.6,
    )
    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.015,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, min(1, max(scores) * 1.3 + 0.08))
    ax.set_ylabel("Mean average precision across 3 forward folds")
    ax.set_title("Baseline comparison | training-only cross-validation", loc="left", pad=15)
    fig.savefig(output / "04_baseline_cv_ap.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(oof["target"], oof["failure_score"])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(recall, precision, color="#2166ac", label="Logistic regression OOF predictions")
    prevalence = oof["target"].mean()
    ax.axhline(prevalence, color="#8b97a3", linestyle="--", label="OOF failure prevalence")
    ax.set(xlabel="Failure recall", ylabel="Failure precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Precision-recall tradeoff | training forward folds", loc="left", pad=15)
    ax.legend(loc="upper right")
    fig.savefig(output / "05_training_oof_precision_recall.png", dpi=160)
    plt.close(fig)


def run_baselines(root: Path) -> dict:
    features, target = load_partition(root, "train")
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Training input allowlist changed.")
    dummy_folds, _ = forward_evaluate(DummyClassifier(strategy="prior"), features, target)
    dummy = {
        "name": "dummy_prior",
        "folds": dummy_folds,
        "mean_ap": float(np.mean([fold["ap"] for fold in dummy_folds])),
    }
    candidates = []
    outputs = {}
    for c in (0.1, 1.0, 10.0):
        for weight in (None, "balanced"):
            name = f"logistic_C{c}_{weight or 'ordinary'}"
            folds, oof = forward_evaluate(logistic_pipeline(c, weight), features, target)
            candidate = {
                "name": name,
                "params": {"C": c, "class_weight": weight},
                "folds": folds,
                "mean_ap": float(np.mean([fold["ap"] for fold in folds])),
                "std_ap": float(np.std([fold["ap"] for fold in folds])),
            }
            print(f"{name}: mean training CV AP={candidate['mean_ap']:.4f}", flush=True)
            candidates.append(candidate)
            outputs[name] = oof
    selected = max(candidates, key=lambda item: item["mean_ap"])
    fitted = logistic_pipeline(selected["params"]["C"], selected["params"]["class_weight"]).fit(
        features, target
    )
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    artifact = artifacts / "logistic_baseline.joblib"
    joblib.dump(fitted, artifact)
    restored = joblib.load(artifact)
    if not np.allclose(
        failure_scores(fitted, features.iloc[:20]),
        failure_scores(restored, features.iloc[:20]),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Saved baseline pipeline does not reproduce its predictions.")
    selected_oof = outputs[selected["name"]]
    selected_oof.to_csv(root / "reports/baseline_oof_predictions.csv", index=False)
    report = {
        "stage": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": DATASET_SHA256,
        "selection_partition": "train",
        "selection_metric": "mean_training_forward_cv_average_precision",
        "dummy": dummy,
        "candidates": candidates,
        "selected_logistic": selected,
        "held_out_validation_evaluated": False,
        "held_out_test_evaluated": False,
        "deployment_threshold_selected": False,
        "versions": {name: version(name) for name in ["scikit-learn", "pandas", "numpy", "joblib"]},
    }
    (root / "reports/baseline_cv.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_baselines(root, dummy, selected, selected_oof)
    errors = selected_oof.copy()
    errors["prediction_at_0_5"] = (errors["failure_score"] >= 0.5).astype(int)
    errors = errors[errors["prediction_at_0_5"] != errors["target"]]
    errors.to_csv(root / "reports/baseline_oof_errors.csv", index=False)
    markdown = (
        "# Step 2: offline baseline\n\n"
        "Models are compared using three forward folds within the 6,000 training records. "
        "Each fold fits preprocessing only on its own training records.\n\n"
        "| Model | Mean forward-CV AP |\n|---|---:|\n"
        f"| Dummy prior | {dummy['mean_ap']:.4f} |\n"
        f"| Selected logistic regression | {selected['mean_ap']:.4f} |\n\n"
        f"Selected configuration: `{selected['params']}`. The selected model is refitted "
        "on the training partition and saved with its preprocessing. Serialization "
        "round-trip predictions matched.\n\n"
        "These are training-CV results used for selection, not final held-out performance. "
        "The pooled OOF precision-recall plot is descriptive; the ranking uses the mean "
        "of fold AP. Validation/test models have not been evaluated.\n\n"
        f"There are {len(errors)} OOF classification errors at the demonstration threshold "
        "of 0.5. The error CSV identifies rows for investigation; a final alert threshold "
        "will be selected during the model-evaluation milestone.\n\n"
        "Next: compare random forests and the engineered-feature configuration using "
        "the same training protocol, freeze the winner, select the threshold on validation, "
        "and run the final test evaluation.\n"
    )
    (root / "reports/baseline_report.md").write_text(markdown, encoding="utf-8")
    return report


def run_comparison(root: Path) -> dict:
    if (root / "reports/final_evaluation.json").exists():
        raise ValueError(
            "Final test was evaluated. Preserve this version before new selection work."
        )
    features, target = load_partition(root, "train")
    baseline = json.loads((root / "reports/baseline_cv.json").read_text(encoding="utf-8"))
    if baseline["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("Baseline was fitted on a different source.")
    logistic = {**baseline["selected_logistic"], "algorithm": "logistic_regression"}
    candidates = [logistic]
    for physical in (False, True):
        for depth in (8, None):
            for weight in (None, "balanced"):
                name = f"forest_{'physical' if physical else 'raw'}_depth{depth}_{weight}"
                params = {"physical": physical, "max_depth": depth, "class_weight": weight}
                folds, _ = forward_evaluate(forest_pipeline(**params), features, target)
                candidate = {
                    "name": name,
                    "algorithm": "random_forest",
                    "params": params,
                    "folds": folds,
                    "mean_ap": float(np.mean([fold["ap"] for fold in folds])),
                    "std_ap": float(np.std([fold["ap"] for fold in folds])),
                }
                candidates.append(candidate)
                print(f"{name}: mean training CV AP={candidate['mean_ap']:.4f}", flush=True)
    selected = max(candidates, key=lambda item: item["mean_ap"])
    if selected["algorithm"] == "random_forest":
        model = forest_pipeline(**selected["params"])
    else:
        model = logistic_pipeline(selected["params"]["C"], selected["params"]["class_weight"])
    model.fit(features, target)
    artifact = root / "artifacts/selected_pipeline.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    np.testing.assert_allclose(
        failure_scores(model, features.iloc[:20]),
        failure_scores(restored, features.iloc[:20]),
        rtol=0,
        atol=1e-12,
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": DATASET_SHA256,
        "selection_partition": "train",
        "selection_metric": "mean_training_forward_cv_average_precision",
        "candidates": candidates,
        "selected": selected,
        "artifact_sha256": checksum(artifact),
        "model_version": f"ai4i-v1-{checksum(artifact)[:10]}",
        "fit_records": 6000,
        "validation_used_for_model_selection": False,
        "test_used_for_model_selection": False,
        "versions": baseline["versions"],
    }
    (root / "reports/model_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "compare"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "baseline":
        report = run_baselines(args.root.resolve())
        summary = {
            "status": "Step 2 baseline completed",
            "dummy_ap": report["dummy"]["mean_ap"],
            "logistic_ap": report["selected_logistic"]["mean_ap"],
        }
    else:
        report = run_comparison(args.root.resolve())
        summary = {
            "status": "Model comparison frozen",
            "selected": report["selected"]["name"],
            "cv_ap": report["selected"]["mean_ap"],
            "version": report["model_version"],
        }
    summary["held_out_test_evaluated"] = False
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
