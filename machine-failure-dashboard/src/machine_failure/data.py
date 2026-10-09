"""Acquire, validate, partition, and explore the official AI4I dataset.

Only the six permitted inputs are returned to model code. Exploration is limited
to the training partition; test-set integrity checks do not select model settings.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset"
)
DATASET_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
NUMERIC_COLUMNS = FEATURE_COLUMNS[1:]
TARGET = "Machine failure"
OUTCOME_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
EXCLUDED_COLUMNS = ("UDI", "Product ID", TARGET, *OUTCOME_COLUMNS)
RAW_COLUMNS = ("UDI", "Product ID", *FEATURE_COLUMNS, TARGET, *OUTCOME_COLUMNS)
PARTITIONS = {"train": (0, 6000), "validation": (6000, 8000), "test": (8000, 10000)}


class DataIntegrityError(ValueError):
    """The data no longer satisfies the predeclared source or schema contract."""


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_dataset(root: Path) -> Path:
    destination = root / "data/raw/ai4i2020.csv"
    if destination.exists():
        if checksum(destination) != DATASET_SHA256:
            raise DataIntegrityError("Existing raw CSV checksum differs from the verified source.")
        return destination
    request = Request(DATASET_URL, headers={"User-Agent": "MachineFailurePortfolio/0.1"})
    with urlopen(request, timeout=40) as response:
        archive = response.read(2_000_001)
    if len(archive) > 2_000_000:
        raise DataIntegrityError("Dataset archive exceeded the expected download size.")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        info = bundle.getinfo("ai4i2020.csv")
        if info.file_size > 2_000_000:
            raise DataIntegrityError("CSV exceeded the expected dataset size.")
        payload = bundle.read(info)
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise DataIntegrityError("Official CSV changed; review provenance before accepting it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def validate_dataset(frame: pd.DataFrame, *, expected_rows: int = 10000) -> None:
    if tuple(frame.columns) != RAW_COLUMNS:
        raise DataIntegrityError(f"Unexpected CSV schema: {frame.columns.tolist()}")
    if len(frame) != expected_rows:
        raise DataIntegrityError(f"Expected {expected_rows} records, found {len(frame)}.")
    if frame.isna().any().any():
        raise DataIntegrityError("Missing values require a declared cleaning policy.")
    if not np.array_equal(frame["UDI"].to_numpy(), np.arange(1, expected_rows + 1)):
        raise DataIntegrityError("UDI order or identifiers changed.")
    if not frame["Type"].isin(["L", "M", "H"]).all():
        raise DataIntegrityError("Product type must be L, M, or H.")
    for name in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise DataIntegrityError(f"{name} must be numeric.")
        if not np.isfinite(frame[name].to_numpy()).all():
            raise DataIntegrityError(f"{name} contains non-finite readings.")
    for name in NUMERIC_COLUMNS[:3]:
        if (frame[name] <= 0).any():
            raise DataIntegrityError(f"{name} must be positive.")
    for name in NUMERIC_COLUMNS[3:]:
        if (frame[name] < 0).any():
            raise DataIntegrityError(f"{name} must be nonnegative.")
    for name in (TARGET, *OUTCOME_COLUMNS):
        if not frame[name].isin([0, 1]).all():
            raise DataIntegrityError(f"{name} must contain binary labels.")


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """An explicit allowlist prevents IDs and known outcomes entering models."""
    return frame.loc[:, list(FEATURE_COLUMNS)].copy()


def make_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    validate_dataset(frame)
    labels = np.empty(len(frame), dtype=object)
    for name, (start, stop) in PARTITIONS.items():
        part = frame.iloc[start:stop]
        if set(part[TARGET].unique()) != {0, 1}:
            raise DataIntegrityError(f"Partition {name} must contain both classes.")
        labels[start:stop] = name
    return pd.DataFrame(
        {"raw_row_index": np.arange(len(frame)), "UDI": frame["UDI"], "partition": labels}
    )


def verify_prepared(root: Path) -> pd.DataFrame:
    raw = root / "data/raw/ai4i2020.csv"
    if checksum(raw) != DATASET_SHA256:
        raise DataIntegrityError("Raw CSV checksum changed.")
    frame = pd.read_csv(raw)
    validate_dataset(frame)
    actual = pd.read_csv(root / "data/splits/manifest.csv", dtype={"partition": "object"})
    expected = make_manifest(frame).astype({"partition": "object"})
    if not actual.equals(expected):
        raise DataIntegrityError("Saved split manifest differs from the declared ordered split.")
    provenance = json.loads((root / "data/raw/provenance.json").read_text(encoding="utf-8"))
    if provenance["csv_sha256"] != DATASET_SHA256:
        raise DataIntegrityError("Saved provenance checksum differs from the raw source.")
    if provenance["manifest_sha256"] != checksum(root / "data/splits/manifest.csv"):
        raise DataIntegrityError("Split-manifest checksum differs from provenance.")
    return frame


def load_partition(root: Path, name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name not in PARTITIONS:
        raise ValueError(f"Unknown partition: {name}")
    frame = verify_prepared(root)
    start, stop = PARTITIONS[name]
    part = frame.iloc[start:stop]
    return select_features(part), part[TARGET].copy()


def quality_summary(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    duplicate_rows = frame.duplicated(subset=list(FEATURE_COLUMNS), keep=False)
    label_variation = frame.groupby(list(FEATURE_COLUMNS), observed=True)[TARGET].nunique()
    flag_union = frame.loc[:, list(OUTCOME_COLUMNS)].any(axis=1).astype(int)
    split_counts = {}
    for name, (start, stop) in PARTITIONS.items():
        labels = frame.iloc[start:stop][TARGET]
        split_counts[name] = {
            "records": len(labels),
            "non_failure": int((labels == 0).sum()),
            "failure": int((labels == 1).sum()),
            "failure_rate": float(labels.mean()),
        }
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_feature_rows": int(duplicate_rows.sum()),
        "conflicting_feature_groups": int((label_variation > 1).sum()),
        "outcome_flag_target_disagreements": int((flag_union != frame[TARGET]).sum()),
        "failure_without_outcome_flag": int(((flag_union == 0) & (frame[TARGET] == 1)).sum()),
        "outcome_flag_without_failure": int(((flag_union == 1) & (frame[TARGET] == 0)).sum()),
        "partitions": split_counts,
        "split_policy": "original_row_order_60_20_20",
        "model_features": list(FEATURE_COLUMNS),
        "excluded_from_features": list(EXCLUDED_COLUMNS),
        "training_ranges": {
            col: {
                "min": float(frame.iloc[:6000][col].min()),
                "max": float(frame.iloc[:6000][col].max()),
            }
            for col in NUMERIC_COLUMNS
        },
        "plot_partition": "train",
        "manifest_records": len(manifest),
    }


def save_figures(train: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory.parents[1] / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    counts = train[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    bars = ax.bar(["No failure", "Failure"], counts, color=["#2166ac", "#e36a38"], width=0.6)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 100,
            f"{count:,}\n{count / len(train):.2%}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Training records")
    ax.set_title("Training class balance | original rows 1-6,000", loc="left", pad=18)
    fig.savefig(directory / "01_training_class_balance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, name in zip(axes.flat, FEATURE_COLUMNS, strict=True):
        if name == "Type":
            counts = train[name].value_counts().reindex(["L", "M", "H"], fill_value=0)
            ax.bar(counts.index, counts.values, color="#2166ac")
            ax.set_title("Product-quality type")
        else:
            ax.hist(train[name], bins=30, color="#2166ac", alpha=0.85, edgecolor="white")
            ax.set_title(name)
        ax.set_ylabel("Records")
    fig.suptitle("Six permitted inputs | training partition only", fontsize=17)
    fig.savefig(directory / "02_training_input_distributions.png", dpi=150)
    plt.close(fig)

    columns = [*NUMERIC_COLUMNS, TARGET]
    matrix = train[columns].corr()
    labels = ["Air temp", "Process temp", "Speed", "Torque", "Wear", "Failure"]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    plot = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6), labels, rotation=35, ha="right")
    ax.set_yticks(range(6), labels)
    for i in range(6):
        for j in range(6):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.65 else "#17212b",
            )
    ax.set_title("Numeric correlations | training partition only", loc="left", pad=20)
    fig.colorbar(plot, ax=ax, label="Pearson correlation", shrink=0.85)
    fig.savefig(directory / "03_training_correlations.png", dpi=160)
    plt.close(fig)


def prepare(root: Path) -> dict:
    raw = download_dataset(root)
    frame = pd.read_csv(raw)
    validate_dataset(frame)
    manifest = make_manifest(frame)
    manifest_path = root / "data/splits/manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    provenance = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance",
        "dataset_page": DATASET_PAGE,
        "archive_url": DATASET_URL,
        "doi": "10.24432/C5HS5C",
        "license": "CC BY 4.0",
        "csv_sha256": checksum(raw),
        "manifest_sha256": checksum(manifest_path),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "dependencies": {
            name: version(name) for name in ["numpy", "pandas", "matplotlib", "scikit-learn"]
        },
    }
    (root / "data/raw/provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    summary = quality_summary(frame, manifest)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_quality.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_figures(frame.iloc[:6000], reports / "figures")
    rows = []
    for name, counts in summary["partitions"].items():
        rows.append(
            f"| {name} | {counts['records']:,} | {counts['non_failure']:,} | "
            f"{counts['failure']:,} | {counts['failure_rate']:.2%} |"
        )
    note = (
        "# Step 1: dataset findings\n\n"
        f"Verified official CSV SHA-256: `{provenance['csv_sha256']}`.\n\n"
        f"Source: [UCI AI4I]({DATASET_PAGE}), CC BY 4.0.\n\n"
        f"{len(frame):,} records, {len(frame.columns)} columns. No missing values were found. "
        f"Duplicate input rows: {summary['duplicate_feature_rows']}; conflicting input-label "
        f"groups: {summary['conflicting_feature_groups']}.\n\n"
        "| Partition | Records | No failure | Failure | Failure rate |\n"
        "|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        "The training partition has a different failure prevalence from the later partitions. "
        "Use AP, failure recall/precision, and threshold-specific counts when comparing models. "
        "A high accuracy alone can conceal missed failures.\n\n"
        f"The union of outcome flags disagrees with the official target for "
        f"{summary['outcome_flag_target_disagreements']} records "
        f"({summary['failure_without_outcome_flag']} failures without any flag and "
        f"{summary['outcome_flag_without_failure']} flagged records with a non-failure target). "
        "Preserve the official target. Outcome flags are excluded from model inputs; "
        "their inconsistencies are documented rather than used to relabel records.\n\n"
        "Six input fields are explicitly allowed. IDs, the target, and the five failure-mode "
        "columns never enter the feature matrix. All exploratory plots and numerical ranges "
        "use training records only. Original row order is preserved in the 60/20/20 split. "
        "This is an ordered synthetic holdout, with no verified real timestamps or machine "
        "trajectories. The task is classification of the current snapshot.\n\n"
        "Generated plots: class balance, six input distributions, and numeric correlations. "
        "Model training is the next milestone.\n"
    )
    (reports / "data_note.md").write_text(note, encoding="utf-8")
    verify_prepared(root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        summary = prepare(root)
        print(
            json.dumps(
                {
                    "rows": summary["rows"],
                    "partitions": summary["partitions"],
                    "plots": 3,
                    "status": "Step 1 data preparation passed",
                },
                indent=2,
            )
        )
    else:
        verify_prepared(root)
        print("Verified: source checksum, schema, ordered split, and provenance.")


if __name__ == "__main__":
    main()
