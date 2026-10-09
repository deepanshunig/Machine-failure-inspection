"""Apply the frozen pipeline and retain the frozen decision rule."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from machine_failure.data import FEATURE_COLUMNS, checksum
from machine_failure.train import failure_scores

INPUT_MAPPING = {
    "product_type": "Type",
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}


class ModelRuntime:
    def __init__(self, root: Path):
        self.metadata = json.loads((root / "artifacts/model_metadata.json").read_text("utf-8"))
        artifact = root / "artifacts/selected_pipeline.joblib"
        if checksum(artifact) != self.metadata["artifact_sha256"]:
            raise ValueError("Model artifact differs from its frozen metadata.")
        self.pipeline = joblib.load(artifact)

    def score(self, rows: list[dict]) -> list[dict]:
        start = perf_counter()
        frame = pd.DataFrame(rows).rename(columns=INPUT_MAPPING).loc[:, list(FEATURE_COLUMNS)]
        scores = failure_scores(self.pipeline, frame)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Model produced invalid scores.")
        inference_ms_per_reading = 1000 * (perf_counter() - start) / len(rows)
        threshold = self.metadata["threshold"]
        results = []
        for row, score in zip(rows, scores, strict=True):
            warnings = []
            for key, field in INPUT_MAPPING.items():
                bounds = self.metadata["training_ranges"].get(field)
                if bounds and not bounds["min"] <= row[key] <= bounds["max"]:
                    warnings.append(f"Outside observed training range: {field}.")
            results.append(
                {
                    "readings": row,
                    "failure_score": float(score),
                    "threshold": threshold,
                    "alert": bool(score >= threshold),
                    "decision": "Inspection suggested"
                    if score >= threshold
                    else "Below alert threshold",
                    "model_version": self.metadata["model_version"],
                    "latency_ms": inference_ms_per_reading,
                    "warnings": warnings,
                }
            )
        return results
