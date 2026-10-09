"""The fitted pipeline must ignore outcomes and survive persistence."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from machine_failure.data import OUTCOME_COLUMNS, TARGET, select_features
from machine_failure.train import failure_scores, logistic_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fitted():
    raw = pd.read_csv(ROOT / "data/raw/ai4i2020.csv").iloc[:1500]
    model = logistic_pipeline(1.0, None).fit(select_features(raw), raw[TARGET])
    return model, raw


def test_outcome_changes_do_not_change_predictions(fitted):
    model, raw = fitted
    readings = raw.iloc[:20].copy()
    expected = failure_scores(model, readings)
    readings[TARGET] = 1 - readings[TARGET]
    readings.loc[:, list(OUTCOME_COLUMNS)] = 1
    np.testing.assert_allclose(failure_scores(model, readings), expected, rtol=0, atol=1e-12)


def test_pipeline_persistence_preserves_preprocessing_and_scores(fitted, tmp_path):
    model, raw = fitted
    saved = tmp_path / "pipeline.joblib"
    joblib.dump(model, saved)
    restored = joblib.load(saved)
    np.testing.assert_allclose(
        failure_scores(restored, select_features(raw.iloc[:20])),
        failure_scores(model, select_features(raw.iloc[:20])),
        rtol=0,
        atol=1e-12,
    )
