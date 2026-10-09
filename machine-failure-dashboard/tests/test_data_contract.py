"""Checks for failures that would compromise data integrity or leak outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machine_failure.data import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    DataIntegrityError,
    load_partition,
    select_features,
    validate_dataset,
    verify_prepared,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    return pd.read_csv(ROOT / "data/raw/ai4i2020.csv")


def test_outcomes_and_identifiers_cannot_enter_features(raw):
    features = select_features(raw)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert not set(EXCLUDED_COLUMNS).intersection(features.columns)


def test_shuffled_source_is_rejected(raw):
    shuffled = raw.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataIntegrityError, match="UDI order"):
        validate_dataset(shuffled)


def test_nonfinite_sensor_is_rejected(raw):
    raw.loc[0, "Torque [Nm]"] = np.inf
    with pytest.raises(DataIntegrityError, match="non-finite"):
        validate_dataset(raw)


def test_missing_reading_is_rejected(raw):
    raw.loc[0, "Type"] = None
    with pytest.raises(DataIntegrityError, match="Missing values"):
        validate_dataset(raw)


def test_source_and_saved_splits_match():
    frame = verify_prepared(ROOT)
    train_x, train_y = load_partition(ROOT, "train")
    validation_x, _ = load_partition(ROOT, "validation")
    test_x, _ = load_partition(ROOT, "test")
    assert len(train_x) == 6000 and len(validation_x) == 2000 and len(test_x) == 2000
    assert train_x.index.max() < validation_x.index.min()
    assert validation_x.index.max() < test_x.index.min()
    pd.testing.assert_series_equal(train_y, frame.iloc[:6000]["Machine failure"])
