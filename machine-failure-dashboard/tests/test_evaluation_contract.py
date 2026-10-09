import numpy as np
import pandas as pd
import pytest

from machine_failure.evaluate import classification_metrics, select_threshold
from machine_failure.features import PHYSICAL_COLUMNS, PhysicalFeatures


def test_threshold_includes_exact_boundary_and_counts_false_alerts():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.3, 0.3, 0.8]
    threshold = select_threshold(labels, scores)
    assert threshold == 0.3
    result = classification_metrics(labels, scores, threshold)
    assert result["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }
    assert result["recall"] == 1
    assert result["false_alerts_per_1000_non_failures"] == 500
    assert result["f1"] == pytest.approx(0.8)
    report = result["classification_report"]
    assert report["failure"] == {
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1-score": pytest.approx(0.8),
        "support": 2,
    }
    assert report["no_failure"]["f1-score"] == pytest.approx(2 / 3)
    assert report["macro avg"]["f1-score"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert report["weighted avg"]["support"] == 4
    assert report["accuracy"] == result["accuracy"]


def test_classification_report_when_model_predicts_no_failures():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.1, 0.1, 0.1], threshold=0.5)
    report = result["classification_report"]
    assert result["f1"] == 0
    assert report["failure"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 2,
    }
    assert report["no_failure"]["f1-score"] == pytest.approx(2 / 3)
    assert report["macro avg"]["f1-score"] == pytest.approx(1 / 3)
    assert report["weighted avg"]["f1-score"] == pytest.approx(1 / 3)


def test_physical_feature_units_and_outcome_exclusion():
    readings = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [60],
            "Torque [Nm]": [10.0],
            "Tool wear [min]": [5],
            "Machine failure": [1],
            "TWF": [1],
        }
    )
    features = PhysicalFeatures().fit_transform(readings)
    # One revolution per second at 10 Nm produces 20*pi watts.
    np.testing.assert_allclose(
        features.loc[0, list(PHYSICAL_COLUMNS)].to_numpy(dtype=float), [10.0, 20 * np.pi, 50.0]
    )
    assert "Machine failure" not in features and "TWF" not in features
