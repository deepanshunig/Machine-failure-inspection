"""Streamlit interaction tests isolate the HTTP client; live integration is checked separately."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from machine_failure.config import PROJECT_ROOT
from machine_failure.frontend import ApiError, DashboardClient, template_csv


class DemoClient:
    def __init__(self):
        self.rows = []
        self.offline = False

    def health(self):
        if self.offline:
            raise ApiError("Cannot reach the prediction service. Start the API and refresh.")
        return {"model_ready": True, "database_ready": True}

    def predict(self, reading):
        row = {
            "prediction_id": len(self.rows) + 1,
            "readings": reading,
            "product_type": reading["product_type"],
            "failure_score": 0.25,
            "threshold": 0.3058576321894592,
            "alert": False,
            "decision": "Below alert threshold",
            "model_version": "frozen-ui-test",
            "latency_ms": 10.0,
            "received_at": "2026-10-09T00:00:00Z",
            "warnings": [],
            "source": "manual",
            "review_status": "pending",
            "review_note": "",
        }
        self.rows.append(row)
        return row

    def predict_batch(self, readings):
        return {"count": len(readings), "items": [self.predict(row) for row in readings]}

    def history(self, **filters):
        rows = list(reversed(self.rows))
        if "alert" in filters:
            rows = [row for row in rows if row["alert"] == filters["alert"]]
        if "product_type" in filters:
            rows = [row for row in rows if row["product_type"] == filters["product_type"]]
        offset = filters.get("offset", 0)
        return {"total": len(rows), "items": rows[offset : offset + filters.get("limit", 50)]}

    def review(self, prediction_id, status, note):
        row = next(row for row in self.rows if row["prediction_id"] == prediction_id)
        row.update(review_status=status, review_note=note)
        return {"prediction_id": prediction_id, "status": status, "note": note}


@pytest.fixture
def app_and_client(monkeypatch):
    client = DemoClient()
    monkeypatch.setattr(DashboardClient, "from_env", classmethod(lambda cls: client))
    app = AppTest.from_file(PROJECT_ROOT / "app/dashboard.py", default_timeout=15).run()
    assert not app.exception
    return app, client


def submit(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_single_prediction_and_inspection_review_flow(app_and_client):
    app, client = app_and_client
    app.radio(key="navigation").set_value("Single prediction").run()
    submit(app, "Run prediction")
    assert len(client.rows) == 1
    assert any("Below alert threshold" in message.value for message in app.success)
    assert (
        next(metric for metric in app.metric if metric.label == "Failure score").value
        == "25.00%"
    )
    assert (
        next(metric for metric in app.metric if metric.label == "Alert threshold").value
        == "30.59%"
    )
    app.radio(key="navigation").set_value("History & reviews").run()
    assert app.dataframe[0].value.iloc[0]["Failure score"] == "25.00%"
    assert app.dataframe[0].value.iloc[0]["Threshold"] == "30.59%"
    app.text_area[0].set_value("Synthetic frontend review")
    app.selectbox(key="review_status_1").set_value("unconfirmed")
    submit(app, "Save review")
    assert client.rows[0]["review_note"] == "Synthetic frontend review"
    assert client.rows[0]["review_status"] == "unconfirmed"
    assert any("Review saved" in message.value for message in app.success)


def test_csv_upload_download_and_invalid_batch_prevents_save(app_and_client):
    app, client = app_and_client
    app.radio(key="navigation").set_value("Batch upload").run()
    app.file_uploader[0].set_value(("sample.csv", template_csv().encode(), "text/csv")).run()
    submit(app, "Score and save batch")
    assert len(client.rows) == 2
    assert len(app.dataframe) >= 2
    results = app.dataframe[-1].value
    assert results["failure_score"].tolist() == ["25.00%", "25.00%"]
    assert results["threshold"].tolist() == ["30.59%", "30.59%"]
    assert any(
        element.label == "Download prediction results" for element in app.get("download_button")
    )
    bad = template_csv().replace(",55,", ",NaN,").encode()
    app.file_uploader[0].set_value(("invalid.csv", bad, "text/csv")).run()
    assert not app.exception
    assert app.error
    assert not any(button.label == "Score and save batch" for button in app.button)
    assert len(client.rows) == 2


def test_offline_state_disables_save_but_preserves_saved_evaluation(app_and_client):
    app, client = app_and_client
    client.offline = True
    app.radio(key="navigation").set_value("Single prediction").run()
    assert app.error
    assert next(button for button in app.button if button.label == "Run prediction").disabled
    app.radio(key="navigation").set_value("Model evaluation").run()
    assert not app.exception
    expected = json.loads((PROJECT_ROOT / "reports/final_evaluation.json").read_text())["test"][
        "f1"
    ]
    assert (
        next(metric for metric in app.metric if metric.label == "Failure F1").value
        == f"{expected:.4f}"
    )
    assert any(
        frame.value.shape[0] == 4 and "F1-score" in frame.value.columns for frame in app.dataframe
    )
