"""HTTP contracts use a test repository; these are not MySQL integration tests."""

import json

import pytest
from fastapi.testclient import TestClient

from machine_failure.api import create_app
from machine_failure.config import PROJECT_ROOT, Settings
from machine_failure.inference import ModelRuntime

READING = {
    "product_type": "L",
    "air_temperature_k": 300.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 100.0,
}


class TestRepository:
    __test__ = False

    def __init__(self):
        self.saved = []
        self.fail = False
        self.filters = None

    def ready(self):
        return not self.fail

    def save_predictions(self, results, metadata, source):
        if self.fail:
            raise RuntimeError("Private database diagnostic")
        saved = []
        for result in results:
            item = {
                **result,
                "prediction_id": len(self.saved) + 1,
                "reading_id": len(self.saved) + 1,
                "received_at": "2026-10-09T00:00:00Z",
                "source": source,
            }
            self.saved.append(item)
            saved.append(item)
        return saved

    def history(self, **filters):
        self.filters = filters
        return {
            "total": len(self.saved),
            "items": self.saved,
            "limit": filters["limit"],
            "offset": filters["offset"],
        }

    def review(self, prediction_id, status, note):
        if not any(row["prediction_id"] == prediction_id for row in self.saved):
            raise LookupError
        return {"prediction_id": prediction_id, "status": status, "note": note}

    def dispose(self):
        pass


@pytest.fixture
def client_and_repository():
    repository = TestRepository()
    app = create_app(Settings(root=PROJECT_ROOT), repository=repository)
    with TestClient(app) as client:
        yield client, repository


def test_prediction_preserves_frozen_pipeline_score(client_and_repository):
    client, repository = client_and_repository
    expected = ModelRuntime(PROJECT_ROOT).score([READING])[0]
    response = client.post("/predict", json=READING)
    assert response.status_code == 200
    result = response.json()
    assert result["failure_score"] == pytest.approx(expected["failure_score"])
    assert result["threshold"] == expected["threshold"]
    assert result["alert"] == expected["alert"]
    assert result["model_version"] == expected["model_version"]
    assert len(repository.saved) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"product_type": "UNKNOWN"},
        {"torque_nm": -1},
        {"rotational_speed_rpm": 0},
        {"Machine failure": 1},
        {"TWF": 0},
        {"air_temperature_k": None},
    ],
)
def test_invalid_reading_does_not_write(client_and_repository, changes):
    client, repository = client_and_repository
    assert client.post("/predict", json={**READING, **changes}).status_code == 422
    assert repository.saved == []


def test_missing_nonfinite_and_invalid_batch_do_not_write(client_and_repository):
    client, repository = client_and_repository
    missing = {key: value for key, value in READING.items() if key != "tool_wear_min"}
    assert client.post("/predict", json=missing).status_code == 422
    nonfinite = json.dumps(READING).replace('"torque_nm": 40.0', '"torque_nm": NaN')
    response = client.post(
        "/predict", content=nonfinite, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert client.post("/predict-batch", json={"readings": [READING, missing]}).status_code == 422
    assert repository.saved == []


def test_database_failure_returns_no_success_or_private_error(client_and_repository):
    client, repository = client_and_repository
    repository.fail = True
    response = client.post("/predict", json=READING)
    assert response.status_code == 503
    assert "Private" not in response.text
    assert repository.saved == []
    assert client.get("/health").status_code == 503


def test_api_key_and_batch_limit():
    repository = TestRepository()
    app = create_app(Settings(root=PROJECT_ROOT, api_key="test-key"), repository=repository)
    with TestClient(app) as client:
        assert client.post("/predict", json=READING).status_code == 401
        headers = {"X-API-Key": "test-key"}
        response = client.post(
            "/predict-batch", json={"readings": [READING, READING]}, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["count"] == 2
        assert (
            client.post(
                "/predict-batch", json={"readings": [READING] * 1001}, headers=headers
            ).status_code
            == 422
        )
        assert len(repository.saved) == 2


def test_history_filters_and_review_not_found(client_and_repository):
    client, repository = client_and_repository
    response = client.get(
        "/predictions", params={"limit": 10, "offset": 2, "alert": "true", "product_type": "L"}
    )
    assert response.status_code == 200
    assert repository.filters["alert"] is True
    assert repository.filters["limit"] == 10
    assert repository.filters["offset"] == 2
    assert (
        client.get(
            "/predictions", params={"start": "2026-10-10T00:00:00Z", "end": "2026-10-09T00:00:00"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predictions/999/review", json={"status": "confirmed", "note": "inspection"}
        ).status_code
        == 404
    )


def test_outside_training_range_is_visible(client_and_repository):
    client, _ = client_and_repository
    response = client.post("/predict", json={**READING, "tool_wear_min": 1000})
    assert response.status_code == 200
    assert any("Tool wear" in warning for warning in response.json()["warnings"])
