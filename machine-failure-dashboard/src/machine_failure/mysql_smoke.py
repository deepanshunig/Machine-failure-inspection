"""Verify Step 4 against configured MySQL; retains two clearly marked demo predictions."""

import json

from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from machine_failure.api import create_app
from machine_failure.config import read_settings
from machine_failure.data import load_partition
from machine_failure.db import ModelVersion, MySQLRepository, Prediction, SensorReading
from machine_failure.inference import INPUT_MAPPING, ModelRuntime


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_check(settings):
    from fastapi.testclient import TestClient

    repository = MySQLRepository(settings)
    try:
        require(repository.ready(), "Initialize the MySQL schema before running the smoke check.")
        runtime = ModelRuntime(settings.root)
        frame, _ = load_partition(settings.root, "train")
        readings = [
            {key: row[field] for key, field in INPUT_MAPPING.items()}
            for row in frame.head(2).to_dict("records")
        ]
        offline = runtime.score(readings)
        headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
        with TestClient(create_app(settings, repository=repository)) as client:
            require(client.get("/health").status_code == 200, "Health check failed.")
            response = client.post("/predict-batch", json={"readings": readings}, headers=headers)
            require(response.status_code == 200, "Batch could not be saved.")
            saved = response.json()["items"]
            for expected, actual in zip(offline, saved, strict=True):
                require(
                    abs(expected["failure_score"] - actual["failure_score"]) < 1e-12,
                    "Offline and API scores differ.",
                )
                require(actual["threshold"] == expected["threshold"], "Threshold differs.")
            prediction_id = saved[0]["prediction_id"]
            for item in saved:
                review = client.post(
                    f"/predictions/{item['prediction_id']}/review",
                    json={"status": "pending", "note": "Step 4 synthetic smoke check"},
                    headers=headers,
                )
                require(review.status_code == 200, "Review write failed.")
            history = client.get(
                "/predictions",
                params={
                    "limit": 100,
                    "product_type": readings[0]["product_type"],
                    "start": saved[0]["received_at"],
                },
                headers=headers,
            )
            require(history.status_code == 200, "History filter failed.")
            require(
                any(
                    item["prediction_id"] == prediction_id
                    and item["review_note"] == "Step 4 synthetic smoke check"
                    for item in history.json()["items"]
                ),
                "Joined review was not retrieved.",
            )

        # A two-row transaction with a deliberately invalid second score must roll back both.
        with Session(repository.engine) as session:
            before = session.scalar(
                select(func.count(SensorReading.id)).where(SensorReading.source == "rollback_check")
            )
            server_version = session.scalar(text("SELECT VERSION()"))
        try:
            repository.save_predictions(
                [offline[0], {**offline[1], "failure_score": 1.5}],
                runtime.metadata,
                "rollback_check",
            )
        except DBAPIError as exc:
            require(exc.orig.args[0] == 3819, "Expected a MySQL CHECK constraint rejection.")
        else:
            raise RuntimeError("MySQL did not enforce the score constraint.")
        with Session(repository.engine) as session:
            after = session.scalar(
                select(func.count(SensorReading.id)).where(SensorReading.source == "rollback_check")
            )
            model_id = session.scalar(
                select(ModelVersion.id).where(
                    ModelVersion.version == runtime.metadata["model_version"]
                )
            )
            nonexistent_id = session.scalar(select(func.max(SensorReading.id))) + 1000
        require(before == after, "The failed batch did not roll back both readings.")

        try:
            with Session(repository.engine) as session, session.begin():
                session.add(
                    Prediction(
                        reading_id=nonexistent_id,
                        model_version_id=model_id,
                        failure_score=0.5,
                        threshold=runtime.metadata["threshold"],
                        alert=True,
                        latency_ms=0.0,
                        warnings=[],
                    )
                )
                session.flush()
        except DBAPIError as exc:
            require(exc.orig.args[0] == 1452, "Expected a MySQL foreign-key rejection.")
        else:
            raise RuntimeError("MySQL did not enforce the reading foreign key.")

        report = {
            "status": "passed",
            "mysql_version": server_version,
            "model_version": runtime.metadata["model_version"],
            "saved_demo_prediction_ids": [row["prediction_id"] for row in saved],
            "verified": [
                "health",
                "API/offline score equality",
                "batch persistence",
                "joined history and receipt-date/product filters",
                "inspection review",
                "score CHECK constraint",
                "foreign key",
                "failed-batch rollback",
            ],
        }
        (settings.root / "reports/mysql_smoke.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        repository.dispose()


def main():
    settings = read_settings()
    if not settings.db_configured:
        raise SystemExit("Set the dedicated MySQL account and password in local .env first.")
    try:
        report = run_check(settings)
    except Exception:
        raise SystemExit(
            "MySQL smoke check did not pass. Check the account, migrations, and MySQL "
            "constraint support. Step 4 remains pending; no credentials have been printed."
        ) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
