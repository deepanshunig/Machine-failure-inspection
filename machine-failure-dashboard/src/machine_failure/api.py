"""Validated prediction endpoints with transactional MySQL persistence."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from secrets import compare_digest
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from machine_failure.config import Settings, read_settings
from machine_failure.contracts import BatchInput, ReviewInput, SensorInput
from machine_failure.db import MySQLRepository
from machine_failure.inference import ModelRuntime


def create_app(settings: Settings | None = None, repository=None) -> FastAPI:
    settings = settings or read_settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = ModelRuntime(settings.root)
        app.state.repository = repository or MySQLRepository(settings)
        try:
            yield
        finally:
            app.state.repository.dispose()

    app = FastAPI(title="Machine Failure Inspection API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc):
        # Invalid inputs such as NaN are not valid JSON output values. Do not echo them.
        details = [
            {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    def authorize(x_api_key: str | None = Header(default=None)):
        if settings.api_key and not compare_digest(
            (x_api_key or "").encode("utf-8"), settings.api_key.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="A valid API key is required.")

    def persist(readings, source):
        try:
            results = app.state.runtime.score([row.model_dump() for row in readings])
        except (ValueError, OverflowError, FloatingPointError):
            raise HTTPException(
                status_code=422, detail="Check reading units and numeric magnitudes."
            )
        try:
            return app.state.repository.save_predictions(
                results, app.state.runtime.metadata, source
            )
        except Exception:
            raise HTTPException(status_code=503, detail="Prediction could not be saved to MySQL.")

    @app.get("/health")
    def health():
        ready = app.state.repository.ready()
        content = {
            "model_ready": True,
            "database_ready": ready,
            "model_version": app.state.runtime.metadata["model_version"],
        }
        return JSONResponse(status_code=200 if ready else 503, content=content)

    @app.get("/model-info", dependencies=[Depends(authorize)])
    def model_info():
        return app.state.runtime.metadata

    @app.post("/predict", dependencies=[Depends(authorize)])
    def predict(readings: SensorInput):
        return persist([readings], "manual")[0]

    @app.post("/predict-batch", dependencies=[Depends(authorize)])
    def predict_batch(batch: BatchInput):
        return {"count": len(batch.readings), "items": persist(batch.readings, "csv")}

    @app.get("/predictions", dependencies=[Depends(authorize)])
    def history(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        alert: bool | None = None,
        product_type: Literal["L", "M", "H"] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ):
        if start and end:
            normalized_start = start.replace(tzinfo=UTC) if not start.tzinfo else start
            normalized_end = end.replace(tzinfo=UTC) if not end.tzinfo else end
            if normalized_start > normalized_end:
                raise HTTPException(status_code=422, detail="Start must precede end.")
        try:
            return app.state.repository.history(
                limit=limit,
                offset=offset,
                alert=alert,
                product_type=product_type,
                start=start,
                end=end,
            )
        except Exception:
            raise HTTPException(status_code=503, detail="Prediction history is unavailable.")

    @app.post("/predictions/{prediction_id}/review", dependencies=[Depends(authorize)])
    def review(prediction_id: int, body: ReviewInput):
        try:
            return app.state.repository.review(prediction_id, body.status, body.note)
        except LookupError:
            raise HTTPException(status_code=404, detail="Prediction does not exist.")
        except Exception:
            raise HTTPException(status_code=503, detail="Inspection review could not be saved.")

    return app


app = create_app()
