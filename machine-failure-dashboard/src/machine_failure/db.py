"""MySQL schema and transactional prediction persistence."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.mysql import DATETIME, DOUBLE
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from machine_failure.config import PROJECT_ROOT, Settings, read_settings

TABLE_OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def utc_iso(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = TABLE_OPTIONS
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    dataset_sha256: Mapped[str] = mapped_column(String(64))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=utc_now)


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    __table_args__ = (Index("ix_readings_received_at", "received_at"), TABLE_OPTIONS)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    product_type: Mapped[str] = mapped_column(Enum("L", "M", "H"))
    air_temperature_k: Mapped[float] = mapped_column(DOUBLE)
    process_temperature_k: Mapped[float] = mapped_column(DOUBLE)
    rotational_speed_rpm: Mapped[float] = mapped_column(DOUBLE)
    torque_nm: Mapped[float] = mapped_column(DOUBLE)
    tool_wear_min: Mapped[float] = mapped_column(DOUBLE)
    source: Mapped[str] = mapped_column(String(16))
    received_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=utc_now)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("failure_score >= 0 AND failure_score <= 1", name="ck_prediction_score"),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_prediction_threshold"),
        Index("ix_predictions_model_date", "model_version_id", "created_at"),
        TABLE_OPTIONS,
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    failure_score: Mapped[float] = mapped_column(DOUBLE, nullable=False)
    threshold: Mapped[float] = mapped_column(DOUBLE, nullable=False)
    alert: Mapped[bool]
    latency_ms: Mapped[float] = mapped_column(DOUBLE)
    warnings: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=utc_now)


class InspectionReview(Base):
    __tablename__ = "inspection_reviews"
    __table_args__ = (UniqueConstraint("prediction_id", name="uq_review_prediction"), TABLE_OPTIONS)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    status: Mapped[str] = mapped_column(Enum("pending", "confirmed", "unconfirmed"))
    note: Mapped[str] = mapped_column(Text, default="")
    reviewed_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), default=utc_now)


def make_engine(settings: Settings, *, select_database: bool = True):
    if not settings.db_configured:
        raise ValueError("Set the MySQL account and password in the local .env file.")
    url = settings.database_url if select_database else settings.database_url.set(database=None)
    options = {"connect_timeout": 5, "read_timeout": 10, "write_timeout": 10}
    if settings.db_ssl_ca:
        options.update(ssl_ca=settings.db_ssl_ca, ssl_verify_cert=True, ssl_verify_identity=True)
    engine = create_engine(url, pool_pre_ping=True, connect_args=options)

    @event.listens_for(engine, "connect")
    def configure_utc(connection, _):
        with connection.cursor() as cursor:
            cursor.execute("SET time_zone = '+00:00'")

    return engine


def register_model(session: Session, metadata: dict) -> ModelVersion:
    model = session.scalar(
        select(ModelVersion).where(ModelVersion.version == metadata["model_version"])
    )
    if model is None:
        model = ModelVersion(
            version=metadata["model_version"],
            algorithm=metadata["algorithm"],
            dataset_sha256=metadata["dataset_sha256"],
            artifact_sha256=metadata["artifact_sha256"],
            metadata_json=metadata,
        )
        session.add(model)
        session.flush()
    elif model.artifact_sha256 != metadata["artifact_sha256"]:
        raise ValueError("Model registry conflicts with the frozen artifact.")
    return model


class MySQLRepository:
    def __init__(self, settings: Settings):
        self.engine = make_engine(settings) if settings.db_configured else None

    def ready(self) -> bool:
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                return set(Base.metadata.tables).issubset(inspect(connection).get_table_names())
        except Exception:
            return False

    def save_predictions(self, results: list[dict], metadata: dict, source: str) -> list[dict]:
        if self.engine is None:
            raise RuntimeError("MySQL is not configured.")
        saved = []
        with Session(self.engine) as session, session.begin():
            model = register_model(session, metadata)
            for result in results:
                reading = SensorReading(**result["readings"], source=source)
                session.add(reading)
                session.flush()
                prediction = Prediction(
                    reading_id=reading.id,
                    model_version_id=model.id,
                    failure_score=result["failure_score"],
                    threshold=result["threshold"],
                    alert=result["alert"],
                    latency_ms=result["latency_ms"],
                    warnings=result["warnings"],
                )
                session.add(prediction)
                session.flush()
                saved.append(
                    {
                        **result,
                        "prediction_id": prediction.id,
                        "reading_id": reading.id,
                        "received_at": utc_iso(reading.received_at),
                    }
                )
        return saved

    def history(
        self, *, limit: int, offset: int, alert=None, product_type=None, start=None, end=None
    ):
        if self.engine is None:
            raise RuntimeError("MySQL is not configured.")
        filters = []
        if alert is not None:
            filters.append(Prediction.alert == alert)
        if product_type:
            filters.append(SensorReading.product_type == product_type)
        if start:
            filters.append(
                SensorReading.received_at >= start.astimezone(UTC).replace(tzinfo=None)
                if start.tzinfo
                else SensorReading.received_at >= start
            )
        if end:
            filters.append(
                SensorReading.received_at <= end.astimezone(UTC).replace(tzinfo=None)
                if end.tzinfo
                else SensorReading.received_at <= end
            )
        query = (
            select(Prediction, SensorReading, ModelVersion, InspectionReview)
            .join(SensorReading, Prediction.reading_id == SensorReading.id)
            .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
            .outerjoin(InspectionReview, InspectionReview.prediction_id == Prediction.id)
            .where(*filters)
            .order_by(Prediction.id.desc())
            .limit(limit)
            .offset(offset)
        )
        count_query = (
            select(func.count(Prediction.id))
            .join(SensorReading, Prediction.reading_id == SensorReading.id)
            .where(*filters)
        )
        with Session(self.engine) as session:
            total = session.scalar(count_query)
            items = []
            for prediction, reading, model, review in session.execute(query):
                items.append(
                    {
                        "prediction_id": prediction.id,
                        "product_type": reading.product_type,
                        "failure_score": prediction.failure_score,
                        "threshold": prediction.threshold,
                        "alert": prediction.alert,
                        "model_version": model.version,
                        "latency_ms": prediction.latency_ms,
                        "source": reading.source,
                        "received_at": utc_iso(reading.received_at),
                        "warnings": prediction.warnings,
                        "review_status": review.status if review else "pending",
                        "review_note": review.note if review else "",
                    }
                )
            return {"total": total, "items": items, "limit": limit, "offset": offset}

    def review(self, prediction_id: int, status: str, note: str) -> dict:
        if self.engine is None:
            raise RuntimeError("MySQL is not configured.")
        with Session(self.engine) as session, session.begin():
            if session.get(Prediction, prediction_id) is None:
                raise LookupError("Prediction does not exist.")
            review = session.scalar(
                select(InspectionReview).where(InspectionReview.prediction_id == prediction_id)
            )
            if review is None:
                review = InspectionReview(prediction_id=prediction_id, status=status, note=note)
                session.add(review)
            review.status, review.note, review.reviewed_at = status, note, utc_now()
            session.flush()
            return {
                "prediction_id": prediction_id,
                "status": status,
                "note": note,
                "reviewed_at": utc_iso(review.reviewed_at),
            }

    def dispose(self):
        if self.engine is not None:
            self.engine.dispose()


def initialize(settings: Settings) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", settings.db_name):
        raise ValueError("Database name must use letters, digits, and underscores.")
    server = make_engine(settings, select_database=False)
    with server.begin() as connection:
        connection.exec_driver_sql(
            f"CREATE DATABASE IF NOT EXISTS `{settings.db_name}` CHARACTER SET utf8mb4"
        )
    server.dispose()
    from alembic import command
    from alembic.config import Config

    config = Config(str(settings.root / "alembic.ini"))
    config.set_main_option("script_location", str(settings.root / "migrations"))
    config.attributes["settings"] = settings
    command.upgrade(config, "head")
    engine = make_engine(settings)
    metadata = json.loads((settings.root / "artifacts/model_metadata.json").read_text("utf-8"))
    with Session(engine) as session, session.begin():
        register_model(session, metadata)
    engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize", "check"])
    args = parser.parse_args()
    settings = read_settings(PROJECT_ROOT)
    if args.command == "initialize":
        try:
            initialize(settings)
        except Exception:
            raise SystemExit(
                "MySQL initialization failed. Check the local .env account, its permissions, "
                "and the running server; no credentials have been printed."
            ) from None
        print("Initialized the dedicated MySQL database, migrations, and model registry.")
    else:
        repository = MySQLRepository(settings)
        try:
            if not repository.ready():
                raise SystemExit("MySQL account, database, or migrations are not ready.")
            print("MySQL connection and application schema verified.")
        finally:
            repository.dispose()


if __name__ == "__main__":
    main()
