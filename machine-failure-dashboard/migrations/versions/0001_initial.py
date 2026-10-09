"""Initial MySQL prediction and review tables."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import DATETIME, DOUBLE

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None
OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}


def upgrade():
    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("version", sa.String(64), nullable=False, unique=True),
        sa.Column("algorithm", sa.String(64), nullable=False),
        sa.Column("dataset_sha256", sa.String(64), nullable=False),
        sa.Column("artifact_sha256", sa.String(64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", DATETIME(fsp=6), nullable=False),
        **OPTIONS,
    )
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("product_type", sa.Enum("L", "M", "H"), nullable=False),
        *[
            sa.Column(name, DOUBLE(), nullable=False)
            for name in (
                "air_temperature_k",
                "process_temperature_k",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",
            )
        ],
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("received_at", DATETIME(fsp=6), nullable=False),
        **OPTIONS,
    )
    op.create_index("ix_readings_received_at", "sensor_readings", ["received_at"])
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "reading_id", sa.BigInteger(), sa.ForeignKey("sensor_readings.id"), nullable=False
        ),
        sa.Column(
            "model_version_id", sa.BigInteger(), sa.ForeignKey("model_versions.id"), nullable=False
        ),
        sa.Column("failure_score", DOUBLE(), nullable=False),
        sa.Column("threshold", DOUBLE(), nullable=False),
        sa.Column("alert", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", DOUBLE(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("created_at", DATETIME(fsp=6), nullable=False),
        sa.CheckConstraint("failure_score >= 0 AND failure_score <= 1", name="ck_prediction_score"),
        sa.CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_prediction_threshold"),
        **OPTIONS,
    )
    op.create_index("ix_predictions_model_date", "predictions", ["model_version_id", "created_at"])
    op.create_table(
        "inspection_reviews",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "prediction_id", sa.BigInteger(), sa.ForeignKey("predictions.id"), nullable=False
        ),
        sa.Column("status", sa.Enum("pending", "confirmed", "unconfirmed"), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("reviewed_at", DATETIME(fsp=6), nullable=False),
        sa.UniqueConstraint("prediction_id", name="uq_review_prediction"),
        **OPTIONS,
    )


def downgrade():
    for name in ("inspection_reviews", "predictions", "sensor_readings", "model_versions"):
        op.drop_table(name)
