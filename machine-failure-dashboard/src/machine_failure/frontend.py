"""Dashboard HTTP client, strict CSV validation, and export helpers."""

import csv
import io
import os
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from machine_failure.config import PROJECT_ROOT
from machine_failure.contracts import SensorInput

INPUT_FIELDS = tuple(SensorInput.model_fields)
MAX_CSV_BYTES = 1_000_000
MAX_BATCH_ROWS = 1000
FIELD_LABELS = {
    "product_type": "Product type",
    "air_temperature_k": "Air temperature (K)",
    "process_temperature_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
}


class CsvValidationError(ValueError):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []


class ApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class DashboardClient:
    base_url: str
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/"), os.getenv("DEMO_API_KEY", "")
        )

    def request(self, method, route, *, allow_unhealthy=False, **kwargs):
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            response = requests.request(
                method, self.base_url + route, headers=headers, timeout=(3, 30), **kwargs
            )
        except requests.Timeout:
            raise ApiError(
                "The prediction service timed out. Check history before retrying a save."
            ) from None
        except requests.RequestException:
            raise ApiError(
                "Cannot reach the prediction service. Start the API and refresh."
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise ApiError("The prediction service returned an unreadable response.") from None
        if not isinstance(payload, dict):
            raise ApiError("The prediction service returned an unexpected response.")
        if not response.ok and not (allow_unhealthy and response.status_code == 503):
            if response.status_code == 401:
                raise ApiError("API access was denied. Check the configured demo API key.")
            detail = payload.get("detail", "The request could not be completed.")
            if isinstance(detail, list):
                detail = "; ".join(item.get("message", "Invalid input") for item in detail[:5])
            raise ApiError(str(detail))
        return payload

    def health(self):
        return self.request("GET", "/health", allow_unhealthy=True)

    def model_info(self):
        return self.request("GET", "/model-info")

    def predict(self, reading):
        validated = SensorInput.model_validate(reading).model_dump()
        return self.request("POST", "/predict", json=validated)

    def predict_batch(self, readings):
        return self.request("POST", "/predict-batch", json={"readings": readings})

    def history(self, **filters):
        return self.request("GET", "/predictions", params=filters)

    def review(self, prediction_id, status, note):
        return self.request(
            "POST", f"/predictions/{prediction_id}/review", json={"status": status, "note": note}
        )


def parse_sensor_csv(data: bytes) -> list[dict]:
    if len(data) > MAX_CSV_BYTES:
        raise CsvValidationError("Upload a CSV no larger than 1 MB (1,000,000 bytes).")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CsvValidationError("Save the CSV using UTF-8 encoding.") from None
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        if len(set(header)) != len(header):
            raise CsvValidationError("CSV headers must be unique.")
        missing = sorted(set(INPUT_FIELDS) - set(header))
        extra = sorted(set(header) - set(INPUT_FIELDS))
        if missing or extra:
            details = []
            if missing:
                details.append("Missing columns: " + ", ".join(missing))
            if extra:
                details.append("Unexpected columns: " + ", ".join(extra))
            raise CsvValidationError(". ".join(details) + ". Use the six-column template.")
        readings, errors, row_count = [], [], 0
        for values in reader:
            if not values:
                continue
            row_count += 1
            if row_count > MAX_BATCH_ROWS:
                raise CsvValidationError("A batch can contain at most 1,000 readings.")
            if len(values) != len(header):
                errors.append(
                    {"CSV row": reader.line_num, "Field": "row", "Problem": "Expected six values."}
                )
                continue
            raw = dict(zip(header, values, strict=True))
            try:
                readings.append(SensorInput.model_validate(raw).model_dump())
            except ValidationError as exc:
                for error in exc.errors():
                    errors.append(
                        {
                            "CSV row": reader.line_num,
                            "Field": FIELD_LABELS.get(error["loc"][0], error["loc"][0]),
                            "Problem": error["msg"],
                        }
                    )
        if not row_count:
            raise CsvValidationError("The CSV needs at least one reading below its header.")
        if errors:
            raise CsvValidationError(
                "Fix the listed rows before submitting. Nothing was saved.", errors
            )
        return readings
    except csv.Error:
        raise CsvValidationError(
            "The CSV is malformed. Check delimiters and quotation marks."
        ) from None


def template_csv() -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(INPUT_FIELDS)
    writer.writerow(["L", 300, 310, 1500, 40, 100])
    writer.writerow(["M", 301, 311, 1400, 55, 180])
    return output.getvalue()


def results_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        row = {"prediction_id": item["prediction_id"], **item.get("readings", {})}
        row.update(
            {
                key: item[key]
                for key in ("failure_score", "threshold", "alert", "model_version", "received_at")
            }
        )
        row["warnings"] = "; ".join(item.get("warnings", []))
        rows.append(row)
    return rows


def results_csv(items: list[dict]) -> str:
    rows = results_rows(items)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
