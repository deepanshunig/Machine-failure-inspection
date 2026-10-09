import csv
import io

import pytest
import requests

from machine_failure.frontend import (
    MAX_CSV_BYTES,
    ApiError,
    CsvValidationError,
    DashboardClient,
    parse_sensor_csv,
    results_csv,
    template_csv,
)


def test_template_can_be_uploaded_with_utf8_bom_and_numeric_values():
    rows = parse_sensor_csv(b"\xef\xbb\xbf" + template_csv().encode("utf-8"))
    assert len(rows) == 2
    assert rows[0]["product_type"] == "L"
    assert rows[0]["rotational_speed_rpm"] == 1500.0


@pytest.mark.parametrize("column", ["Machine failure", "TWF", "UDI"])
def test_csv_rejects_outcome_and_identifier_columns(column):
    content = template_csv().splitlines()
    content[0] += "," + column
    content[1] += ",0"
    content[2] += ",0"
    with pytest.raises(CsvValidationError, match="Unexpected columns"):
        parse_sensor_csv("\n".join(content).encode())


def test_invalid_later_csv_row_rejects_the_whole_batch_with_row_details():
    lines = template_csv().splitlines()
    lines[2] = lines[2].replace(",55,", ",NaN,")
    with pytest.raises(CsvValidationError) as caught:
        parse_sensor_csv("\n".join(lines).encode())
    assert caught.value.errors[0]["CSV row"] == 3
    assert caught.value.errors[0]["Field"] == "Torque (Nm)"


def test_csv_limits_and_duplicate_headers():
    header, first, _ = template_csv().splitlines()
    with pytest.raises(CsvValidationError, match="at most 1,000"):
        parse_sensor_csv((header + "\n" + (first + "\n") * 1001).encode())
    with pytest.raises(CsvValidationError, match="no larger than 1 MB"):
        parse_sensor_csv(b"x" * (MAX_CSV_BYTES + 1))
    with pytest.raises(CsvValidationError, match="unique"):
        parse_sensor_csv((header + ",product_type\n").encode())


def test_csv_missing_or_short_rows_and_empty_data():
    with pytest.raises(CsvValidationError, match="Missing columns"):
        parse_sensor_csv(b"product_type\nL\n")
    header = template_csv().splitlines()[0]
    with pytest.raises(CsvValidationError, match="at least one"):
        parse_sensor_csv(header.encode())
    with pytest.raises(CsvValidationError) as caught:
        parse_sensor_csv((header + "\nL,300,310\n").encode())
    assert caught.value.errors[0]["CSV row"] == 2


def test_export_retains_prediction_identity_decision_and_warnings():
    item = {
        "prediction_id": 12,
        "readings": parse_sensor_csv(template_csv().encode())[0],
        "failure_score": 0.6,
        "threshold": 0.3,
        "alert": True,
        "model_version": "frozen-test",
        "received_at": "2026-10-09T00:00:00Z",
        "warnings": ["Outside training range"],
    }
    row = next(csv.DictReader(io.StringIO(results_csv([item]))))
    assert row["prediction_id"] == "12"
    assert row["alert"] == "True"
    assert row["product_type"] == "L"
    assert float(row["failure_score"]) == 0.6
    assert float(row["threshold"]) == 0.3
    assert row["warnings"] == "Outside training range"


def test_unreachable_api_has_an_actionable_error_without_credentials(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("Internal private diagnostic")

    monkeypatch.setattr(requests, "request", fail)
    with pytest.raises(ApiError, match="Cannot reach") as caught:
        DashboardClient("http://127.0.0.1:1", api_key="secret-test-key").health()
    assert "secret" not in str(caught.value)
    assert "Internal" not in str(caught.value)
