# Step 4: MySQL and API — complete

Verified against the installed MySQL **9.7.0** server at `127.0.0.1:3306`, using the dedicated `machine_failure_dashboard` database and a separate project account. The application account's credentials are stored only in local `.env`, which is ignored by Git. The runtime uses the project account.

## Database evidence

- Alembic revision `0001` applied. `alembic check` reports no new upgrade operations.
- Tables: `model_versions`, `sensor_readings`, `predictions`, `inspection_reviews`.
- All application tables use InnoDB and utf8mb4.
- Reading and prediction writes commit together. An invalid second score rolls back the complete batch.
- MySQL rejects scores outside `[0,1]` and predictions referencing a nonexistent reading.
- All five statements in `sql/inspection_queries.sql` executed successfully.
- The model registry contains the frozen version `ai4i-v1-6ba56612ca`, including the added F1/classification report.

The CHECK rejection is tested by MySQL error code 3819 through SQLAlchemy's `DBAPIError`; the driver categorizes this error as an operational exception. Foreign-key rejection is checked by error code 1452.

## API evidence

The API is running locally at `http://127.0.0.1:8001`; port 8000 was already occupied. The same URL is saved as `API_URL` in `.env`.

| Behaviour | Result |
|---|---|
| Health, OpenAPI and interactive documentation | Passed |
| Model information, including F1 and classification report | Passed |
| Single and batch scores match the frozen offline pipeline | Passed |
| Predictions saved in real MySQL | Passed |
| Product type, alert and UTC receipt-date history filters | Passed |
| Inspection review creation and update; joined retrieval | Passed |
| Invalid batch rejected before any write | Passed |
| Score constraint, foreign key and failed-batch rollback | Passed |
| Regression tests | 22 passed |
| Ruff | Passed |

Machine-readable evidence: `mysql_smoke.json` and `api_http_smoke.json`. Smoke runs retain synthetic demo snapshots; retries also produced demo records. These application logs are separate from the official held-out evaluation.

## Commands

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe -m machine_failure.db check
.\.venv\Scripts\python.exe -m machine_failure.mysql_smoke
.\.venv\Scripts\python.exe -m alembic check
```

Start the API after a restart, if port 8001 is free:

```powershell
.\.venv\Scripts\python.exe -m uvicorn machine_failure.api:app --host 127.0.0.1 --port 8001
```

Next milestone: Step 5, the Streamlit frontend and end-to-end interface checks.
