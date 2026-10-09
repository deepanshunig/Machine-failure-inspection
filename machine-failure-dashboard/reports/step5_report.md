# Step 5: frontend — complete

The Streamlit dashboard is running at **http://127.0.0.1:8502**, with FastAPI at **http://127.0.0.1:8001**. Port 8501 was occupied. All operational requests go through the API; the frontend has no direct MySQL connection.

## Delivered screens

1. Overview: actual history counts and recent snapshots, alongside a separately labelled frozen benchmark.
2. Single prediction: six input fields with units, persisted result, score, threshold, decision, model version, latency and range warnings.
3. Batch upload: template download, strict CSV validation, input preview, atomic API batch save, result table and CSV download.
4. History and reviews: decision/product/date filters, pagination, inspection status/note creation and updates.
5. Model evaluation: all saved metrics including F1 and per-class/macro/weighted reports; precision–recall and confusion plots; errors and JSON download.
6. Data explorer: schema findings, split counts, training-only plots and feature preview.

## Acceptance evidence

| Check | Result |
|---|---|
| Browser single prediction, including an out-of-range wear value | Saved prediction #11; score 0.8402; warning visible |
| Valid two-row CSV uploaded and scored through browser | Predictions #12 and #13 saved in MySQL |
| Downloaded results CSV | Actual file read and checked for both IDs, frozen model/threshold and result fields |
| Invalid CSV with `NaN` torque in row 3 | Row/field error displayed, save control absent, history count unchanged at 10 |
| Browser inspection review | Prediction #13 updated to `unconfirmed` with a clearly labelled synthetic-demo note; retrieved from MySQL |
| Combined history filters | Inspection suggestions + product L + UTC date range returned the two expected snapshots |
| Evaluation UI | F1 0.7671 and both classes/macro/weighted results match the frozen report; both charts loaded |
| Narrow browser layout | Sidebar collapses, columns stack, form/table controls remain available; header spacing corrected |
| Offline interaction test | Save disabled; saved evaluation still available |
| Automated tests | 34 passed |
| Ruff | Passed |

Machine-readable evidence: `frontend_smoke.json`. Browser screenshots are in `screenshots/`. The model artifact checksum remains unchanged.

## Input and error handling

The API and frontend CSV validator share `contracts.py`. UTF-8 CSVs accept exactly six input columns. The frontend rejects duplicate/missing/extra headers, wrong row widths, invalid product types, negative/nonpositive values according to the contract, non-finite values, files over 1,000,000 bytes, and batches over 1,000 readings. Invalid CSVs never reach the save endpoint.

Timeout errors advise checking history before retrying because a committed server-side save may outlast a client timeout. Connection errors show an actionable message and disable saving. Reviews and application logs are separate from official benchmark labels.

## Run command

From the project directory, with the API running:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.address 127.0.0.1 --server.port 8502
```

The installed/tested Streamlit release is 1.65.0. No cloud deployment was performed. The remaining planned milestone is Step 6: packaging, CI and portfolio documentation.
