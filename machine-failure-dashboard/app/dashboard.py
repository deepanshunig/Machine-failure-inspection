"""Streamlit inspection dashboard; all operational reads and writes go through the API."""

import hashlib
import json
from datetime import UTC, datetime, time, timedelta

import pandas as pd
import streamlit as st

from machine_failure.config import PROJECT_ROOT
from machine_failure.data import load_partition
from machine_failure.frontend import (
    FIELD_LABELS,
    ApiError,
    CsvValidationError,
    DashboardClient,
    parse_sensor_csv,
    results_csv,
    results_rows,
    template_csv,
)

PAGES = [
    "Overview",
    "Single prediction",
    "Batch upload",
    "History & reviews",
    "Model evaluation",
    "Data explorer",
]
st.set_page_config(page_title="Inspection Lab | Machine Failure", page_icon="⚙️", layout="wide")
st.markdown(
    """
<style>
.block-container { padding-top:4.8rem; padding-bottom:3rem; max-width:1400px; }
h1 { letter-spacing:-.035em; font-weight:750 !important; }
h2,h3 { letter-spacing:-.025em; }
[data-testid="stSidebar"] { border-right:1px solid #dce5ef; }
[data-testid="stMetric"] { background:white; border:1px solid #dce5ef; border-radius:14px;
padding:18px 20px; }
[data-testid="stMetricLabel"] { color:#516582; }
[data-testid="stForm"] { background:white; border-radius:14px; padding:22px; }
.eyebrow { font-size:11px; font-weight:700; letter-spacing:.16em; color:#087f8c;
margin-bottom:10px; }
.brand { font-size:22px; font-weight:750; letter-spacing:-.04em; margin-bottom:2px; }
.brand-sub { color:#516582; font-size:12px; margin-bottom:26px; }
</style>
""",
    unsafe_allow_html=True,
)


def read_report(name):
    return json.loads((PROJECT_ROOT / "reports" / name).read_text(encoding="utf-8"))


def metric_cards(values):
    for column, (label, value) in zip(st.columns(len(values)), values, strict=True):
        column.metric(label, value)


def table(rows, **kwargs):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", **kwargs)


def page_header(title, description):
    st.markdown(
        '<div class="eyebrow">INSPECTION LAB / MACHINE FAILURE</div>', unsafe_allow_html=True
    )
    st.title(title)
    st.caption(description)


def show_result(result):
    st.subheader("Saved prediction")
    (st.warning if result["alert"] else st.success)(result["decision"])
    metric_cards(
        [
            ("Failure score", f"{result['failure_score']:.2%}"),
            ("Alert threshold", f"{result['threshold']:.2%}"),
            ("Prediction ID", f"#{result['prediction_id']}"),
        ]
    )
    st.caption(
        f"Model {result['model_version']} · saved {result['received_at']} · "
        f"inference {result['latency_ms']:.1f} ms"
    )
    st.caption(
        "The score is shown on a 0–100% scale for this sensor snapshot. "
        "It is not a calibrated failure probability."
    )
    for warning in result["warnings"]:
        st.warning(warning)
    with st.expander("Submitted reading"):
        table([{FIELD_LABELS[key]: value for key, value in result["readings"].items()}])


def overview(client, ready, report):
    page_header(
        "Machine failure inspection",
        "Evaluate a sensor snapshot, log an inspection "
        "suggestion, and review the result. Synthetic-data portfolio prototype.",
    )
    st.subheader("Prediction activity")
    if ready:
        try:
            recent = client.history(limit=5)
            alerts = client.history(limit=1, alert=True)
            metric_cards(
                [
                    ("Logged snapshots", f"{recent['total']:,}"),
                    ("Inspection alerts", f"{alerts['total']:,}"),
                    ("Storage", "MySQL connected"),
                ]
            )
            st.caption("Counts come from application history, including synthetic demo entries.")
            if recent["items"]:
                st.subheader("Recent predictions")
                table(
                    [
                        {
                            "ID": row["prediction_id"],
                            "Product": row["product_type"],
                            "Failure score": f"{row['failure_score']:.2%}",
                            "Decision": "Inspect" if row["alert"] else "Below threshold",
                            "Review": row["review_status"],
                            "Received (UTC)": row["received_at"],
                        }
                        for row in recent["items"]
                    ]
                )
            else:
                st.info("No predictions saved yet. Open Single prediction to create the first one.")
        except ApiError as exc:
            st.error(str(exc))
    else:
        st.info("Prediction activity is unavailable until the model and database are connected.")
    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("From reading to review")
        st.markdown(
            "1. Enter six product and sensor values, or upload a CSV.\n"
            "2. Run the frozen model and save the result.\n"
            "3. Review inspection suggestions in History & reviews."
        )
        st.caption("All temperatures use Kelvin. The threshold stays fixed across predictions.")
    with right:
        st.subheader("Frozen benchmark")
        metrics = report["test"]
        metric_cards(
            [
                ("Average precision", f"{metrics['average_precision']:.3f}"),
                ("Failure F1", f"{metrics['f1']:.3f}"),
            ]
        )
        st.caption(
            f"{metrics['records']:,} held-out snapshots · {metrics['failures']} labelled "
            "failures. See Model evaluation for the complete report."
        )


def single_prediction(client, ready, metadata):
    page_header(
        "Single prediction",
        "Enter a sensor snapshot. The result is saved to MySQL after a successful prediction.",
    )
    main, aside = st.columns([2, 1], gap="large")
    with main:
        with st.form("single_prediction_form"):
            a, b = st.columns(2)
            with a:
                product = st.selectbox("Product type", ["L", "M", "H"], key="product_type")
                air = st.number_input(
                    "Air temperature (K)",
                    min_value=0.01,
                    value=300.0,
                    step=0.1,
                    key="air_temperature_k",
                )
                speed = st.number_input(
                    "Rotational speed (rpm)",
                    min_value=0.01,
                    value=1500.0,
                    step=10.0,
                    key="rotational_speed_rpm",
                )
            with b:
                wear = st.number_input(
                    "Tool wear (min)", min_value=0.0, value=100.0, step=1.0, key="tool_wear_min"
                )
                process = st.number_input(
                    "Process temperature (K)",
                    min_value=0.01,
                    value=310.0,
                    step=0.1,
                    key="process_temperature_k",
                )
                torque = st.number_input(
                    "Torque (Nm)", min_value=0.0, value=40.0, step=0.1, key="torque_nm"
                )
            submitted = st.form_submit_button(
                "Run prediction", type="primary", disabled=not ready, width="stretch"
            )
        if submitted:
            st.session_state.pop("single_result", None)
            try:
                with st.spinner("Scoring and saving the reading…"):
                    st.session_state["single_result"] = client.predict(
                        {
                            "product_type": product,
                            "air_temperature_k": air,
                            "process_temperature_k": process,
                            "rotational_speed_rpm": speed,
                            "torque_nm": torque,
                            "tool_wear_min": wear,
                        }
                    )
            except ApiError as exc:
                st.error(str(exc))
        if result := st.session_state.get("single_result"):
            show_result(result)
    with aside:
        with st.container(border=True):
            st.subheader("Reading guide")
            st.write(
                "L, M and H are product-quality categories. Provide the current sensor "
                "values in the units shown."
            )
            st.write(
                "A score at or above the alert threshold suggests an inspection. A lower "
                "score can still miss a failure."
            )
            st.caption("Values outside the observed training range are flagged in the result.")
        with st.expander("Observed training ranges"):
            table(
                [
                    {"Input": key, "Minimum": bounds["min"], "Maximum": bounds["max"]}
                    for key, bounds in metadata["training_ranges"].items()
                ]
            )


def batch_upload(client, ready):
    page_header(
        "Batch upload",
        "Validate every reading, then score and save the entire batch. "
        "Maximum 1,000 rows and 1 MB per file.",
    )
    st.download_button(
        "Download CSV template",
        template_csv(),
        "sensor_template.csv",
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
    )
    uploaded = st.file_uploader(
        "Upload sensor readings",
        type=["csv"],
        max_upload_size=1,
        key="sensor_csv",
        help="UTF-8 CSV with exactly the six template columns.",
    )
    if uploaded is None:
        st.info(
            "Start with the template. Add your readings below the header, then upload the file."
        )
        return
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != st.session_state.get("batch_digest"):
        st.session_state.pop("batch_result", None)
        st.session_state["batch_digest"] = digest
    try:
        readings = parse_sensor_csv(data)
    except CsvValidationError as exc:
        st.error(str(exc))
        if exc.errors:
            table(exc.errors[:100])
            if len(exc.errors) > 100:
                st.caption(f"Showing the first 100 of {len(exc.errors)} validation errors.")
        return
    st.success(f"{len(readings):,} readings passed validation. Ready to submit.")
    with st.expander("Preview validated readings", expanded=True):
        table(readings[:10])
        if len(readings) > 10:
            st.caption("Showing the first 10 readings.")
    if st.button("Score and save batch", type="primary", disabled=not ready, key="save_batch"):
        st.session_state.pop("batch_result", None)
        try:
            with st.spinner("Scoring and saving the complete batch…"):
                st.session_state["batch_result"] = client.predict_batch(readings)
        except ApiError as exc:
            st.error(str(exc))
    if result := st.session_state.get("batch_result"):
        items = result["items"]
        st.subheader("Saved batch results")
        metric_cards(
            [
                ("Saved snapshots", f"{result['count']:,}"),
                ("Inspection alerts", str(sum(item["alert"] for item in items))),
                ("With range warnings", str(sum(bool(item["warnings"]) for item in items))),
            ]
        )
        table(
            [
                {
                    **row,
                    "failure_score": f"{row['failure_score']:.2%}",
                    "threshold": f"{row['threshold']:.2%}",
                }
                for row in results_rows(items)
            ]
        )
        st.caption("Scores use a 0–100% scale; CSV exports retain the original 0–1 values.")
        st.download_button(
            "Download prediction results",
            results_csv(items),
            "prediction_results.csv",
            "text/csv",
            on_click="ignore",
            icon=":material/download:",
        )


def history_reviews(client, ready):
    page_header(
        "History & reviews",
        "Filter saved snapshots and add an inspection note. "
        "Receipt dates are application timestamps in UTC.",
    )
    if not ready:
        st.info("Connect the API and MySQL database to load prediction history.")
        return
    today = datetime.now(UTC).date()
    with st.form("history_filters"):
        a, b, c, d = st.columns([1.2, 1, 1, 1])
        decision = a.selectbox(
            "Decision", ["All decisions", "Inspection suggested", "Below threshold"]
        )
        product = b.selectbox("Product type", ["All types", "L", "M", "H"], key="history_product")
        start = c.date_input("From (UTC)", value=today - timedelta(days=7))
        end = d.date_input("Through (UTC)", value=today)
        use_dates = st.checkbox("Apply receipt-date range", value=False)
        applied = st.form_submit_button("Apply filters", type="primary")
    if applied:
        if use_dates and start > end:
            st.error("The start date must be on or before the end date.")
            return
        filters = {}
        if decision != "All decisions":
            filters["alert"] = decision == "Inspection suggested"
        if product != "All types":
            filters["product_type"] = product
        if use_dates:
            filters["start"] = datetime.combine(start, time.min, UTC).isoformat()
            filters["end"] = datetime.combine(end, time.max, UTC).isoformat()
        st.session_state["history_filters_value"] = filters
        st.session_state["history_offset"] = 0
    offset = st.session_state.get("history_offset", 0)
    try:
        history = client.history(
            limit=25, offset=offset, **st.session_state.get("history_filters_value", {})
        )
    except ApiError as exc:
        st.error(str(exc))
        return
    if message := st.session_state.pop("review_feedback", None):
        st.success(message)
    items = history["items"]
    st.caption(f"{history['total']:,} matching snapshots · page {offset // 25 + 1}")
    if not items:
        st.info("No predictions match these filters.")
        return
    table(
        [
            {
                "ID": row["prediction_id"],
                "Product": row["product_type"],
                "Failure score": f"{row['failure_score']:.2%}",
                "Threshold": f"{row['threshold']:.2%}",
                "Decision": "Inspect" if row["alert"] else "Below threshold",
                "Review": row["review_status"],
                "Note": row["review_note"],
                "Received (UTC)": row["received_at"],
                "Model": row["model_version"],
            }
            for row in items
        ]
    )
    previous, next_col, _ = st.columns([1, 1, 5])
    if previous.button("Previous", disabled=offset == 0):
        st.session_state["history_offset"] = max(0, offset - 25)
        st.rerun()
    if next_col.button("Next", disabled=offset + len(items) >= history["total"]):
        st.session_state["history_offset"] = offset + 25
        st.rerun()
    st.divider()
    st.subheader("Inspection review")
    by_id = {row["prediction_id"]: row for row in items}
    selected = st.selectbox(
        "Prediction to review",
        list(by_id),
        format_func=lambda value: (
            f"#{value} · {by_id[value]['product_type']} · "
            f"{'Inspect' if by_id[value]['alert'] else 'Below threshold'}"
        ),
        key="review_prediction_id",
    )
    row = by_id[selected]
    statuses = ["pending", "confirmed", "unconfirmed"]
    with st.form(f"review_form_{selected}"):
        status = st.selectbox(
            "Review status",
            statuses,
            index=statuses.index(row["review_status"]),
            key=f"review_status_{selected}",
        )
        note = st.text_area(
            "Inspection note",
            value=row["review_note"],
            max_chars=2000,
            key=f"review_note_{selected}",
            placeholder="What did the inspection find?",
        )
        st.caption(
            "Reviews record user feedback. They do not update the benchmark labels or model."
        )
        saved = st.form_submit_button("Save review", type="primary")
    if saved:
        try:
            client.review(selected, status, note)
            st.session_state["review_feedback"] = f"Review saved for prediction #{selected}."
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    with st.expander("Selected snapshot details"):
        st.write(f"Inference latency: {row['latency_ms']:.1f} ms · source: {row['source']}")
        for warning in row["warnings"]:
            st.warning(warning)


def model_evaluation(report):
    page_header(
        "Model evaluation",
        "Frozen results on 2,000 held-out synthetic snapshots. "
        "Application prediction counts do not change these metrics.",
    )
    metrics = report["test"]
    metric_cards(
        [
            ("Average precision", f"{metrics['average_precision']:.4f}"),
            ("Failure precision", f"{metrics['precision']:.2%}"),
            ("Failure recall", f"{metrics['recall']:.2%}"),
            ("Failure F1", f"{metrics['f1']:.4f}"),
        ]
    )
    st.caption(
        f"Model {report['model_version']} · validation-selected threshold "
        f"{metrics['threshold']:.2%} · 39 failures in the test partition"
    )
    a, b = st.columns(2)
    a.image(str(PROJECT_ROOT / "reports/figures/06_test_precision_recall.png"), width="stretch")
    b.image(str(PROJECT_ROOT / "reports/figures/07_test_confusion_matrix.png"), width="stretch")
    st.subheader("Classification report")
    details = metrics["classification_report"]
    table(
        [
            {
                "Class / average": label,
                "Precision": details[key]["precision"],
                "Recall": details[key]["recall"],
                "F1-score": details[key]["f1-score"],
                "Support": int(details[key]["support"]),
            }
            for key, label in [
                ("no_failure", "No failure (0)"),
                ("failure", "Failure (1)"),
                ("macro avg", "Macro average"),
                ("weighted avg", "Weighted average"),
            ]
        ]
    )
    metric_cards(
        [
            ("ROC-AUC", f"{metrics['roc_auc']:.4f}"),
            ("F2-score", f"{metrics['f2']:.4f}"),
            ("Accuracy", f"{metrics['accuracy']:.2%}"),
            ("False alerts / 1,000 normal", f"{metrics['false_alerts_per_1000_non_failures']:.2f}"),
        ]
    )
    st.subheader("Threshold and baseline comparison")
    table(
        [
            {
                "Evaluation": label,
                "Threshold": f"{report[key]['threshold']:.2%}",
                "Average precision": report[key]["average_precision"],
                "Precision": report[key]["precision"],
                "Recall": report[key]["recall"],
                "F1": report[key]["f1"],
            }
            for key, label in [
                ("validation", "Validation — selected threshold"),
                ("test", "Test — frozen threshold"),
                ("test_at_default_threshold", "Test — threshold 50%"),
                ("dummy_test", "Dummy baseline"),
            ]
        ]
    )
    st.info(
        "The test set has only 39 failures and comes from a synthetic source. These results "
        "do not establish performance on real machines or predict a future failure date."
    )
    with st.expander("Reviewed error examples"):
        st.markdown((PROJECT_ROOT / "reports/error_review.md").read_text(encoding="utf-8"))
    st.download_button(
        "Download complete evaluation JSON",
        json.dumps(report, indent=2),
        "final_evaluation.json",
        "application/json",
        on_click="ignore",
    )


@st.cache_data
def training_preview():
    features, _ = load_partition(PROJECT_ROOT, "train")
    return features.head(20)


def data_explorer():
    page_header(
        "Data explorer",
        "Official UCI AI4I 2020 synthetic dataset. Feature plots and "
        "the reading preview use training records only.",
    )
    quality = read_report("data_quality.json")
    metric_cards(
        [
            ("Dataset snapshots", "10,000"),
            ("Training snapshots", "6,000"),
            ("Missing values", str(quality["missing_values"])),
            ("Outcome-flag disagreements", str(quality["outcome_flag_target_disagreements"])),
        ]
    )
    table(
        [
            {
                "Partition": key.title(),
                "Records": value["records"],
                "Failures": value["failure"],
                "Failure rate": f"{value['failure_rate']:.2%}",
            }
            for key, value in quality["partitions"].items()
        ]
    )
    st.subheader("Training class balance")
    st.image(str(PROJECT_ROOT / "reports/figures/01_training_class_balance.png"), width="stretch")
    with st.expander("Training input distributions"):
        st.image(
            str(PROJECT_ROOT / "reports/figures/02_training_input_distributions.png"),
            width="stretch",
        )
    with st.expander("Training numeric correlations"):
        st.image(
            str(PROJECT_ROOT / "reports/figures/03_training_correlations.png"), width="stretch"
        )
    st.subheader("Training reading preview")
    st.dataframe(training_preview(), hide_index=True, width="stretch")
    st.caption(
        "The six inputs exclude identifiers, the target, and all failure-mode outcome flags. "
        "Original row order defines the 60/20/20 split; it is not a verified real timestamp."
    )
    st.markdown(
        "[UCI dataset and attribution](https://archive.ics.uci.edu/dataset/601/"
        "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset) · CC BY 4.0"
    )


def main():
    client = DashboardClient.from_env()
    report = read_report("final_evaluation.json")
    metadata = json.loads(
        (PROJECT_ROOT / "artifacts/model_metadata.json").read_text(encoding="utf-8")
    )
    health, error = {}, None
    try:
        health = client.health()
    except ApiError as exc:
        error = str(exc)
    ready = bool(health.get("model_ready") and health.get("database_ready"))
    with st.sidebar:
        st.markdown(
            '<div class="brand">Inspection Lab</div><div class="brand-sub">'
            "MACHINE FAILURE CLASSIFICATION</div>",
            unsafe_allow_html=True,
        )
        page = st.radio("Workspace", PAGES, key="navigation", label_visibility="collapsed")
        st.divider()
        if ready:
            st.success("Model & MySQL connected", icon=":material/check_circle:")
        elif health.get("model_ready"):
            st.warning("MySQL unavailable")
        else:
            st.error("Prediction service offline")
        if st.button("Refresh connection", width="stretch"):
            st.rerun()
        st.caption(f"Frozen model\n\n{metadata['model_version']}")
        st.caption("Synthetic-data prototype · Sensor snapshot classification")
    if error:
        st.error(error)
    elif not ready:
        st.warning("The database is unavailable. Predictions cannot be saved until it reconnects.")
    if page == "Overview":
        overview(client, ready, report)
    elif page == "Single prediction":
        single_prediction(client, ready, metadata)
    elif page == "Batch upload":
        batch_upload(client, ready)
    elif page == "History & reviews":
        history_reviews(client, ready)
    elif page == "Model evaluation":
        model_evaluation(report)
    elif page == "Data explorer":
        data_explorer()


main()
