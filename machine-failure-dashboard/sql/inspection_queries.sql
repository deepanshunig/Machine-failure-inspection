-- Prediction history joined to readings, model registry and optional inspection reviews.
SELECT p.id, r.product_type, r.received_at, p.failure_score, p.threshold, p.alert,
       m.version, COALESCE(i.status, 'pending') AS review_status
FROM predictions p
JOIN sensor_readings r ON r.id = p.reading_id
JOIN model_versions m ON m.id = p.model_version_id
LEFT JOIN inspection_reviews i ON i.prediction_id = p.id
ORDER BY p.id DESC LIMIT 50;

-- Snapshot and alert counts; these are application logs rather than benchmark metrics.
SELECT r.product_type, COUNT(*) AS snapshots, SUM(p.alert) AS inspection_alerts
FROM sensor_readings r JOIN predictions p ON p.reading_id = r.id
GROUP BY r.product_type;

SELECT COALESCE(i.status, 'pending') AS review_status, COUNT(*) AS snapshot_count
FROM predictions p LEFT JOIN inspection_reviews i ON i.prediction_id = p.id
GROUP BY COALESCE(i.status, 'pending');

SELECT m.version, COUNT(*) AS prediction_count, AVG(p.latency_ms) AS mean_inference_ms
FROM predictions p JOIN model_versions m ON m.id = p.model_version_id
GROUP BY m.version;

-- Application receipt time in UTC; it is not a timestamp from the source dataset.
SELECT p.id, r.received_at, p.alert
FROM sensor_readings r JOIN predictions p ON p.reading_id = r.id
WHERE r.received_at >= UTC_TIMESTAMP() - INTERVAL 1 DAY
ORDER BY r.received_at DESC;
-- ix_readings_received_at supports the date filter. Foreign-key indexes support joins.
-- ix_predictions_model_date supports filtering one model version by prediction time.
