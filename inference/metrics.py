from prometheus_client import Counter, Histogram, Gauge, generate_latest

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions per sentiment class",
    ["sentiment", "model"]
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Latency per /predict request",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence scores per model",
    ["model"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

INFERENCE_VIEW = Gauge(
    "inference_view",
    "Last inference run counts (overwrites each run)",
    ["sentiment"]
)

ACCUMULATION_VIEW = Gauge(
    "accumulation_view",
    "Accumulated counts over rolling 91-day window",
    ["sentiment"]
)

MEMORY_VIEW = Gauge(
    "memory_view",
    "Historical archived counts",
    ["sentiment", "period"]
)

ACCUMULATION_CREATED_AT = Gauge(
    "accumulation_created_at",
    "Unix timestamp when the current accumulation window started"
)

ACCUMULATION_TOTAL = Gauge(
    "accumulation_total",
    "Total accumulated predictions in current window"
)

DRIFT_DATA = Gauge(
    "drift_data_psi",
    "Data drift PSI score for text length features",
    ["model"]
)

DRIFT_PREDICTION = Gauge(
    "drift_prediction_psi",
    "Prediction drift PSI score for confidence distribution",
    ["model"]
)

DRIFT_TARGET = Gauge(
    "drift_target_psi",
    "Target drift PSI score for sentiment label distribution",
    ["model"]
)

SEMANTIC_DRIFT_PSI = Gauge(
    "semantic_drift_psi",
    "Semantic data drift PSI score via frozen-base embeddings + PCA + KMeans",
    ["model"]
)


def get_metrics():
    return generate_latest()
