import os
import time
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, HTMLResponse, FileResponse
from inference.decision_engine import decide
from inference.metrics import (
    PREDICTIONS_TOTAL, INFERENCE_LATENCY, CONFIDENCE,
    INFERENCE_VIEW, ACCUMULATION_VIEW, MEMORY_VIEW,
    ACCUMULATION_CREATED_AT, ACCUMULATION_TOTAL,
    DRIFT_DATA, DRIFT_PREDICTION, DRIFT_TARGET,
    SEMANTIC_DRIFT_PSI,
    get_metrics
)
from inference.metrics_lifecycle import (
    get_accumulation, accumulate_metrics, get_memory, LABELS
)
from inference.monitoring_utils import (
    compute_production_psi, load_baseline,
    get_frequency, get_categorical_frequency, calculate_psi
)

app = FastAPI(title="Sentiment Analyzer")

INFERENCE_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "inference_logs"
)
os.makedirs(INFERENCE_LOG_DIR, exist_ok=True)
INFERENCE_LOG = os.path.join(INFERENCE_LOG_DIR, "predictions.jsonl")


@app.on_event("startup")
def kill_existing():
    if not os.path.exists("/.dockerenv"):
        os.system("pkill -f 'uvicorn inference.main' > /dev/null 2>&1")
    accumulation = get_accumulation()
    memory = get_memory()
    _update_lifecycle_gauges(_last_inference, accumulation, memory)

    semantic_path = os.path.join(INFERENCE_LOG_DIR, "drift_semantic_results.json")
    if os.path.exists(semantic_path):
        try:
            with open(semantic_path) as f:
                sr = json.load(f)
            for m_name in ["twitter", "amazon"]:
                m_psi = sr.get("models", {}).get(m_name, {}).get("psi")
                if m_psi is not None:
                    SEMANTIC_DRIFT_PSI.labels(model=m_name).set(m_psi)
            e_psi = sr.get("ensemble", {}).get("psi")
            if e_psi is not None:
                SEMANTIC_DRIFT_PSI.labels(model="ensemble").set(e_psi)
        except Exception:
            pass


twitter_models = None
amazon_models = None
_last_inference = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
_last_csv = None
TEMP_CSV = "/tmp/_sentiment_result.csv"


def get_twitter_models():
    global twitter_models
    if twitter_models is None:
        from inference.model_loader import load_model
        tokenizer, model = load_model("twitter")
        twitter_models = (tokenizer, model)
    return twitter_models


def get_amazon_models():
    global amazon_models
    if amazon_models is None:
        from inference.model_loader import load_model
        tokenizer, model = load_model("amazon")
        amazon_models = (tokenizer, model)
    return amazon_models


def _classify_texts(texts):
    results = decide(texts, get_twitter_models(), get_amazon_models())
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        counts[r.sentiment] += 1
        PREDICTIONS_TOTAL.labels(sentiment=r.sentiment, model=r.model_used).inc()
        CONFIDENCE.labels(model=r.model_used).observe(r.confidence)
    return results, counts


def _log_inference(texts, results):
    lines = []
    now = datetime.utcnow().isoformat()
    for i, r in enumerate(results):
        lines.append(json.dumps({
            "timestamp": now,
            "text": texts[i],
            "text_length": len(texts[i]),
            "sentiment": r.sentiment,
            "label": r.label,
            "confidence": r.confidence,
            "model_used": r.model_used
        }))
    with open(INFERENCE_LOG, "a") as f:
        f.write("\n".join(lines) + "\n")


def _update_lifecycle_gauges(inference_counts, accumulation_data, memory_data):
    for s in LABELS:
        INFERENCE_VIEW.labels(sentiment=s).set(inference_counts.get(s, 0))
        ACCUMULATION_VIEW.labels(sentiment=s).set(accumulation_data.get(s, 0))
    ACCUMULATION_TOTAL.set(accumulation_data.get("total", 0))
    ACCUMULATION_CREATED_AT.set(accumulation_data.get("created_at", 0))
    for arch in memory_data.get("archives", []):
        label = arch.get("start_date", "unknown")[:10]
        for s in LABELS:
            MEMORY_VIEW.labels(sentiment=s, period=label).set(arch.get(s, 0))


def _record_inference(counts):
    global _last_inference
    _last_inference = {**counts, "total": sum(counts.values())}
    accumulation = accumulate_metrics(counts)
    memory = get_memory()
    _update_lifecycle_gauges(counts, accumulation, memory)


@app.post("/predict", response_class=Response)
def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    df = pd.read_csv(file.file)
    df = df.dropna(subset=[df.columns[0]])
    texts = df.iloc[:, 0].tolist()
    results, counts = _classify_texts(texts)
    df["sentiment"] = [r.sentiment for r in results]
    df["label"] = [r.label for r in results]
    df["confidence"] = [r.confidence for r in results]
    df["model_used"] = [r.model_used for r in results]
    elapsed = time.perf_counter() - start
    INFERENCE_LATENCY.observe(elapsed)
    _record_inference(counts)
    _log_inference(texts, results)
    csv_output = df.to_csv(index=False)
    global _last_csv
    _last_csv = csv_output
    with open(TEMP_CSV, "w") as f:
        f.write(csv_output)
    return Response(content=csv_output, media_type="text/csv")


@app.get("/lifecycle/metrics")
def lifecycle_metrics():
    acc = get_accumulation()
    mem = get_memory()
    return {
        "inference": _last_inference,
        "accumulation": acc,
        "memory": mem
    }


@app.get("/metrics")
def metrics():
    return Response(content=get_metrics(), media_type="text/plain")


@app.get("/download")
def download():
    if not _last_csv:
        return Response("No results yet", status_code=404)
    return Response(content=_last_csv, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=results.csv"
    })


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    index = os.path.join(STATIC_DIR, "index.html")
    with open(index) as f:
        return f.read()


@app.get("/drift/metrics")
def drift_metrics():
    if not os.path.exists(INFERENCE_LOG):
        return {"drift": {}, "risk": {}}
    try:
        log_df = pd.read_json(INFERENCE_LOG, lines=True)
    except Exception:
        return {"drift": {}, "risk": {}}
    if log_df.empty:
        return {"drift": {}, "risk": {}}

    results = {}
    for model_name in ["twitter", "amazon"]:
        model_df = log_df[log_df["model_used"] == model_name].copy()
        if model_df.empty:
            continue
        baseline_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts", "models"
        )
        m_results = {}
        for drift_type in ["data", "target", "prediction"]:
            try:
                kwargs = {"base_dir": baseline_dir}
                if drift_type == "target":
                    kwargs["target_col"] = "sentiment"
                elif drift_type == "data":
                    kwargs["feature_cols"] = ["text_length"]
                score = compute_production_psi(
                    model_name, model_df, drift_type, **kwargs
                )
                if score is not None:
                    if drift_type == "data":
                        m_results[drift_type] = list(score.values())[0] \
                            if isinstance(score, dict) and score else None
                    elif drift_type == "target":
                        m_results[drift_type] = score.get("target")
                    elif drift_type == "prediction":
                        m_results[drift_type] = score.get("confidence")
            except Exception:
                m_results[drift_type] = None
        results[model_name] = m_results

    combined = {"data": None, "target": None, "prediction": None}
    for model_name, m_res in results.items():
        for k in ["data", "target", "prediction"]:
            v = m_res.get(k)
            if v is not None:
                combined[k] = max(combined[k] or 0, v) if combined[k] is not None else v

    from inference.monitoring_utils import psi_to_risk
    risk = {}
    for k, v in combined.items():
        if v is not None:
            _, level, pct = psi_to_risk(v)
            risk[k] = {"psi": round(v, 4), "level": level, "risk_pct": pct}
        else:
            risk[k] = {"psi": None, "level": "unknown", "risk_pct": 0}

    if combined.get("data") is not None:
        DRIFT_DATA.labels(model="ensemble").set(combined["data"])
    if combined.get("prediction") is not None:
        DRIFT_PREDICTION.labels(model="ensemble").set(combined["prediction"])
    if combined.get("target") is not None:
        DRIFT_TARGET.labels(model="ensemble").set(combined["target"])

    semantic_path = os.path.join(INFERENCE_LOG_DIR, "drift_semantic_results.json")
    semantic_risk = None
    if os.path.exists(semantic_path):
        try:
            with open(semantic_path) as f:
                sr = json.load(f)
            for m_name in ["twitter", "amazon"]:
                m_psi = sr.get("models", {}).get(m_name, {}).get("psi")
                if m_psi is not None:
                    SEMANTIC_DRIFT_PSI.labels(model=m_name).set(m_psi)
            e_psi = sr.get("ensemble", {}).get("psi")
            if e_psi is not None:
                SEMANTIC_DRIFT_PSI.labels(model="ensemble").set(e_psi)
                from inference.monitoring_utils import psi_to_risk
                _, s_level, s_pct = psi_to_risk(e_psi)
                semantic_risk = {"psi": round(e_psi, 4), "level": s_level, "risk_pct": s_pct}
        except Exception:
            pass

    return {"drift": combined, "risk": risk, "semantic": semantic_risk}
