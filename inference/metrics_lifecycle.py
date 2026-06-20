import json
import os
import time
from datetime import datetime, timedelta

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "metrics_lifecycle")
os.makedirs(STORAGE_DIR, exist_ok=True)

ACCUMULATION_FILE = os.path.join(STORAGE_DIR, "accumulation.json")
MEMORY_FILE = os.path.join(STORAGE_DIR, "memory.json")
ACCUMULATION_WINDOW_DAYS = 91

LABELS = ["positive", "negative", "neutral"]


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def get_inference_metrics():
    return {"positive": 0, "negative": 0, "neutral": 0, "total": 0}


def update_inference_metrics(counts):
    counts["total"] = sum(counts.get(k, 0) for k in LABELS)
    return counts


def get_accumulation():
    data = _read_json(ACCUMULATION_FILE)
    now = time.time()
    if data is None:
        data = {
            "positive": 0, "negative": 0, "neutral": 0, "total": 0,
            "created_at": now
        }
        _write_json(ACCUMULATION_FILE, data)
        return data
    elapsed_days = (now - data["created_at"]) / 86400
    if elapsed_days >= ACCUMULATION_WINDOW_DAYS:
        archive = {k: data[k] for k in ["positive", "negative", "neutral", "total"]}
        archive["start_date"] = datetime.fromtimestamp(data["created_at"]).isoformat()
        archive["end_date"] = datetime.fromtimestamp(now).isoformat()
        _append_memory(archive)
        data = {
            "positive": 0, "negative": 0, "neutral": 0, "total": 0,
            "created_at": now
        }
        _write_json(ACCUMULATION_FILE, data)
    return data


def _append_memory(archive):
    memory = _read_json(MEMORY_FILE)
    if memory is None:
        memory = {"archives": []}
    memory["archives"].append(archive)
    if len(memory["archives"]) > 100:
        memory["archives"] = memory["archives"][-100:]
    _write_json(MEMORY_FILE, memory)


def accumulate_metrics(counts):
    data = get_accumulation()
    for k in LABELS:
        data[k] += counts.get(k, 0)
    data["total"] = sum(data.get(k, 0) for k in LABELS)
    _write_json(ACCUMULATION_FILE, data)
    return data


def get_memory():
    data = _read_json(MEMORY_FILE)
    if data is None:
        data = {"archives": []}
    return data
