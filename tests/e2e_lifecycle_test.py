"""E2E lifecycle pipeline test — validates all three Grafana metric states."""

import json
import tempfile
import time
import os
import sys
import requests

BASE = "http://localhost:8000"
PROM = "http://localhost:9090"
TMP = tempfile.gettempdir()


def log(msg):
    print(f"[TEST] {msg}")


def check(condition, msg):
    if not condition:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    log(f"PASS  {msg}")


def csv_predict(text):
    """POST a single-review CSV to /predict, return the response text."""
    csv = f"review\n{text}\n"
    p = os.path.join(TMP, "_e2e_review.csv")
    with open(p, "w") as f:
        f.write(csv)
    with open(p, "rb") as f:
        return requests.post(f"{BASE}/predict", files={"file": f})


# 1. UI served
r = requests.get(f"{BASE}/")
check(r.status_code == 200, "UI HTML returns 200")
check("Sentiment Analyzer" in r.text, "UI contains title")
check("Classify CSV" in r.text, "UI contains classify button")
check("Inference" in r.text, "UI has Inference nav button")
check("Accumulation" in r.text, "UI has Accumulation nav button")
check("Memory" in r.text, "UI has Memory nav button")
check("Chart.js" in r.text or "chart.js" in r.text.lower(), "UI includes Chart.js")

# 2. CSV predict endpoint
r = csv_predict("This product is amazing!")
check(r.status_code == 200, "Predict returns 200")
check("sentiment" in r.text, "Response has sentiment")
data = r.text.split("\n")
header = data[0].split(",")
si = header.index("sentiment")
row = data[1].split(",")
sent = row[si].strip('"')
check(sent in ("positive", "negative", "neutral"), "Sentiment valid")
log(f"  Sample review sentiment: {sent}")

# 3. Lifecycle metrics endpoint
r = requests.get(f"{BASE}/lifecycle/metrics")
check(r.status_code == 200, "Lifecycle metrics returns 200")
lcm = r.json()
check("inference" in lcm, "Lifecycle has inference")
check("accumulation" in lcm, "Lifecycle has accumulation")
check("memory" in lcm, "Lifecycle has memory")
check("created_at" in lcm["accumulation"], "Accumulation has created_at")
check(lcm["accumulation"]["total"] >= 1, "Accumulation total >= 1 after predict")

# 4. Prometheus scrape works
r = requests.get(f"{PROM}/api/v1/query?query=inference_view")
check(r.status_code == 200, "Prometheus query inference_view works")
results = r.json()["data"]["result"]
check(len(results) > 0, "inference_view has data points")

r = requests.get(f"{PROM}/api/v1/query?query=accumulation_view")
check(r.status_code == 200, "Prometheus query accumulation_view works")

r = requests.get(f"{PROM}/api/v1/query?query=accumulation_total")
check(r.status_code == 200, "Prometheus query accumulation_total works")
results = r.json()["data"]["result"]
if results:
    val = int(results[0]["value"][1])
    check(val >= 1, f"accumulation_total >= 1 (got {val})")

r = requests.get(f"{PROM}/api/v1/query?query=predictions_total")
check(r.status_code == 200, "Prometheus query predictions_total works")

# 5. Inference view resets each run
r1 = csv_predict("Terrible product")
iv1 = requests.get(f"{BASE}/lifecycle/metrics").json()["inference"]
r2 = csv_predict("Amazing love it")
iv2 = requests.get(f"{BASE}/lifecycle/metrics").json()["inference"]
check(iv1.get("positive", 0) + iv1.get("negative", 0) + iv1.get("neutral", 0) <= 1, "Inference view single-run (run 1)")
check(iv2.get("positive", 0) + iv2.get("negative", 0) + iv2.get("neutral", 0) <= 1, "Inference view single-run (run 2)")

# 6. Accumulation accumulates
acc = requests.get(f"{BASE}/lifecycle/metrics").json()["accumulation"]
check(acc["total"] >= 3, f"Accumulation total >= 3 after 3 predicts (got {acc['total']})")

# 7. Memory archives accessible
mem = requests.get(f"{BASE}/lifecycle/metrics").json()["memory"]
check("archives" in mem, "Memory has archives list")

# 8. /metrics endpoint serves Prometheus format
r = requests.get(f"{BASE}/metrics")
check(r.status_code == 200, "Prometheus /metrics endpoint works")
check("inference_view{" in r.text or "inference_view" in r.text, "/metrics contains inference_view metric")
check("accumulation_view{" in r.text or "accumulation_view" in r.text, "/metrics contains accumulation_view metric")

# 9. CSV predict with full file
with open("/home/mohamed/sentiment_analyzer/test_reviews.csv", "rb") as f:
    r = requests.post(f"{BASE}/predict", files={"file": f})
check(r.status_code == 200, "CSV predict returns 200")
check("sentiment" in r.text, "CSV predict output has sentiment")

# 10. Prometheus target check
r = requests.get(f"{PROM}/api/v1/targets")
check(r.status_code == 200, "Prometheus targets endpoint works")
targets = r.json()["data"]["activeTargets"]
inf = [t for t in targets if "inference" in t["labels"]["job"]]
check(len(inf) > 0, "Prometheus has inference target")
check(inf[0]["health"] == "up", "Inference target is UP")

log("")
log("=== ALL 10 E2E TESTS PASSED ===")
