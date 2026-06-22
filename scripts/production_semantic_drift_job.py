#!/usr/bin/env python3
"""
Production Semantic Drift Monitoring Job — run every 24h via cron.

Reads inference logs, enforces volume guardrails (>= 500 samples),
computes semantic PSI per model (twitter / amazon) via frozen-base
RoBERTa [CLS] embeddings → UMAP → KMeans, then ensembles via max()
like the inference decision engine.

Usage:
    python scripts/production_semantic_drift_job.py [--window-hours 24]
"""
import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference.semantic_monitoring_utils import compute_semantic_psi
from inference.monitoring_utils import psi_to_risk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_LOG = os.path.join(
    PROJECT_ROOT, "artifacts", "inference_logs", "predictions.jsonl"
)
RESULTS_FILE = os.path.join(
    PROJECT_ROOT, "artifacts", "inference_logs", "drift_semantic_results.json"
)

MIN_SAMPLES = 500
WINDOW_DEFAULT_HOURS = 24
WINDOW_FALLBACK_HOURS = 7 * 24
ALERT_THRESHOLD = 0.25


def load_recent_logs(window_hours=WINDOW_DEFAULT_HOURS):
    """Load inference log entries from the last N hours as a DataFrame."""
    if not os.path.exists(INFERENCE_LOG):
        return None, 0
    try:
        df = pd.read_json(INFERENCE_LOG, lines=True)
    except Exception:
        return None, 0
    if df.empty:
        return None, 0
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    df["_ts"] = pd.to_datetime(df["timestamp"])
    filtered = df[df["_ts"] >= cutoff].copy()
    return filtered, len(filtered)


def main():
    """Compute semantic PSI drift per model and ensemble, with volume guardrail and lookback expansion."""
    parser = argparse.ArgumentParser(description="24h semantic drift monitoring")
    parser.add_argument("--window-hours", type=int, default=WINDOW_DEFAULT_HOURS)
    args = parser.parse_args()

    window_hours = args.window_hours
    log_df, n_samples = load_recent_logs(window_hours)
    window_label = f"{window_hours}h"
    lookback_expanded = False

    # Volume guardrail: if fewer than MIN_SAMPLES in default window, expand lookback to 7 days
    if n_samples < MIN_SAMPLES:
        print(f"INSUFFICIENT_VOLUME_FOR_DRIFT: {n_samples} samples in {window_label} "
              f"(minimum {MIN_SAMPLES}). Expanding to {WINDOW_FALLBACK_HOURS}h window...")
        log_df, n_samples = load_recent_logs(WINDOW_FALLBACK_HOURS)
        window_label = f"{WINDOW_FALLBACK_HOURS}h (expanded)"
        lookback_expanded = True

    if log_df is None or log_df.empty or n_samples == 0:
        print("No inference logs found. Skipping.")
        return

    if n_samples < MIN_SAMPLES:
        print(f"INSUFFICIENT_VOLUME_FOR_DRIFT: only {n_samples} samples even "
              f"after {WINDOW_FALLBACK_HOURS}h lookback. Skipping drift computation.")
        result = {
            "generated_at": datetime.utcnow().isoformat(),
            "window_hours": WINDOW_FALLBACK_HOURS,
            "lookback_expanded": True,
            "status": "INSUFFICIENT_VOLUME_FOR_DRIFT",
            "total_predictions": n_samples,
            "minimum_required": MIN_SAMPLES,
            "models": {},
            "ensemble": {},
            "ensemble_risk": {}
        }
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote insufficient-volume result to {RESULTS_FILE}")
        return

    print(f"Found {n_samples} predictions in {window_label} window. "
          f"{'(lookback expanded from '+str(args.window_hours)+'h)' if lookback_expanded else ''}")

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "window_hours": window_hours,
        "lookback_expanded": lookback_expanded,
        "total_predictions": n_samples,
        "status": "OK",
        "models": {}
    }

    for model_name in ["twitter", "amazon"]:
        model_df = log_df[log_df["model_used"] == model_name]
        if model_df.empty:
            print(f"  [{model_name}] no predictions in window, skipping")
            output["models"][model_name] = {
                "psi": None, "level": "unknown", "risk_pct": 0,
                "status": "no_predictions"
            }
            continue

        texts = model_df["text"].tolist()
        print(f"  [{model_name}] {len(texts)} texts, computing semantic drift...")
        try:
            psi = compute_semantic_psi(model_name, texts)
        except Exception as e:
            print(f"  [{model_name}] semantic PSI error: {e}")
            psi = None

        if psi is not None:
            _, level, risk_pct = psi_to_risk(psi)
            alert = "ALERT" if psi >= ALERT_THRESHOLD else "OK"
            print(f"    PSI={psi:.4f}  level={level}  risk={risk_pct}%  [{alert}]")
            output["models"][model_name] = {
                "psi": round(psi, 4),
                "level": level,
                "risk_pct": risk_pct,
                "status": alert
            }
        else:
            print(f"    No baseline found, skipping")
            output["models"][model_name] = {
                "psi": None, "level": "unknown", "risk_pct": 0,
                "status": "no_baseline"
            }

    combined_psi = None
    for m_name, m_data in output["models"].items():
        v = m_data.get("psi")
        if v is not None:
            combined_psi = max(combined_psi or 0, v) if combined_psi is not None else v
    output["ensemble"] = {"psi": combined_psi}
    if combined_psi is not None:
        _, level, risk_pct = psi_to_risk(combined_psi)
        output["ensemble_risk"] = {
            "psi": round(combined_psi, 4),
            "level": level,
            "risk_pct": risk_pct
        }
    else:
        output["ensemble_risk"] = {"psi": None, "level": "unknown", "risk_pct": 0}

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    print("\n=== SEMANTIC DRIFT SUMMARY ===")
    print(f"  Window: {window_label}")
    print(f"  Samples: {n_samples}")
    print(f"  Status: {output['status']}")
    print(f"  Ensemble PSI: {combined_psi or 0:.4f}")
    print(f"  Level: {output['ensemble_risk'].get('level', '?')}")
    print(f"  Risk: {output['ensemble_risk'].get('risk_pct', 0)}%")
    if combined_psi is not None and combined_psi >= ALERT_THRESHOLD:
        print(f"  *** ALERT: PSI >= {ALERT_THRESHOLD} — retraining pipeline trigger threshold reached ***")


if __name__ == "__main__":
    main()
