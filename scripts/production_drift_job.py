#!/usr/bin/env python3
"""
Production Drift Monitoring Job — run every 24h via cron or scheduler.

Reads the last 24 hours of inference logs, loads baseline artifacts,
computes Data / Prediction / Target PSI per model, writes results to a
shared JSON file, and optionally pushes to Prometheus.

Usage:
    python production_drift_job.py [--window-hours 24]
"""
import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference.monitoring_utils import (
    compute_production_psi, psi_to_risk
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_LOG = os.path.join(
    PROJECT_ROOT, "artifacts", "inference_logs", "predictions.jsonl"
)
BASELINE_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
DRIFT_RESULTS = os.path.join(
    PROJECT_ROOT, "artifacts", "inference_logs", "drift_results.json"
)
PROMETHEUS_DIR = os.path.join(
    PROJECT_ROOT, "artifacts", "inference_logs"
)


def load_recent_logs(window_hours=24):
    if not os.path.exists(INFERENCE_LOG):
        return None
    try:
        df = pd.read_json(INFERENCE_LOG, lines=True)
    except Exception:
        return None
    if df.empty:
        return None
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    df["_ts"] = pd.to_datetime(df["timestamp"])
    return df[df["_ts"] >= cutoff].copy()


def compute_drift(model_name, log_df):
    results = {}
    for drift_type in ["data", "target", "prediction"]:
        try:
            score = compute_production_psi(
                model_name, log_df, drift_type,
                base_dir=BASELINE_DIR
            )
            if score is not None:
                if drift_type == "data":
                    results[drift_type] = list(score.values())[0] \
                        if isinstance(score, dict) and score else None
                elif drift_type == "target":
                    results[drift_type] = score.get("target")
                elif drift_type == "prediction":
                    results[drift_type] = score.get("confidence")
            else:
                results[drift_type] = None
        except Exception as e:
            print(f"  [{model_name}] {drift_type} error: {e}")
            results[drift_type] = None
    return results


def main():
    parser = argparse.ArgumentParser(description="24h batch drift monitoring")
    parser.add_argument("--window-hours", type=int, default=24)
    args = parser.parse_args()

    print(f"Loading last {args.window_hours}h of inference logs...")
    log_df = load_recent_logs(args.window_hours)
    if log_df is None or log_df.empty:
        print("No inference logs found in window. Skipping.")
        return

    print(f"Found {len(log_df)} predictions in window.")

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "window_hours": args.window_hours,
        "total_predictions": len(log_df),
        "models": {}
    }

    for model_name in ["twitter", "amazon"]:
        model_df = log_df[log_df["model_used"] == model_name]
        if model_df.empty:
            print(f"  [{model_name}] no predictions in window, skipping")
            continue
        print(f"  [{model_name}] {len(model_df)} predictions, computing drift...")
        drift_scores = compute_drift(model_name, model_df)
        risk = {}
        for k, v in drift_scores.items():
            if v is not None:
                _, level, pct = psi_to_risk(v)
                risk[k] = {"psi": round(v, 4), "level": level, "risk_pct": pct}
            else:
                risk[k] = {"psi": None, "level": "unknown", "risk_pct": 0}
        output["models"][model_name] = {"drift": drift_scores, "risk": risk}

    combined = {"data": None, "target": None, "prediction": None}
    for m_name, m_data in output["models"].items():
        for k in ["data", "target", "prediction"]:
            v = m_data["drift"].get(k)
            if v is not None:
                combined[k] = max(combined[k] or 0, v) if combined[k] is not None else v
    output["ensemble"] = combined
    output["ensemble_risk"] = {}
    for k, v in combined.items():
        if v is not None:
            _, level, pct = psi_to_risk(v)
            output["ensemble_risk"][k] = {"psi": round(v, 4), "level": level, "risk_pct": pct}

    os.makedirs(os.path.dirname(DRIFT_RESULTS), exist_ok=True)
    with open(DRIFT_RESULTS, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {DRIFT_RESULTS}")

    print("\n=== ENSEMBLE DRIFT SUMMARY ===")
    for k, v in combined.items():
        level = output["ensemble_risk"].get(k, {}).get("level", "?")
        pct = output["ensemble_risk"].get(k, {}).get("risk_pct", 0)
        print(f"  {k.title():12s} PSI={v or 0:.4f}  level={level}  risk={pct}%")


if __name__ == "__main__":
    main()
