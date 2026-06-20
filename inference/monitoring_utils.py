import os
import json
import numpy as np
import pandas as pd
from glob import glob

EPSILON = 1e-4


def compute_quantile_bins(data, num_bins=10):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.array([0.0, 1.0])
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(data, percentiles)
    bins = np.unique(bins)
    if len(bins) == 1:
        bins = np.array([bins[0] - 0.5, bins[0] + 0.5])
    return bins


def get_frequency(data, bins):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    counts, _ = np.histogram(data, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), EPSILON / len(counts), dtype=float)
    return counts.astype(float) / total


def get_categorical_frequency(data, categories):
    data = np.asarray(data)
    total = len(data)
    if total == 0:
        return np.full(len(categories), EPSILON / len(categories), dtype=float)
    counts = np.array([float((data == c).sum()) for c in categories])
    return counts / total


def calculate_psi(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_to_risk(psi):
    risk_pct = min(max((psi / 0.3) * 100, 0), 100)
    if psi < 0.1:
        return round(psi, 4), "low", round(risk_pct, 1)
    elif psi < 0.2:
        return round(psi, 4), "medium", round(risk_pct, 1)
    elif psi < 0.3:
        return round(psi, 4), "high", round(risk_pct, 1)
    else:
        return round(psi, 4), "critical", round(risk_pct, 1)


def generate_and_save_baselines(model_name, train_df, test_df=None,
                                feature_cols=None, target_col=None,
                                base_dir="artifacts/models"):
    save_dir = os.path.join(base_dir, model_name, "monitoring")
    os.makedirs(save_dir, exist_ok=True)

    if feature_cols is None:
        feature_cols = ["text_length"]
    if "text_length" in feature_cols and "text_length" not in train_df.columns:
        train_df = train_df.copy()
        train_df["text_length"] = train_df.iloc[:, 0].astype(str).str.len()
        if test_df is not None:
            test_df = test_df.copy()
            test_df["text_length"] = test_df.iloc[:, 0].astype(str).str.len()

    B_data = {}
    for col in feature_cols:
        train_vals = train_df[col].dropna().values
        bins = compute_quantile_bins(train_vals)
        expected = get_frequency(train_vals, bins)
        B_data[col] = {"bins": bins.tolist(), "expected": expected.tolist()}

    data_drift = {"features": B_data}
    with open(os.path.join(save_dir, "data_drift_baseline.json"), "w") as f:
        json.dump(data_drift, f, indent=2)

    np_data = {}
    for col, v in B_data.items():
        np_data[f"{col}_bins"] = np.array(v["bins"])
        np_data[f"{col}_expected"] = np.array(v["expected"])
    np.savez_compressed(os.path.join(save_dir, "data_drift_baseline.npz"), **np_data)

    if target_col is not None:
        train_targets = train_df[target_col].dropna()
        categories = sorted(train_targets.unique().tolist())
        target_expected = get_categorical_frequency(train_targets.values, categories)

        target_drift = {"categories": categories, "expected": target_expected.tolist()}
        with open(os.path.join(save_dir, "target_drift_baseline.json"), "w") as f:
            json.dump(target_drift, f, indent=2)
        np.savez_compressed(
            os.path.join(save_dir, "target_drift_baseline.npz"),
            expected=np.array(target_expected),
            categories=np.array(categories)
        )

    confidence_vals = np.linspace(0.5, 1.0, 6)
    pred_bins = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    pred_expected = np.full(len(pred_bins) - 1, 1.0 / (len(pred_bins) - 1))
    pred_drift = {"bins": pred_bins.tolist(), "expected": pred_expected.tolist()}
    with open(os.path.join(save_dir, "prediction_drift_baseline.json"), "w") as f:
        json.dump(pred_drift, f, indent=2)
    np.savez_compressed(
        os.path.join(save_dir, "prediction_drift_baseline.npz"),
        bins=np.array(pred_bins),
        expected=np.array(pred_expected)
    )

    return save_dir


def load_baseline(model_name, drift_type, base_dir="artifacts/models"):
    load_dir = os.path.join(base_dir, model_name, "monitoring")
    json_path = os.path.join(load_dir, f"{drift_type}_drift_baseline.json")
    npz_path = os.path.join(load_dir, f"{drift_type}_drift_baseline.npz")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    if os.path.exists(npz_path):
        return np.load(npz_path)
    return None


def compute_production_psi(model_name, production_df, drift_type,
                           feature_cols=None, target_col=None,
                           base_dir="artifacts/models"):
    baseline = load_baseline(model_name, drift_type)
    if baseline is None:
        return None

    if drift_type == "data":
        if feature_cols is None:
            feature_cols = ["text_length"]
        if "text_length" in feature_cols and "text_length" not in production_df.columns:
            production_df = production_df.copy()
            production_df["text_length"] = production_df.iloc[:, 0].astype(str).str.len()
        scores = {}
        for col in feature_cols:
            if col not in baseline.get("features", {}):
                continue
            fb = baseline["features"][col]
            actual_dist = get_frequency(production_df[col].dropna().values, np.array(fb["bins"]))
            scores[col] = calculate_psi(np.array(fb["expected"]), actual_dist)
        return scores

    elif drift_type == "target":
        if target_col is None or target_col not in production_df.columns:
            return None
        categories = baseline.get("categories", [])
        expected = np.array(baseline.get("expected", []))
        actual_dist = get_categorical_frequency(production_df[target_col].dropna().values, categories)
        return {"target": calculate_psi(expected, actual_dist)}

    elif drift_type == "prediction":
        if "confidence" not in production_df.columns:
            return None
        bins = np.array(baseline.get("bins", []))
        expected = np.array(baseline.get("expected", []))
        actual_dist = get_frequency(production_df["confidence"].dropna().values, bins)
        return {"confidence": calculate_psi(expected, actual_dist)}

    return None
