#!/usr/bin/env python3
"""
Offline Semantic Baseline Generation — run after Train/Test split in your pipeline.

Extracts frozen-base RoBERTa [CLS] embeddings from training texts,
reduces dimensions with PCA, clusters with KMeans (K=10), and saves
the reference distribution + fitted transformers to:
    artifacts/models/<model_name>/monitoring/semantic_baseline.npz
    artifacts/models/<model_name>/monitoring/semantic_pca.pkl
    artifacts/models/<model_name>/monitoring/semantic_kmeans.pkl

Usage:
    python scripts/generate_semantic_baselines.py \
        --model twitter \
        --train-csv /path/to/train.csv \
        [--text-column review] \
        [--n-components 5] \
        [--n-clusters 10]
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference.semantic_monitoring_utils import fit_semantic_baseline


def main():
    parser = argparse.ArgumentParser(description="Generate semantic drift baselines")
    parser.add_argument("--model", required=True, choices=["twitter", "amazon"],
                        help="Model name (matches configs/model/<name>_*.yaml)")
    parser.add_argument("--train-csv", required=True,
                        help="Path to training CSV with raw text column")
    parser.add_argument("--text-column", default=None,
                        help="Column name containing text (default: first column)")
    parser.add_argument("--n-components", type=int, default=5,
                        help="PCA components (default: 5)")
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="KMeans clusters (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.train_csv):
        print(f"ERROR: training CSV not found: {args.train_csv}")
        sys.exit(1)

    df = pd.read_csv(args.train_csv)
    if args.text_column:
        if args.text_column not in df.columns:
            print(f"ERROR: column '{args.text_column}' not in CSV columns: {list(df.columns)}")
            sys.exit(1)
        texts = df[args.text_column].dropna().astype(str).tolist()
    else:
        texts = df.iloc[:, 0].dropna().astype(str).tolist()

    print(f"Model: {args.model}")
    print(f"Training texts: {len(texts)}")
    print(f"PCA components: {args.n_components}")
    print(f"KMeans clusters: {args.n_clusters}")
    print(f"Extracting embeddings and fitting baseline...")

    expected = fit_semantic_baseline(
        model_name=args.model,
        texts=texts,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
    )

    print(f"Baseline cluster proportions: {[round(p, 4) for p in expected]}")
    print(f"Done. Baseline saved to artifacts/models/{args.model}/monitoring/")


if __name__ == "__main__":
    main()
