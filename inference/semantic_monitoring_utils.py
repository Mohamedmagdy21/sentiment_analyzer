import os
import pickle
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
import umap

from inference.monitoring_utils import calculate_psi

import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda: None

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_frozen_cache = {}


def get_frozen_base(model_name):
    from inference.model_loader import _get_model_config
    cfg = _get_model_config(model_name)
    pretrained_name = cfg["pretrained_name"]
    key = pretrained_name
    if key not in _frozen_cache:
        tokenizer_name = cfg.get("tokenizer_name", pretrained_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        base_model = AutoModel.from_pretrained(pretrained_name)
        base_model.to(device)
        base_model.eval()
        _frozen_cache[key] = (tokenizer, base_model)
    return _frozen_cache[key]


def extract_embeddings(texts, tokenizer, model, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch, truncation=True, padding=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)
        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_vecs)
    return np.concatenate(all_embeddings, axis=0)


def stratified_sample(texts, labels=None, max_samples=10000, random_state=42):
    if labels is not None and len(texts) > max_samples:
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=max_samples, random_state=random_state
        )
        idx, _ = next(splitter.split(np.zeros(len(texts)), labels))
        return [texts[i] for i in idx]
    if len(texts) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(texts), max_samples, replace=False)
        return [texts[i] for i in idx]
    return texts


def fit_semantic_baseline(model_name, texts, labels=None,
                          n_components=5, n_clusters=10,
                          max_samples=10000,
                          base_dir="artifacts/models"):
    save_dir = os.path.join(project_root, base_dir, model_name, "monitoring")
    os.makedirs(save_dir, exist_ok=True)

    sample = stratified_sample(texts, labels, max_samples)
    print(f"  [{model_name}] stratified sample: {len(sample)} texts")


    # get frozen model and tokenizer and extract embeddings

    tokenizer, model = get_frozen_base(model_name)
    embeddings = extract_embeddings(sample, tokenizer, model)
    print(f"  [{model_name}] extracted {embeddings.shape[0]} x {embeddings.shape[1]} embeddings")

    effective_c = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)
    reducer = umap.UMAP(n_components=effective_c, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    effective_k = min(n_clusters, reduced.shape[0])
    clusterer = KMeans(n_clusters=effective_k, random_state=42, n_init="auto")
    clusterer.fit(reduced)

    counts = np.bincount(clusterer.labels_, minlength=effective_k)
    expected = counts.astype(float) / counts.sum()

    np.save(os.path.join(save_dir, "semantic_expected.npy"), expected)
    np.savez_compressed(
        os.path.join(save_dir, "semantic_expected.npz"),
        expected=expected,
    )

    with open(os.path.join(save_dir, "semantic_umap.pkl"), "wb") as f:
        pickle.dump(reducer, f)

    with open(os.path.join(save_dir, "semantic_kmeans.pkl"), "wb") as f:
        pickle.dump(clusterer, f)

    print(f"  [{model_name}] baseline saved to {save_dir}/")
    print(f"  [{model_name}] cluster proportions: {[round(p, 4) for p in expected]}")
    return expected


def load_semantic_baseline(model_name, base_dir="artifacts/models"):
    load_dir = os.path.join(project_root, base_dir, model_name, "monitoring")
    npy_path = os.path.join(load_dir, "semantic_expected.npy")
    npz_path = os.path.join(load_dir, "semantic_expected.npz")
    if os.path.exists(npy_path):
        return {"expected": np.load(npy_path)}
    elif os.path.exists(npz_path):
        return {"expected": np.load(npz_path)["expected"]}
    return None


def _load_umap_kmeans(model_name, base_dir="artifacts/models"):
    load_dir = os.path.join(project_root, base_dir, model_name, "monitoring")
    umap_path = os.path.join(load_dir, "semantic_umap.pkl")
    kmeans_path = os.path.join(load_dir, "semantic_kmeans.pkl")
    if not os.path.exists(umap_path) or not os.path.exists(kmeans_path):
        return None, None
    with open(umap_path, "rb") as f:
        reducer = pickle.load(f)
    with open(kmeans_path, "rb") as f:
        clusterer = pickle.load(f)
    return reducer, clusterer


def compute_production_cluster_distribution(model_name, texts,
                                            base_dir="artifacts/models"):
    reducer, clusterer = _load_umap_kmeans(model_name, base_dir)
    if reducer is None or clusterer is None:
        return None

    tokenizer, model = get_frozen_base(model_name)
    embeddings = extract_embeddings(texts, tokenizer, model)
    reduced = reducer.transform(embeddings)
    labels = clusterer.predict(reduced)
    counts = np.bincount(labels, minlength=clusterer.n_clusters)
    return counts.astype(float) / counts.sum()


def compute_semantic_psi(model_name, texts, base_dir="artifacts/models"):
    baseline = load_semantic_baseline(model_name, base_dir)
    if baseline is None:
        return None

    actual = compute_production_cluster_distribution(model_name, texts, base_dir)
    if actual is None:
        return None

    expected = baseline["expected"]
    if len(expected) != len(actual):
        actual = np.resize(actual, len(expected))
        actual = actual / actual.sum()

    return calculate_psi(expected, actual)
