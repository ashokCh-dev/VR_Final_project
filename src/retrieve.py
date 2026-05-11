"""HNSW gallery index load + nearest-neighbor search."""
import json
from pathlib import Path

import hnswlib
import numpy as np

from .config import ARTIFACTS_DIR, EMB_DIM, INDEX_FILES, resolve_artifact


def build_hnsw(embeddings, dim=EMB_DIM, ef_construction=200, M=32, ef_query=150):
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    idx.add_items(embeddings, list(range(len(embeddings))))
    idx.set_ef(ef_query)
    return idx


def save_hnsw(index, item_ids, img_names, bin_path, meta_path):
    index.save_index(str(bin_path))
    json.dump({"item_ids": item_ids, "img_names": img_names},
              open(meta_path, "w"))


def load_hnsw(bin_path, meta_path, dim=EMB_DIM, ef_query=150):
    meta = json.load(open(meta_path))
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(str(bin_path))
    idx.set_ef(ef_query)
    return idx, meta["item_ids"], meta["img_names"]


def load_named(condition_key):
    """Convenience: load by config key like 'C_alpha0.7'. Uses resolve_artifact
    so .bin files can live in either the writable artifacts/ dir or a Kaggle
    read-only mount."""
    bin_name, meta_name, alpha, model_type = INDEX_FILES[condition_key]
    idx, ids, names = load_hnsw(resolve_artifact(bin_name), resolve_artifact(meta_name))
    return idx, ids, names, alpha, model_type


def search(index, query_emb, k):
    """query_emb: (1, D) or (N, D) numpy. Returns (labels, distances)."""
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    labels, distances = index.knn_query(query_emb, k=k)
    return labels, distances


def candidates_from_labels(labels, item_ids, img_names, distances=None):
    """Map HNSW label rows -> list of (item_id, img_name, score) per query."""
    out = []
    for i, row in enumerate(labels):
        cands = []
        for j, pos in enumerate(row):
            cands.append({
                "item_id": item_ids[int(pos)],
                "img_name": img_names[int(pos)],
                "score": float(1.0 - distances[i][j]) if distances is not None else None,
            })
        out.append(cands)
    return out
