"""Corrected retrieval metrics — copied from clip_with_saved.ipynb Cell 4 (fixed)."""
from collections import defaultdict

import numpy as np


def recall_at_k(retrieved_ids, relevant_item_id, k):
    return int(any(iid == relevant_item_id for iid in retrieved_ids[:k]))


def ndcg_at_k(retrieved_ids, relevant_item_id, n_relevant, k):
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, iid in enumerate(top_k)
        if iid == relevant_item_id
    )
    n_ideal = min(n_relevant, k)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(retrieved_ids, relevant_item_id, n_relevant, k):
    top_k = retrieved_ids[:k]
    hits, prec_sum = 0, 0.0
    for rank, iid in enumerate(top_k, 1):
        if iid == relevant_item_id:
            hits += 1
            prec_sum += hits / rank
    denom = min(n_relevant, k)
    return prec_sum / denom if denom > 0 else 0.0


def evaluate_retrieval(query_embs, query_ids, gallery_ids, index, k_list=(5, 10, 15)):
    """Run HNSW search on `query_embs` against `index`, score against gallery_ids."""
    gallery_id_to_count = defaultdict(int)
    for iid in gallery_ids:
        gallery_id_to_count[iid] += 1

    max_k = max(k_list)
    gallery_id_arr = np.array(gallery_ids)
    per_query = {f"{m}@{k}": [] for m in ["Recall", "NDCG", "mAP"] for k in k_list}

    labels, _ = index.knn_query(query_embs, k=max_k)

    for q_idx, q_id in enumerate(query_ids):
        retrieved_item_ids = gallery_id_arr[labels[q_idx]].tolist()
        n_relevant = gallery_id_to_count.get(q_id, 0)
        for k in k_list:
            per_query[f"Recall@{k}"].append(recall_at_k(retrieved_item_ids, q_id, k))
            per_query[f"NDCG@{k}"].append(ndcg_at_k(retrieved_item_ids, q_id, n_relevant, k))
            per_query[f"mAP@{k}"].append(average_precision_at_k(retrieved_item_ids, q_id, n_relevant, k))

    return {m: float(np.mean(v)) for m, v in per_query.items()}


def evaluate_from_ranked_lists(ranked_item_ids_per_query, query_ids, gallery_ids, k_list=(5, 10, 15)):
    """
    Same metrics, but for the post-rerank case where you already have an
    ordered list of item_ids per query (no HNSW search needed).
    `ranked_item_ids_per_query[i]` should already be in the desired order.
    """
    gallery_id_to_count = defaultdict(int)
    for iid in gallery_ids:
        gallery_id_to_count[iid] += 1
    per_query = {f"{m}@{k}": [] for m in ["Recall", "NDCG", "mAP"] for k in k_list}
    for q_idx, q_id in enumerate(query_ids):
        retrieved = ranked_item_ids_per_query[q_idx]
        n_relevant = gallery_id_to_count.get(q_id, 0)
        for k in k_list:
            per_query[f"Recall@{k}"].append(recall_at_k(retrieved, q_id, k))
            per_query[f"NDCG@{k}"].append(ndcg_at_k(retrieved, q_id, n_relevant, k))
            per_query[f"mAP@{k}"].append(average_precision_at_k(retrieved, q_id, n_relevant, k))
    return {m: float(np.mean(v)) for m, v in per_query.items()}
