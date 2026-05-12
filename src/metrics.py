"""Corrected retrieval metrics — copied from clip_with_saved.ipynb Cell 4 (fixed).

`bootstrap_per_query` lets us report mean ± std for conditions where we don't
have multiple trained checkpoints, by resampling the query set with different
seeds. This is the random-component-of-choice for the frozen-CLIP conditions
(A, B) and the vanilla-fine-tuned conditions (C without _hn), per the prof's
"any random component, justify in viva" rule.
"""
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


def evaluate_retrieval(query_embs, query_ids, gallery_ids, index, k_list=(5, 10, 15),
                       return_per_query=False):
    """Run HNSW search on `query_embs` against `index`, score against gallery_ids.

    If `return_per_query=True`, returns (means_dict, per_query_dict). The
    per-query dict has one list of length N (number of queries) per metric and
    is what `bootstrap_mean_std` consumes."""
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

    means = {m: float(np.mean(v)) for m, v in per_query.items()}
    if return_per_query:
        return means, per_query
    return means


def bootstrap_mean_std(per_query_scores, n_iters=4, sample_frac=0.8, seeds=(83, 588, 527, 33)):
    """Resample the query set with replacement and return per-metric mean ± std.

    `per_query_scores`: dict from `evaluate_retrieval(..., return_per_query=True)`.
    `seeds` overrides random seeds (one per iteration). `n_iters` is capped to
    len(seeds). Returns dict of {metric: {"mean": float, "std": float}}.
    """
    out = {}
    n_iters = min(n_iters, len(seeds))
    for metric, scores in per_query_scores.items():
        arr = np.asarray(scores, dtype=np.float64)
        n = len(arr)
        sample_size = max(1, int(n * sample_frac))
        run_means = []
        for s in seeds[:n_iters]:
            rng = np.random.default_rng(int(s))
            idx = rng.integers(0, n, size=sample_size)  # sample with replacement
            run_means.append(float(arr[idx].mean()))
        out[metric] = {"mean": float(np.mean(run_means)), "std": float(np.std(run_means, ddof=1))}
    return out


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
