"""BLIP-2 ITM re-ranking — re-order top-K' candidates by image-text matching score.

For short product captions, ITM has weak discriminative power within an already
visually-similar candidate set, so we use it as a *tiebreaker* on top of the
ANN cosine score rather than as the sole signal. Tune `itm_weight` in [0, 1].
"""
from .blip2 import itm_scores


def rerank_with_itm(query_pil, candidates, captions_dict, top_k=10,
                    itm_weight=0.3, ann_weight=0.7, mode="blend"):
    """
    candidates: list of dicts with keys {"item_id", "img_name", "score"} from
        retrieve.candidates_from_labels. `score` is the cosine similarity from ANN.
    captions_dict: {image_name: caption_text}, our gallery captions.
    mode:
      - "blend"   -> combined = ann_weight * ann + itm_weight * itm  (default)
      - "itm"     -> sort by itm alone (legacy behaviour, generally worst on short captions)
      - "product" -> combined = ann * itm  (multiplicative, ITM as gate)
    Returns: list of candidates in re-ranked order, truncated to `top_k`.
    Each candidate is annotated with `ann_score`, `itm_score`, and `combined_score`.
    """
    if not candidates:
        return []
    caps = [captions_dict.get(c["img_name"], "") for c in candidates]
    scores = itm_scores(query_pil, caps)
    enriched = []
    for cand, itm in zip(candidates, scores):
        c = dict(cand)
        c["ann_score"] = c.pop("score", None) or 0.0
        c["itm_score"] = float(itm)
        if mode == "itm":
            c["combined_score"] = c["itm_score"]
        elif mode == "product":
            c["combined_score"] = c["ann_score"] * c["itm_score"]
        else:  # blend
            c["combined_score"] = ann_weight * c["ann_score"] + itm_weight * c["itm_score"]
        enriched.append(c)
    enriched.sort(key=lambda c: c["combined_score"], reverse=True)
    return enriched[:top_k]
