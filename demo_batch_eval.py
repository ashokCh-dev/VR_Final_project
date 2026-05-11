"""
Batch evaluation: runs the full retrieval pipeline over a folder of query images
and reports Recall@K, NDCG@K, mAP@K for K∈{5,10,15}.

Examples:
  # Evaluate all DeepFashion query-split images against condition C, α=0.7
  python demo_batch_eval.py --condition C_alpha0.7

  # Evaluate a custom folder of images (item_id parsed from path)
  python demo_batch_eval.py --query_dir /path/to/queries --condition C_alpha0.7

  # Same but with BLIP-2 ITM re-rank
  python demo_batch_eval.py --condition C_alpha0.7 --rerank

Ground truth for the DeepFashion split: two images match iff they share an item_id.
We parse item_id from path component "id_XXXXXXXX".
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src import config
from src.clip_encoder import ClipEncoder
from src.data_utils import (
    bbox_crop,
    item_id_from_path,
    load_captions,
    parse_bbox_file,
    parse_split_file,
)
from src.metrics import evaluate_from_ranked_lists, evaluate_retrieval
from src.retrieve import candidates_from_labels, load_named, search


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="C_alpha0.7",
                   help="Which gallery index / model condition to use (key in INDEX_FILES, "
                        "or arbitrary name when --bin_path/--meta_path overrides are given)")
    p.add_argument("--bin_path", default=None, help="Override HNSW .bin path")
    p.add_argument("--meta_path", default=None, help="Override HNSW meta .json path")
    p.add_argument("--checkpoint", default=None, help="Override CLIP checkpoint path")
    p.add_argument("--alpha", type=float, default=None, help="Override fusion alpha")
    p.add_argument("--model_type", default=None, choices=["frozen", "ft", "ft_hn"],
                   help="Override model_type when using path overrides")
    p.add_argument("--query_dir", default=None,
                   help="Folder of query images. If omitted, uses the DeepFashion query split.")
    p.add_argument("--k", nargs="+", type=int, default=config.TOP_K_LIST,
                   help="K values for Recall/NDCG/mAP")
    p.add_argument("--rerank", action="store_true",
                   help="Apply BLIP-2 ITM re-ranking to top-RERANK_K candidates")
    p.add_argument("--rerank_k", type=int, default=config.RERANK_K,
                   help="Number of candidates pulled before ITM re-rank")
    p.add_argument("--rerank_mode", choices=["blend", "itm", "product"], default="blend",
                   help="How to combine ANN cosine score with ITM probability")
    p.add_argument("--itm_weight", type=float, default=0.3,
                   help="ITM weight in blend mode (ann_weight = 1 - itm_weight)")
    p.add_argument("--yolo", action="store_true",
                   help="Use YOLO to crop query images. Default off (uses bbox annotations "
                        "when --query_dir is the DeepFashion split).")
    p.add_argument("--max_queries", type=int, default=None,
                   help="Cap on number of queries (for quick smoke tests)")
    p.add_argument("--out", default=None, help="Path to write metrics JSON")
    p.add_argument("--use_hn_checkpoint", action="store_true",
                   help="Load clip_finetuned_hn.pt instead of clip_finetuned.pt for C-* conditions")
    return p.parse_args()


def gather_queries_from_split(max_queries=None):
    """Returns list of (Path, item_id, image_name_relative) for the DeepFashion query split."""
    all_rows = parse_split_file()
    query_rows = [r for r in all_rows if r["split"] == "query"]
    if max_queries:
        query_rows = query_rows[:max_queries]
    out = []
    for r in query_rows:
        p = config.IMG_ROOT / r["image_name"]
        if p.exists():
            out.append((p, r["item_id"], r["image_name"]))
    return out


def gather_queries_from_dir(query_dir, max_queries=None):
    """Returns list of (Path, item_id, image_name_relative)."""
    query_dir = Path(query_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in query_dir.rglob("*") if p.suffix.lower() in exts)
    if max_queries:
        paths = paths[:max_queries]
    out = []
    for p in paths:
        iid = item_id_from_path(str(p))
        if iid is None:
            print(f"  WARN: could not parse item_id from {p}; skipping")
            continue
        out.append((p, iid, str(p)))
    return out


def main():
    args = parse_args()
    print(f"Device: {config.DEVICE}")
    print(f"Condition: {args.condition}")

    # Resolve from explicit overrides or fall back to INDEX_FILES lookup
    if args.bin_path or args.meta_path:
        if not (args.bin_path and args.meta_path):
            raise SystemExit("--bin_path and --meta_path must be given together")
        alpha = args.alpha if args.alpha is not None else 0.7
        model_type = args.model_type or "ft_hn"
        bin_path = Path(args.bin_path)
        meta_path = Path(args.meta_path)
    else:
        bin_name, meta_name, alpha, model_type = config.INDEX_FILES[args.condition]
        bin_path = config.resolve_artifact(bin_name)
        meta_path = config.resolve_artifact(meta_name)
        if args.alpha is not None:
            alpha = args.alpha
    print(f"  alpha={alpha}, model_type={model_type}, rerank={args.rerank}, yolo={args.yolo}")

    # ── Load CLIP ──
    if args.checkpoint:
        print(f"  Loading CLIP from override checkpoint: {args.checkpoint}")
        encoder = ClipEncoder(checkpoint_path=Path(args.checkpoint))
    elif model_type == "ft_hn":
        print(f"  Loading hard-neg fine-tuned CLIP: {config.CLIP_FT_HN.name}")
        encoder = ClipEncoder(checkpoint_path=config.CLIP_FT_HN)
    elif model_type == "ft":
        ckpt = config.CLIP_FT_HN if args.use_hn_checkpoint and config.CLIP_FT_HN.exists() else config.CLIP_FT
        print(f"  Loading fine-tuned CLIP: {ckpt.name}")
        encoder = ClipEncoder(checkpoint_path=ckpt)
    else:
        print("  Loading frozen CLIP")
        encoder = ClipEncoder(checkpoint_path=None)

    # ── Load gallery index ──
    from src.retrieve import load_hnsw
    index, gal_ids, gal_names = load_hnsw(bin_path, meta_path)
    captions = load_captions() if alpha < 1.0 or args.rerank else {}
    print(f"  Gallery: {len(gal_ids)} items in index")

    # ── Gather queries ──
    if args.query_dir:
        queries = gather_queries_from_dir(args.query_dir, args.max_queries)
    else:
        queries = gather_queries_from_split(args.max_queries)
    print(f"  Queries: {len(queries)}")

    # ── Embed all queries ──
    bbox_map = parse_bbox_file() if not args.yolo else {}
    if args.yolo:
        from src.yolo_crop import yolo_crop  # lazy: avoids ultralytics startup unless asked

    print("  Encoding queries...")
    q_embs = []
    q_ids = []
    q_pils = []  # kept for re-rank (only if --rerank)
    for img_path, item_id, name in tqdm(queries):
        try:
            pil = Image.open(img_path).convert("RGB")
            if args.yolo:
                pil_crop, _ = yolo_crop(pil)
            else:
                bbox = bbox_map.get(name)
                pil_crop = bbox_crop(pil, bbox) if bbox else pil
            cap = captions.get(name, "") if alpha < 1.0 else ""
            emb = encoder.embed_query(pil_crop, caption=cap, alpha=alpha)
            q_embs.append(emb[0])
            q_ids.append(item_id)
            if args.rerank:
                q_pils.append(pil_crop)
        except Exception as e:
            print(f"  WARN: failed on {img_path}: {e}")
    q_embs = np.vstack(q_embs).astype("float32")
    print(f"  Query embeddings: {q_embs.shape}")

    # ── Retrieve + (optional) re-rank ──
    max_k = max(args.k)
    fetch_k = args.rerank_k if args.rerank else max_k

    print(f"  HNSW search (k={fetch_k})...")
    t0 = time.time()
    labels, distances = search(index, q_embs, k=fetch_k)
    print(f"    {time.time() - t0:.1f}s")

    if args.rerank:
        from src.rerank import rerank_with_itm
        print(f"  BLIP-2 ITM re-ranking top-{fetch_k} -> top-{max_k} "
              f"(mode={args.rerank_mode}, itm_w={args.itm_weight})...")
        cand_lists = candidates_from_labels(labels, gal_ids, gal_names, distances)
        ranked_item_ids = []
        for q_pil, cands in tqdm(list(zip(q_pils, cand_lists))):
            ranked = rerank_with_itm(
                q_pil, cands, captions, top_k=max_k,
                itm_weight=args.itm_weight, ann_weight=1.0 - args.itm_weight,
                mode=args.rerank_mode,
            )
            ranked_item_ids.append([c["item_id"] for c in ranked])
        results = evaluate_from_ranked_lists(ranked_item_ids, q_ids, gal_ids, k_list=tuple(args.k))
    else:
        results = evaluate_retrieval(q_embs, q_ids, gal_ids, index, k_list=tuple(args.k))

    # ── Report ──
    print("\n" + "=" * 50)
    print(f"Results — {args.condition}" + (" + ITM rerank" if args.rerank else ""))
    print("=" * 50)
    for k in args.k:
        for m in ["Recall", "NDCG", "mAP"]:
            mk = f"{m}@{k}"
            print(f"  {mk}: {results[mk]:.4f}")

    out_path = Path(args.out) if args.out else config.ARTIFACTS_DIR / (
        f"batch_eval_{args.condition}{'_itm' if args.rerank else ''}.json"
    )
    payload = {
        "condition": args.condition,
        "alpha": alpha,
        "rerank": args.rerank,
        "yolo": args.yolo,
        "n_queries": len(q_ids),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
