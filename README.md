# Visual Product Search Engine

Image-to-image retrieval over DeepFashion In-Shop, with CLIP fine-tuning and BLIP-2 ITM re-ranking. Course project for the Visual Recognition class.

```
Query image
  └─ YOLO crop ──► CLIP image encoder ──► HNSW ANN search ──► top-K candidates
                                                 │
                                                 └─ (optional) BLIP-2 ITM rerank ──► top-K final
```

Gallery is indexed offline: each catalog image → bbox crop → CLIP image encoder, fused with BLIP-2-caption text encoder (`v = α·image + (1−α)·text`, L2-normalized) → HNSW.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/config.py` | Single source of truth for paths + condition registry. Auto-detects Kaggle vs local. |
| `src/data_utils.py` | DeepFashion split/bbox parsing, bbox crop helper. |
| `src/metrics.py` | Corrected Recall@K / NDCG@K / mAP@K **+** `bootstrap_mean_std` for query-set resampling. |
| `src/clip_encoder.py` | OpenCLIP load, image/text encode, fused embedding. |
| `src/yolo_crop.py` | YOLO online crop with confidence threshold + fallback. |
| `src/retrieve.py` | HNSW build / save / load / search. |
| `src/blip2.py` | BLIP-2 captioning (8-bit) + BLIP-2 ITM scorer (BLIP-1 fallback). |
| `src/rerank.py` | ITM-based re-ranking (`blend` / `itm` / `product` modes). |
| `train_hn.py` | Hard-neg CLIP fine-tuning. 4060-tuned (batch=8, grad-accum=4). `--seed`, `--suffix`. |
| `rebuild_indices_hn.py` | Rebuild C-condition HNSW indices for any checkpoint. |
| `demo_batch_eval.py` | Batch evaluation. `--condition`, `--rerank`, `--yolo`, `--bootstrap N`. |
| `app.py` | Streamlit demo (upload → YOLO crop → confirm → retrieve → optional rerank). |
| `kaggle_overnight.ipynb` | Pipeline notebook for Kaggle (full-set ITM eval + multi-seed training). |
| `run_overnight.sh` | Equivalent shell sequencer for local overnight runs. |
| `clip-hard-negative-mining.ipynb` | Original research notebook for the hard-neg variant (kept as lab log). |
| `kernel-metadata.json` | For `kaggle kernels push -p .` CLI workflow. |

## Status — what's done

- BLIP-2 captioning (38,494 short product captions; loaded from `taralsanka/blip-updated-captions` on Kaggle, mirrored to `artifacts/captions.json` locally).
- CLIP ViT-L/14 fine-tuning, last 4 vision blocks + projection (~51M / 428M params trainable). Two variants:
  - **Vanilla** (InfoNCE only) — `clip_finetuned.pt`.
  - **Hard-neg** (InfoNCE + Triplet + HNSW-mined hard negatives, pool refreshed every 3 epochs) — `clip_finetuned_hn.pt`. This is our headline model.
- 7 HNSW gallery indices built (A, B α=0.5/0.7, C-vanilla α=0.5/0.7, C-hn α=0.5/0.7).
- Full-set evaluation on 14,218 queries for condition C-hn α=0.7 across **two real fine-tuning seeds (83, 588)**.
- Full-set BLIP-2 ITM re-rank evaluation on Kaggle (~5h on T4).
- Streamlit demo + batch-eval CLI + Kaggle-portable pipeline + GitHub repo.

## Headline numbers (C_alpha0.7_hn, 14,218 queries)

| Metric | Seed 83 | Seed 588 | mean ± std |
| --- | --- | --- | --- |
| R@5  | 0.8364 | 0.8386 | 0.8375 ± 0.0011 |
| R@10 | 0.8796 | 0.8862 | 0.8829 ± 0.0033 |
| R@15 | 0.9022 | 0.9058 | 0.9040 ± 0.0018 |
| NDCG@10 | 0.5775 | 0.5823 | 0.5799 ± 0.0024 |
| mAP@10  | 0.4793 | 0.4836 | 0.4815 ± 0.0022 |

## Known bugs and findings

### 1. mAP > 1 in the original notebook (fixed)

The original `clip-blip-2.ipynb` reported impossible mAP values (e.g. mAP@10 = 1.67) because the AP denominator used `k` instead of `min(n_relevant, k)`, and top-K fetch was off. The corrected implementation now lives in [`src/metrics.py`](src/metrics.py) and is the single source of truth. Sanity check: all reported mAPs are in [0, 1].

### 2. BLIP-2 ITM re-ranking hurts on short product captions

The problem statement calls for BLIP-2 ITM re-ranking as the online step 4. We implemented it (`Salesforce/blip2-itm-vit-g`) and ran it on the full 14k query set. Result: **R@10 essentially unchanged (0.881 vs 0.880), but NDCG@10 drops 5 pts (0.578 → 0.551) and mAP@10 drops 3 pts (0.479 → 0.449).**

Cause: top-50 ANN candidates already share very similar short captions ("black floral print dress"). ITM can't discriminate between them, so it effectively randomizes the ordering and replaces a strong visual signal with noise. A low-weight blend (`itm_weight=0.2`) brings the numbers back to ≈ baseline but adds no value.

This is reported as a **negative ablation** in the final report.

### 3. The original notebooks' "std = 0 across 4 seeds" was misleading

The original code looped over seeds [83, 588, 527, 33] but nothing in the inner loop actually depended on the seed — the gallery index was fixed, the query order didn't change retrieval. So `std = 0` was an artifact of the loop, not a meaningful variance estimate.

How we handle this in the final results:
- For **trained** conditions (C-hn): real model-weight variance via two full fine-tuning runs with different seeds (83, 588). One more seed (527) was queued on Kaggle but cut off by the 12 h kernel limit.
- For **non-trained** conditions (A, B, C-vanilla): query-set bootstrap (`--bootstrap 4` flag, 80% resampling with replacement over 4 seeds). Justified per the prof's "any random component, justify in viva" guidance.

### 4. Caption coverage

`captions.json` covers train + gallery splits (38,494 images) but **not the query split** (14,218 images). That's correct for our pipeline — queries don't need pre-computed captions because we encode them with the visual branch only. If you switch to a query-side text-based scheme, you'd need to caption queries at runtime.

## Open problems / what's pending

- **Ablation table eval** — running locally now (~15 min) for A_alpha1.0, B_alpha0.7, B_alpha0.5, C_alpha0.7, C_alpha0.5, C_alpha0.5_hn. JSONs land in `artifacts/eval_*.json` with bootstrap mean±std.
- **Third training seed (527)** — was killed by Kaggle's 12 h limit. Decision: report mean±std over seeds 83 + 588 only.
- **Local hardware quirk** — on this laptop the dGPU mode needs a fresh boot; integrated-GPU mode silently fails (`nvidia-smi` errors with "NVML not available"). Verify with `nvidia-smi` before any GPU-bound run.
- **Final report draft** — pending the ablation eval results.
- **Streamlit demo** — implemented end-to-end but not yet user-tested with the hard-neg checkpoint.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, torch 2.10+cu128 on a single RTX 4060 Laptop (8 GB VRAM).

## Data + checkpoints

Not in this repo (sizes range from 480 KB to 5 GB). Pull from Kaggle:

```bash
kaggle datasets download ashok1145/vr-fproj                 # DeepFashion (extract to vr_final_proj_dataset/)
kaggle datasets download taralsanka/blip-updated-captions   # captions.json (BLIP-2)
kaggle datasets download taralsanka/best-yolo-pt            # YOLO weights
kaggle datasets download taralsanka/clip-saved-outputs      # Vanilla fine-tuned CLIP + 5 HNSW indices
kaggle datasets download ashok1145/clip-hn-checkpoint       # Hard-neg fine-tuned CLIP (produced by train_hn.py)
```

Place under `artifacts/` and `yolo/` as appropriate.

## Running locally

```bash
# Train hard-neg CLIP (default seed 83, ~1 h on 4060)
python train_hn.py
# Multi-seed:  python train_hn.py --seed 588 --suffix _seed588

# Rebuild C-condition HNSW indices with the new checkpoint
python rebuild_indices_hn.py

# Batch eval, condition C with hard-neg
python demo_batch_eval.py --condition C_alpha0.7_hn

# Same with bootstrap mean±std and BLIP-2 ITM rerank
python demo_batch_eval.py --condition C_alpha0.7_hn --bootstrap 4 \
    --rerank --rerank_mode blend --itm_weight 0.2

# Streamlit demo
streamlit run app.py
```

## Running on Kaggle

Open `kaggle_overnight.ipynb` (Web UI: File → Import Notebook → from this GitHub URL), attach the 5 datasets listed above, enable GPU + Internet, Run All. Or use the CLI:

```bash
kaggle kernels push -p .
kaggle kernels status ashok1145/vr-final-overnight
kaggle kernels output ashok1145/vr-final-overnight -p kaggle_results/
```

## Evaluation conditions

| Condition | α | Model | Notes |
| --- | --- | --- | --- |
| `A_alpha1.0`    | 1.0 | Frozen CLIP (vision only) | Baseline |
| `B_alpha0.7`    | 0.7 | Frozen CLIP + BLIP-2 captions | Frozen + caption fusion |
| `B_alpha0.5`    | 0.5 | Frozen CLIP + BLIP-2 captions | More weight on text — typically worse |
| `C_alpha0.7`    | 0.7 | Fine-tuned CLIP (InfoNCE only) + captions | Vanilla fine-tune |
| `C_alpha0.5`    | 0.5 | Fine-tuned CLIP (InfoNCE only) + captions | |
| `C_alpha0.7_hn` | 0.7 | Fine-tuned CLIP (InfoNCE + Triplet + hard-neg) + captions | **Headline model** |
| `C_alpha0.5_hn` | 0.5 | Same with α=0.5 | |
