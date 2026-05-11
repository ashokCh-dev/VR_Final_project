# Visual Product Search Engine

Image-to-image retrieval over DeepFashion In-Shop, with CLIP fine-tuning and BLIP-2 ITM re-ranking.

## Pipeline

```
Query image
  └─ YOLO crop ──► CLIP image encoder ──► HNSW ANN search ──► top-K candidates
                                                 │
                                                 └─ (optional) BLIP-2 ITM rerank ──► top-K final
```

Gallery is indexed offline: each catalog image goes YOLO/bbox crop → CLIP image encoder → fused with BLIP-2-captioned text encoder vector (α·image + (1−α)·text, L2-normalized) → HNSW.

## What's in this repo

| File | Purpose |
| --- | --- |
| `src/config.py` | Single source of truth for paths, model names, condition keys. Auto-detects Kaggle vs local. |
| `src/data_utils.py` | DeepFashion split/bbox parsing, bbox crop helper, item_id-from-path. |
| `src/metrics.py` | Corrected Recall@K / NDCG@K / mAP@K (fixes the `n_relevant` and k-fetch bugs from the original notebooks). |
| `src/clip_encoder.py` | OpenCLIP load + image/text encode + fused embedding. |
| `src/yolo_crop.py` | YOLO online crop with confidence threshold + fallback. |
| `src/retrieve.py` | HNSW build / save / load / search. |
| `src/blip2.py` | BLIP-2 captioning (8-bit) + BLIP-2 ITM scorer (BLIP-1 fallback). |
| `src/rerank.py` | ITM-based re-ranking (blend / itm / product modes). |
| `train_hn.py` | Hard-negative CLIP fine-tuning (4060-tuned). `--seed`, `--suffix`. |
| `rebuild_indices_hn.py` | Rebuild C-condition HNSW indices using a fresh checkpoint. |
| `demo_batch_eval.py` | Batch evaluation. `--condition`, `--rerank`, `--yolo`, path overrides. |
| `app.py` | Streamlit demo (upload → YOLO crop → confirm → retrieve → optional rerank). |
| `run_overnight.sh` | Sequences full ITM eval + multi-seed retraining. |
| `clip-blip-2.ipynb`, `clip-hard-negative-mining.ipynb`, `clip_with_saved.ipynb` | Original research notebooks (kept as the lab log). |

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
```

Place under `artifacts/` and `yolo/` as appropriate. The hard-neg checkpoint (`clip_finetuned_hn.pt`) is produced by `python train_hn.py`.

## Running

```bash
# Train hard-neg CLIP (default seed 83, ~1 h on 4060)
python train_hn.py
# Rebuild C-condition HNSW indices with the new checkpoint
python rebuild_indices_hn.py
# Batch eval, condition C with hard-neg
python demo_batch_eval.py --condition C_alpha0.7_hn
# Same with BLIP-2 ITM rerank (blended)
python demo_batch_eval.py --condition C_alpha0.7_hn --rerank --rerank_mode blend --itm_weight 0.2
# Streamlit demo
streamlit run app.py
```

On Kaggle, attach the four datasets above + a dataset containing `clip_finetuned_hn.pt`, then run the same commands — `src/config.py` auto-detects the environment.

## Evaluation conditions

| Condition | α | Model |
| --- | --- | --- |
| `A_alpha1.0`    | 1.0 | Frozen CLIP (vision only) |
| `B_alpha0.7`    | 0.7 | Frozen CLIP + BLIP-2 captions |
| `B_alpha0.5`    | 0.5 | Frozen CLIP + BLIP-2 captions |
| `C_alpha0.7`    | 0.7 | Fine-tuned CLIP (vanilla InfoNCE) + captions |
| `C_alpha0.5`    | 0.5 | Fine-tuned CLIP (vanilla InfoNCE) + captions |
| `C_alpha0.7_hn` | 0.7 | Fine-tuned CLIP (InfoNCE + Triplet + hard-neg mining) + captions |
| `C_alpha0.5_hn` | 0.5 | Same with α=0.5 |
