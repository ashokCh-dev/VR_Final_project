"""Path and model configuration.

Auto-detects Kaggle vs. local. Use `resolve_artifact(name)` for any artifact
that might live in either /kaggle/input (read-only, downloaded datasets) or
the writable artifacts/ directory.
"""
import os
from pathlib import Path

import torch

# ── Environment detection ───────────────────────────────────────────────────
IS_KAGGLE = Path("/kaggle/input").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if IS_KAGGLE:
    # Read-only mounted dataset paths on Kaggle.
    DATASET_ROOT  = Path("/kaggle/input/datasets/ashok1145/vr-fproj/vr_final_proj_dataset")
    CAPTIONS_FILE = Path("/kaggle/input/datasets/taralsanka/blip-updated-captions/captions.json")
    YOLO_WEIGHTS  = Path("/kaggle/input/datasets/taralsanka/best-yolo-pt/best_yolo8l.pt")

    # Vanilla CLIP + the original 5 HNSW indices ship together
    _SAVED_DIR    = Path("/kaggle/input/datasets/taralsanka/clip-saved-outputs")
    CLIP_FT       = _SAVED_DIR / "clip_finetuned.pt"

    # Hard-neg checkpoint must be attached as a separate dataset.
    # We search a few likely slugs (override with HN_CKPT_PATH env var if needed).
    _HN_CANDIDATES = [
        os.environ.get("HN_CKPT_PATH"),
        "/kaggle/input/datasets/ashok1145/clip-hn-checkpoint/clip_finetuned_hn.pt",
        "/kaggle/input/clip-hn-checkpoint/clip_finetuned_hn.pt",
        "/kaggle/input/datasets/ashok1145/clip-finetuned-hn/clip_finetuned_hn.pt",
    ]
    CLIP_FT_HN = next(
        (Path(p) for p in _HN_CANDIDATES if p and Path(p).exists()),
        Path("/kaggle/input/datasets/ashok1145/clip-hn-checkpoint/clip_finetuned_hn.pt"),
    )

    # Writable working dir
    ARTIFACTS_DIR = Path("/kaggle/working/artifacts")
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    # Search order for read-only artifact lookups (e.g. .bin index files)
    _READONLY_DIRS = [_SAVED_DIR]
else:
    PROJ_ROOT     = Path(__file__).resolve().parent.parent
    DATASET_ROOT  = PROJ_ROOT / "vr_final_proj_dataset"
    ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
    CAPTIONS_FILE = ARTIFACTS_DIR / "captions.json"
    YOLO_WEIGHTS  = PROJ_ROOT / "yolo" / "best_yolo8l.pt"
    CLIP_FT_HN    = ARTIFACTS_DIR / "clip_finetuned_hn.pt"
    CLIP_FT       = ARTIFACTS_DIR / "clip_finetuned.pt"
    _READONLY_DIRS = []

IMG_ROOT   = DATASET_ROOT / "img" / "img"
SPLIT_FILE = DATASET_ROOT / "eval" / "list_eval_partition.txt"
BBOX_FILE  = DATASET_ROOT / "Anno" / "list_bbox_inshop.txt"


def resolve_artifact(name):
    """Find an artifact by filename. Checks the writable ARTIFACTS_DIR first
    (so freshly-built _hn files take precedence) then the read-only Kaggle
    dataset dirs. Returns ARTIFACTS_DIR/name as a fallback for files we expect
    to write."""
    p = ARTIFACTS_DIR / name
    if p.exists():
        return p
    for d in _READONLY_DIRS:
        cand = d / name
        if cand.exists():
            return cand
    return p


# ── Index registry ──────────────────────────────────────────────────────────
INDEX_FILES = {
    "A_alpha1.0":    ("gallery_index_A.bin",            "gallery_meta_A.json",            1.0, "frozen"),
    "B_alpha0.7":    ("gallery_index_B_alpha07.bin",    "gallery_meta_B_alpha07.json",    0.7, "frozen"),
    "B_alpha0.5":    ("gallery_index_B_alpha05.bin",    "gallery_meta_B_alpha05.json",    0.5, "frozen"),
    "C_alpha0.7":    ("gallery_index_C_alpha07.bin",    "gallery_meta_C_alpha07.json",    0.7, "ft"),
    "C_alpha0.5":    ("gallery_index_C_alpha05.bin",    "gallery_meta_C_alpha05.json",    0.5, "ft"),
    "C_alpha0.7_hn": ("gallery_index_C_alpha07_hn.bin", "gallery_meta_C_alpha07_hn.json", 0.7, "ft_hn"),
    "C_alpha0.5_hn": ("gallery_index_C_alpha05_hn.bin", "gallery_meta_C_alpha05_hn.json", 0.5, "ft_hn"),
}

# ── Model + retrieval config ────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL    = "ViT-L-14"
CLIP_PRETRAIN = "openai"
EMB_DIM       = 768
BBOX_PAD      = 0.05
TOP_K_LIST    = [5, 10, 15]
DEFAULT_ALPHA = 0.7
DEFAULT_K     = 10
RERANK_K      = 50

BLIP2_CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
BLIP2_ITM_MODEL     = "Salesforce/blip2-itm-vit-g"
BLIP_ITM_MODEL      = "Salesforce/blip-itm-large-coco"  # BLIP-1 fallback
