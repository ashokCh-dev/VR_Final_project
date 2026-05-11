"""
Streamlit demo for the Visual Product Search Engine.

Run:
    streamlit run app.py

Flow:
  1. Upload a query image.
  2. YOLO crop is shown side-by-side with the original.
  3. User clicks "Confirm crop" (or "Re-crop with lower YOLO conf" / "Use full image").
  4. CLIP encodes the confirmed crop. HNSW returns top-K candidates.
  5. (Optional) BLIP-1 ITM re-rank is applied.
  6. Results are shown as a grid with item_id + similarity score.
"""
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src import config
from src.data_utils import load_captions

st.set_page_config(page_title="Visual Product Search", layout="wide")

# ── Cached heavy loads ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP encoder…")
def get_encoder(model_type: str, prefer_hn: bool):
    from src.clip_encoder import ClipEncoder
    if model_type == "frozen":
        return ClipEncoder(checkpoint_path=None)
    if model_type == "ft_hn":
        return ClipEncoder(checkpoint_path=config.CLIP_FT_HN)
    ckpt = config.CLIP_FT_HN if prefer_hn and config.CLIP_FT_HN.exists() else config.CLIP_FT
    return ClipEncoder(checkpoint_path=ckpt)


@st.cache_resource(show_spinner="Loading gallery index…")
def get_index(condition_key: str):
    from src.retrieve import load_named
    return load_named(condition_key)


@st.cache_resource
def get_captions_dict():
    return load_captions()


@st.cache_resource(show_spinner="Loading YOLO…")
def get_yolo():
    from src.yolo_crop import yolo_crop, _get_model
    _get_model()
    return yolo_crop


# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Pipeline settings")

_keys = list(config.INDEX_FILES.keys())
_default_cond = "C_alpha0.7_hn" if "C_alpha0.7_hn" in _keys else "C_alpha0.7"
condition = st.sidebar.selectbox(
    "Condition", _keys, index=_keys.index(_default_cond),
    help="A=vision-only, B=frozen+captions, C=fine-tuned+captions, _hn=hard-neg model",
)
yolo_conf = st.sidebar.slider("YOLO confidence", 0.05, 0.6, 0.25, 0.05)
top_k = st.sidebar.slider("Top-K results", 5, 30, config.DEFAULT_K)
use_rerank = st.sidebar.checkbox("Use BLIP-1 ITM re-rank", value=False,
                                  help="Re-rank top-50 ANN candidates by image-text matching")
prefer_hn = st.sidebar.checkbox(
    "Prefer hard-neg checkpoint (if available)",
    value=config.CLIP_FT_HN.exists(),
    help=f"clip_finetuned_hn.pt exists: {config.CLIP_FT_HN.exists()}",
)

# ── Initial state ───────────────────────────────────────────────────────────
for k in ["original_pil", "crop_pil", "crop_meta", "confirmed", "results"]:
    st.session_state.setdefault(k, None)

# Reset when condition changes
if st.session_state.get("_last_condition") != condition:
    st.session_state["_last_condition"] = condition
    st.session_state["results"] = None

# ── Main layout ─────────────────────────────────────────────────────────────
st.title("Visual Product Search Engine")
st.caption(
    "DeepFashion In-Shop · YOLO crop → CLIP fused embedding → HNSW ANN → "
    "(optional) BLIP-1 ITM re-rank"
)

uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    if st.session_state["original_pil"] is None or uploaded.name != st.session_state.get("_last_uploaded"):
        st.session_state["original_pil"] = pil
        st.session_state["crop_pil"] = None
        st.session_state["confirmed"] = False
        st.session_state["results"] = None
        st.session_state["_last_uploaded"] = uploaded.name

# ── Step 1: YOLO crop with user confirmation ────────────────────────────────
if st.session_state["original_pil"] is not None and not st.session_state["confirmed"]:
    st.subheader("Step 1 — Detect and crop the product")
    yolo_crop = get_yolo()

    if st.session_state["crop_pil"] is None:
        with st.spinner("Running YOLO…"):
            crop, meta = yolo_crop(st.session_state["original_pil"], conf=yolo_conf)
        st.session_state["crop_pil"] = crop
        st.session_state["crop_meta"] = meta

    c1, c2 = st.columns(2)
    with c1:
        st.image(st.session_state["original_pil"], caption="Original", use_container_width=True)
    with c2:
        meta = st.session_state["crop_meta"]
        cap = f"YOLO crop (conf={meta['score']:.2f})" if meta else "YOLO found nothing — using full image"
        st.image(st.session_state["crop_pil"], caption=cap, use_container_width=True)

    b1, b2, b3 = st.columns(3)
    if b1.button("Confirm crop", type="primary"):
        st.session_state["confirmed"] = True
        st.session_state["results"] = None
        st.rerun()
    if b2.button("Re-crop (lower YOLO conf)"):
        with st.spinner("Re-running YOLO with lower conf…"):
            crop, meta = yolo_crop(st.session_state["original_pil"], conf=max(0.05, yolo_conf - 0.1))
        st.session_state["crop_pil"] = crop
        st.session_state["crop_meta"] = meta
        st.rerun()
    if b3.button("Use full image (skip crop)"):
        st.session_state["crop_pil"] = st.session_state["original_pil"]
        st.session_state["crop_meta"] = None
        st.session_state["confirmed"] = True
        st.session_state["results"] = None
        st.rerun()

# ── Step 2 & 3: encode + retrieve (+ rerank) ────────────────────────────────
if st.session_state["confirmed"]:
    st.subheader("Step 2 — Retrieved products")

    bin_name, meta_name, alpha, model_type = config.INDEX_FILES[condition]

    if st.session_state["results"] is None:
        encoder = get_encoder(model_type, prefer_hn)
        index, gal_ids, gal_names, _, _ = get_index(condition)
        captions_dict = get_captions_dict() if (alpha < 1.0 or use_rerank) else {}

        # For the query side: no pre-computed caption, so for α<1 we'd ideally
        # caption-on-the-fly via BLIP-2. To keep the demo snappy and self-contained,
        # we use vision-only encoding on the query (alpha=1.0) regardless. The
        # gallery is still fused at the configured alpha — this matches the
        # standard "asymmetric" image-to-image retrieval setup.
        with st.spinner("CLIP encoding…"):
            t0 = time.time()
            q_emb = encoder.embed_query(st.session_state["crop_pil"], caption="", alpha=1.0)
            enc_ms = (time.time() - t0) * 1000

        from src.retrieve import candidates_from_labels, search
        fetch_k = config.RERANK_K if use_rerank else top_k
        with st.spinner(f"HNSW search (k={fetch_k})…"):
            t0 = time.time()
            labels, distances = search(index, q_emb, k=fetch_k)
            ann_ms = (time.time() - t0) * 1000
        cands = candidates_from_labels(labels, gal_ids, gal_names, distances)[0]

        rerank_ms = 0.0
        if use_rerank:
            from src.rerank import rerank_with_itm
            with st.spinner(f"BLIP-1 ITM re-ranking top-{fetch_k}…"):
                t0 = time.time()
                cands = rerank_with_itm(
                    st.session_state["crop_pil"], cands, captions_dict, top_k=top_k
                )
                rerank_ms = (time.time() - t0) * 1000
        else:
            cands = cands[:top_k]

        st.session_state["results"] = {
            "cands": cands,
            "timing": {"clip_ms": enc_ms, "ann_ms": ann_ms, "rerank_ms": rerank_ms},
            "condition": condition,
            "use_rerank": use_rerank,
        }

    r = st.session_state["results"]
    cands = r["cands"]
    t = r["timing"]
    st.info(
        f"CLIP {t['clip_ms']:.0f} ms · ANN {t['ann_ms']:.0f} ms"
        + (f" · ITM rerank {t['rerank_ms']:.0f} ms" if r["use_rerank"] else "")
        + f" · {len(cands)} results"
    )

    cols_per_row = 5
    rows = (len(cands) + cols_per_row - 1) // cols_per_row
    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            i = row_idx * cols_per_row + j
            if i >= len(cands):
                break
            c = cands[i]
            img_path = config.IMG_ROOT / c["img_name"]
            with cols[j]:
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"missing\n{c['img_name']}")
                score_label = "ITM" if "itm_score" in c else "cos"
                score_val = c.get("itm_score", c.get("ann_score", c.get("score", 0.0)))
                st.caption(
                    f"**#{i+1}** · {c['item_id']}\n\n"
                    f"{score_label}={score_val:.3f}"
                )

    if st.button("New query"):
        for k in ["original_pil", "crop_pil", "crop_meta", "confirmed", "results", "_last_uploaded"]:
            st.session_state[k] = None
        st.rerun()
