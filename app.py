"""
Streamlit demo for the Visual Product Search Engine.

Run:
    streamlit run app.py

Flow:
  1. Upload a query image.
  2. YOLO runs once and produces three crops — upper body, lower body, full body
     (heuristic split of the union-of-detections, since our YOLO is single-class).
  3. User picks a body region via radio and confirms (or re-crops with lower YOLO conf).
  4. CLIP encodes the confirmed crop. HNSW returns top-K candidates.
  5. (Optional) BLIP-2 ITM re-rank is applied.
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
    from src.yolo_crop import body_region_crops, _get_model
    _get_model()
    return body_region_crops


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
use_rerank = st.sidebar.checkbox("Use BLIP-2 ITM re-rank", value=False,
                                  help="Re-rank top-50 ANN candidates by image-text matching. "
                                       "Note: known to hurt mAP on short product captions.")
prefer_hn = st.sidebar.checkbox(
    "Prefer hard-neg checkpoint (if available)",
    value=config.CLIP_FT_HN.exists(),
    help=f"clip_finetuned_hn.pt exists: {config.CLIP_FT_HN.exists()}",
)

# ── Initial state ───────────────────────────────────────────────────────────
for k in ["original_pil", "region_crops", "selected_region", "confirmed", "results"]:
    st.session_state.setdefault(k, None)

# Reset when condition changes
if st.session_state.get("_last_condition") != condition:
    st.session_state["_last_condition"] = condition
    st.session_state["results"] = None

# ── Main layout ─────────────────────────────────────────────────────────────
st.title("Visual Product Search Engine")
st.caption(
    "DeepFashion In-Shop · YOLO crop (upper / lower / full body) → CLIP fused embedding "
    "→ HNSW ANN → (optional) BLIP-2 ITM re-rank"
)

uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    if st.session_state["original_pil"] is None or uploaded.name != st.session_state.get("_last_uploaded"):
        st.session_state["original_pil"] = pil
        st.session_state["region_crops"] = None
        st.session_state["selected_region"] = "full"
        st.session_state["confirmed"] = False
        st.session_state["results"] = None
        st.session_state["_last_uploaded"] = uploaded.name

# ── Step 1: YOLO crop with user confirmation ────────────────────────────────
if st.session_state["original_pil"] is not None and not st.session_state["confirmed"]:
    st.subheader("Step 1 — Detect garment regions and pick the body part to search")
    body_region_crops = get_yolo()

    if st.session_state["region_crops"] is None:
        with st.spinner("Running YOLO…"):
            crops = body_region_crops(st.session_state["original_pil"], conf=yolo_conf)
        st.session_state["region_crops"] = crops

    crops = st.session_state["region_crops"]
    n_dets = len(crops["detections"])

    if n_dets == 0:
        st.warning(
            "YOLO didn't detect anything above the confidence threshold — "
            "the body regions are sliced from the full image instead. "
            "Try lowering YOLO confidence in the sidebar."
        )
    else:
        st.success(
            f"YOLO found {n_dets} clothing detection{'s' if n_dets > 1 else ''}. "
            "Three body-region crops below are heuristic splits of the union of those detections."
        )

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.image(st.session_state["original_pil"], caption="Original", use_container_width=True)
    with c1:
        st.image(crops["upper"], caption="Upper body", use_container_width=True)
    with c2:
        st.image(crops["lower"], caption="Lower body", use_container_width=True)
    with c3:
        st.image(crops["full"],  caption="Full body",  use_container_width=True)

    region_label = st.radio(
        "Search for:",
        options=["Upper body", "Lower body", "Full body"],
        index=2,
        horizontal=True,
        key="region_radio",
    )
    region_key = {"Upper body": "upper", "Lower body": "lower", "Full body": "full"}[region_label]
    st.session_state["selected_region"] = region_key

    b1, b2 = st.columns(2)
    if b1.button(f"Confirm — search for {region_label.lower()}", type="primary"):
        st.session_state["confirmed"] = True
        st.session_state["results"] = None
        st.rerun()
    if b2.button("Re-run YOLO with lower confidence"):
        with st.spinner("Re-running YOLO with lower conf…"):
            crops = body_region_crops(
                st.session_state["original_pil"], conf=max(0.05, yolo_conf - 0.1)
            )
        st.session_state["region_crops"] = crops
        st.rerun()

# ── Step 2 & 3: encode + retrieve (+ rerank) ────────────────────────────────
if st.session_state["confirmed"]:
    region = st.session_state["selected_region"]
    region_label = {"upper": "upper body", "lower": "lower body", "full": "full body"}[region]
    st.subheader(f"Step 2 — Retrieved products  (searching for {region_label})")

    query_crop = st.session_state["region_crops"][region]
    bin_name, meta_name, alpha, model_type = config.INDEX_FILES[condition]

    if st.session_state["results"] is None:
        encoder = get_encoder(model_type, prefer_hn)
        index, gal_ids, gal_names, _, _ = get_index(condition)
        captions_dict = get_captions_dict() if (alpha < 1.0 or use_rerank) else {}

        # Query side: no pre-computed caption, so we use vision-only encoding (alpha=1.0
        # for the query). The gallery is still fused at the configured alpha — standard
        # asymmetric image-to-image retrieval.
        with st.spinner("CLIP encoding…"):
            t0 = time.time()
            q_emb = encoder.embed_query(query_crop, caption="", alpha=1.0)
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
            with st.spinner(f"BLIP-2 ITM re-ranking top-{fetch_k}…"):
                t0 = time.time()
                cands = rerank_with_itm(query_crop, cands, captions_dict, top_k=top_k)
                rerank_ms = (time.time() - t0) * 1000
        else:
            cands = cands[:top_k]

        st.session_state["results"] = {
            "cands": cands,
            "timing": {"clip_ms": enc_ms, "ann_ms": ann_ms, "rerank_ms": rerank_ms},
            "condition": condition,
            "region": region,
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

    # Show the actual crop that was used
    with st.expander("Query crop used for this search"):
        st.image(query_crop, caption=f"{region_label} (sent to CLIP)", width=240)

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
                score_label = "ITM" if "combined_score" in c else "cos"
                score_val = c.get("combined_score", c.get("ann_score", c.get("score", 0.0)))
                st.caption(
                    f"**#{i+1}** · {c['item_id']}\n\n"
                    f"{score_label}={score_val:.3f}"
                )

    if st.button("New query"):
        for k in ["original_pil", "region_crops", "selected_region", "confirmed",
                  "results", "_last_uploaded"]:
            st.session_state[k] = None
        st.rerun()
