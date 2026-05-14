"""
Streamlit demo for the Visual Product Search Engine.

Run:
    streamlit run app.py

Flow:
  1. Upload a query image.
  2. Fashionpedia multi-class detector localises every clothing item
     (jacket, pants, dress, hat, …) and the user picks one labelled item.
     Items whose category has no DeepFashion gallery equivalent are flagged
     as "Not in catalog". A "Use full image" escape hatch covers edge cases
     where the detector misses the garment entirely.
  3. CLIP encodes the chosen crop. HNSW returns top-K candidates.
  4. (Optional) BLIP-2 ITM re-rank is applied.
  5. Results are shown as a grid with item_id + similarity score.
"""
import time

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


@st.cache_resource(show_spinner="Loading Fashionpedia detector (first run downloads ~140 MB)…")
def warm_fashion_detector():
    """Trigger one-time Fashionpedia load (cached). Function references
    (`detect_items`, `crop_item`) are imported fresh at the call site so
    edits to the module are picked up on Streamlit auto-rerun."""
    from src.clothing_detect import _load
    _load()
    return True


# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Pipeline settings")

_keys = list(config.INDEX_FILES.keys())
_default_cond = "C_alpha0.7_hn" if "C_alpha0.7_hn" in _keys else "C_alpha0.7"
condition = st.sidebar.selectbox(
    "Condition", _keys, index=_keys.index(_default_cond),
    help="A=vision-only, B=frozen+captions, C=fine-tuned+captions, _hn=hard-neg model",
)
detect_conf = st.sidebar.slider(
    "Detector confidence", 0.05, 0.6, 0.25, 0.05,
    help="Lower → more candidate detections; higher → only confident ones.",
)
show_parts = st.sidebar.checkbox(
    "Show garment parts (sleeve, collar, neckline, …)",
    value=False,
    help=(
        "Off (default) shows only whole garments and accessories — parts "
        "like 'sleeve' or 'neckline' are filtered out because they don't "
        "correspond to anything searchable in the DeepFashion gallery. "
        "Turn on for debugging or if no whole garment was detected."
    ),
)
top_k = st.sidebar.slider("Top-K results", 5, 30, config.DEFAULT_K)
use_rerank = st.sidebar.checkbox("Use BLIP-2 ITM re-rank", value=False,
                                  help="Re-rank top-50 ANN candidates by image-text matching. "
                                       "Note: known to hurt mAP on short product captions.")
prefer_hn = st.sidebar.checkbox(
    "Prefer hard-neg checkpoint (if available)",
    value=config.CLIP_FT_HN.exists(),
    help=f"{config.CLIP_FT_HN.name} exists: {config.CLIP_FT_HN.exists()}",
)

# ── Session state ───────────────────────────────────────────────────────────
for k in [
    "original_pil",
    "detected_items",     # list of Fashionpedia detections
    "selected_item",      # chosen detection dict
    "query_crop",         # final crop sent to CLIP
    "query_label",        # text label for display
    "confirmed",
    "results",
]:
    st.session_state.setdefault(k, None)

# Reset when condition or parts-toggle changes
_state_key = (condition, show_parts)
if st.session_state.get("_last_state_key") != _state_key:
    prev = st.session_state.get("_last_state_key")
    st.session_state["_last_state_key"] = _state_key
    st.session_state["results"] = None
    # If parts toggle changed, also clear cached detections
    if prev is None or prev[1] != show_parts:
        st.session_state["confirmed"] = False
        st.session_state["detected_items"] = None

# ── Main layout ─────────────────────────────────────────────────────────────
st.title("Visual Product Search Engine")
st.caption(
    "DeepFashion In-Shop · clothing detector → CLIP fused embedding → HNSW ANN "
    "→ (optional) BLIP-2 ITM re-rank"
)

uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    if st.session_state["original_pil"] is None or uploaded.name != st.session_state.get("_last_uploaded"):
        st.session_state["original_pil"] = pil
        st.session_state["detected_items"] = None
        st.session_state["selected_item"] = None
        st.session_state["query_crop"] = None
        st.session_state["query_label"] = None
        st.session_state["confirmed"] = False
        st.session_state["results"] = None
        st.session_state["_last_uploaded"] = uploaded.name

# ════════════════════════════════════════════════════════════════════════════
# Step 1 — detection + user pick
# ════════════════════════════════════════════════════════════════════════════
if st.session_state["original_pil"] is not None and not st.session_state["confirmed"]:

    st.subheader("Step 1 — Detect clothing items and pick one to search")
    warm_fashion_detector()
    from src.clothing_detect import detect_items, crop_item, is_in_catalog, catalog_categories

    if st.session_state["detected_items"] is None:
        with st.spinner("Running Fashionpedia detector…"):
            items = detect_items(
                st.session_state["original_pil"], conf=detect_conf,
                include_parts=show_parts,
            )
        st.session_state["detected_items"] = items

    items = st.session_state["detected_items"]

    c_orig, c_grid = st.columns([1, 2])
    with c_orig:
        st.image(st.session_state["original_pil"], caption="Original",
                 use_container_width=True)

    with c_grid:
        if not items:
            st.warning(
                "No clothing items detected above the confidence threshold. "
                "Try lowering the detector confidence in the sidebar, enable "
                "*Show garment parts* to see partial detections, or click "
                "**Use full image (no crop)** below."
            )
        else:
            n_searchable = sum(1 for it in items if is_in_catalog(it["label"]))
            n_oos = len(items) - n_searchable
            msg = (
                f"Detected {len(items)} item{'s' if len(items) != 1 else ''}. "
                f"{n_searchable} searchable in the DeepFashion catalog"
            )
            if n_oos:
                msg += f"; {n_oos} marked out-of-stock (accessories, no equivalent in gallery)"
            msg += ". Click a searchable one to retrieve."
            st.success(msg)

            cols_per_row = 4
            rows = (len(items) + cols_per_row - 1) // cols_per_row
            for row_idx in range(rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    i = row_idx * cols_per_row + j
                    if i >= len(items):
                        break
                    it = items[i]
                    thumb = crop_item(st.session_state["original_pil"], it)
                    in_cat = is_in_catalog(it["label"])
                    with cols[j]:
                        st.image(thumb, use_container_width=True)
                        label_disp = f"**{it['label']}**  ({it['score']:.2f})"
                        if in_cat:
                            cats = catalog_categories(it["label"])
                            st.caption(label_disp + f"\n\n_in catalog: {', '.join(cats)}_")
                            if st.button(f"Search this  ▶", key=f"pick_{i}"):
                                st.session_state["selected_item"] = it
                                st.session_state["query_crop"] = crop_item(
                                    st.session_state["original_pil"], it
                                )
                                st.session_state["query_label"] = (
                                    f"{it['label']} ({it['score']:.2f})"
                                )
                                st.session_state["confirmed"] = True
                                st.session_state["results"] = None
                                st.rerun()
                        else:
                            st.caption(label_disp + "\n\n:gray[**Not in catalog (out of stock)**]")
                            st.button(
                                "Not stocked",
                                key=f"pick_{i}",
                                disabled=True,
                                help=(
                                    f"DeepFashion In-Shop gallery has no "
                                    f"'{it['label']}' items — accessories "
                                    "(hats, glasses, shoes, bags, ties, …) "
                                    "are out of catalogue scope."
                                ),
                            )

    b1, b2 = st.columns(2)
    if b1.button("Re-run detector with lower confidence"):
        with st.spinner("Re-running detector…"):
            items = detect_items(
                st.session_state["original_pil"],
                conf=max(0.05, detect_conf - 0.10),
                include_parts=show_parts,
            )
        st.session_state["detected_items"] = items
        st.rerun()
    if b2.button("Use full image (no crop)"):
        st.session_state["query_crop"] = st.session_state["original_pil"]
        st.session_state["query_label"] = "full image (no crop)"
        st.session_state["confirmed"] = True
        st.session_state["results"] = None
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# Step 2 & 3 — encode + retrieve (+ rerank)
# ════════════════════════════════════════════════════════════════════════════
if st.session_state["confirmed"]:
    query_crop = st.session_state["query_crop"]
    query_label = st.session_state["query_label"] or "selected crop"
    st.subheader(f"Step 2 — Retrieved products  (searching for {query_label})")

    bin_name, meta_name, alpha, model_type = config.INDEX_FILES[condition]

    if st.session_state["results"] is None:
        encoder = get_encoder(model_type, prefer_hn)
        index, gal_ids, gal_names, _, _ = get_index(condition)
        captions_dict = get_captions_dict() if (alpha < 1.0 or use_rerank) else {}

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
            "query_label": query_label,
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

    with st.expander("Query crop used for this search"):
        st.image(query_crop, caption=f"{query_label} (sent to CLIP)", width=240)

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
        for k in [
            "original_pil", "detected_items", "selected_item",
            "query_crop", "query_label",
            "confirmed", "results", "_last_uploaded",
        ]:
            st.session_state[k] = None
        st.rerun()
