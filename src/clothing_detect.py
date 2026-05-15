"""Multi-class clothing detector for the inference picker.

Uses `valentinafeve/yolos-fashionpedia` from HuggingFace — a YOLOS detector
fine-tuned on Fashionpedia (46 fashion categories: shirt, jacket, pants,
skirt, dress, tie, hat, shoe, bag, etc.). Used at inference time only, to let
the user pick which clothing item in a multi-garment image to search for.

Crops are padded by `BBOX_PAD` (5 %) to match the offline-gallery crop
convention so query crops stay in the same distribution the fine-tuned CLIP
encoder was trained on.
"""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from .config import BBOX_PAD, DEVICE

_MODEL_ID = "valentinafeve/yolos-fashionpedia"
_processor = None
_model = None

# Fashionpedia's 46-class taxonomy mixes whole garments, accessories, and
# garment *parts* (sleeve, neckline, collar, lapel, hood, pocket, zipper, …)
# plus decorative motifs (ruffle, bow, sequin, …). For "search for a whole
# product" the parts/decorations are noise — there is no "sleeve only"
# product in DeepFashion to retrieve — so we allowlist the wearable classes.
_GARMENT_LABELS = {
    # whole garments
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress",
    "jumpsuit", "cape",
    # accessories that are themselves retrievable products
    "glasses", "hat", "headband, head covering, hair accessory", "tie",
    "glove", "watch", "belt", "leg warmer", "tights, stockings", "sock",
    "shoe", "bag, wallet", "scarf", "umbrella",
}

# DeepFashion In-Shop gallery has 17 categories (Blouses_Shirts, Cardigans,
# Denim, Dresses, Graphic_Tees, Jackets_Coats, Jackets_Vests, Leggings, Pants,
# Rompers_Jumpsuits, Shirts_Polos, Shorts, Skirts, Suiting, Sweaters,
# Sweatshirts_Hoodies, Tees_Tanks) — all whole-garment categories, no
# accessories. This mapping says which Fashionpedia label corresponds to one
# or more DeepFashion catalog categories. Labels absent from this dict (hat,
# glasses, shoe, bag, tie, watch, belt, …) have no equivalent in the gallery
# and should be marked "not in catalog" by the UI rather than searched —
# because the HNSW would still return *some* nearest-neighbour result, just
# a visually nonsensical one (e.g. clicking "hat" returns random shirts).
_FASHIONPEDIA_TO_DEEPFASHION = {
    "shirt, blouse":            ["Blouses_Shirts", "Shirts_Polos"],
    "top, t-shirt, sweatshirt": ["Tees_Tanks", "Graphic_Tees", "Sweatshirts_Hoodies"],
    "sweater":                  ["Sweaters"],
    "cardigan":                 ["Cardigans"],
    "jacket":                   ["Jackets_Coats", "Jackets_Vests"],
    "vest":                     ["Jackets_Vests"],
    "pants":                    ["Pants", "Denim", "Suiting"],
    "shorts":                   ["Shorts"],
    "skirt":                    ["Skirts"],
    "coat":                     ["Jackets_Coats"],
    "dress":                    ["Dresses"],
    "jumpsuit":                 ["Rompers_Jumpsuits"],
    "tights, stockings":        ["Leggings"],
}


def is_in_catalog(label: str) -> bool:
    """True if the Fashionpedia label has at least one matching DeepFashion
    catalog category; False otherwise (accessories like hat/glasses/shoe and
    out-of-vocabulary classes return False)."""
    return label in _FASHIONPEDIA_TO_DEEPFASHION


def catalog_categories(label: str):
    """Return the list of DeepFashion category folder names this Fashionpedia
    label maps to. Empty list if the label is not in catalog."""
    return list(_FASHIONPEDIA_TO_DEEPFASHION.get(label, ()))


def auto_crop(pil_image, conf=0.25, pad=BBOX_PAD):
    """Top-1 whole-garment crop for automated pipelines (batch eval, scripts).

    Runs the Fashionpedia detector, filters to whole-garment classes (parts
    like sleeve/neckline excluded), then prefers detections that map to a
    DeepFashion catalog category. Falls back to the full image if nothing
    is detected, so the pipeline never crashes on edge-case inputs.

    Returns `(pil_crop, info)` where `info` is a dict with keys
    `{label, score, bbox, fallback}` — `fallback="full_image"` when no
    detection passed the filter."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image).convert("RGB")

    items = detect_items(pil_image, conf=conf, max_items=8, include_parts=False)

    # Prefer items that have a DeepFashion catalog mapping; otherwise accept
    # any whole-garment detection (still better than the full image).
    in_cat = [it for it in items if it["label"] in _FASHIONPEDIA_TO_DEEPFASHION]
    pool = in_cat if in_cat else items

    if not pool:
        return pil_image, {"label": None, "score": 0.0, "bbox": None, "fallback": "full_image"}

    top = pool[0]  # already sorted by descending confidence
    return crop_item(pil_image, top, pad=pad), {
        "label": top["label"],
        "score": top["score"],
        "bbox":  top["bbox"],
        "fallback": None,
    }


def _load():
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        _model = (
            AutoModelForObjectDetection.from_pretrained(_MODEL_ID).to(DEVICE).eval()
        )
    return _processor, _model


def _iou(box_a, box_b):
    """Intersection-over-union for two (x1, y1, x2, y2) boxes. Returns 0 if
    either box is degenerate."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _per_class_nms(items, iou_thresh=0.5):
    """Per-class non-max suppression. Items must already be sorted by
    descending confidence. Drops any detection whose IoU with a *kept*
    detection of the same class exceeds `iou_thresh`.

    This deduplicates overlapping boxes for the same garment (which the
    Fashionpedia detector often emits at multiple scales) without killing
    genuinely separate instances (e.g. two distinct jackets in a group
    photo, where the boxes don't overlap)."""
    kept = []
    for it in items:
        skip = False
        for k in kept:
            if k["label"] == it["label"] and _iou(k["bbox"], it["bbox"]) > iou_thresh:
                skip = True
                break
        if not skip:
            kept.append(it)
    return kept


def detect_items(pil_image, conf=0.30, max_items=8, include_parts=False,
                 nms_iou=0.5):
    """Run Fashionpedia detector. Returns a list of detection dicts sorted by
    descending confidence:

        [{'label': 'jacket', 'score': 0.87, 'bbox': (x1, y1, x2, y2)}, ...]

    By default, only whole-garment and accessory labels are returned —
    garment-part labels (sleeve, neckline, collar, …) and decorative motif
    labels (ruffle, bow, sequin, …) are filtered out because they don't
    correspond to anything retrievable in the DeepFashion gallery. Pass
    `include_parts=True` to disable the filter and inspect raw detections.

    Per-class NMS (`nms_iou`, default 0.5) deduplicates overlapping boxes for
    the same garment at multiple scales without killing genuine multi-instance
    detections (different garments in distinct image regions are all kept).
    Set `nms_iou=1.0` to disable.

    Empty list if nothing scores above `conf` (after filtering)."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image).convert("RGB")
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    processor, model = _load()
    inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([pil_image.size[::-1]], device=DEVICE)
    results = processor.post_process_object_detection(
        outputs, threshold=conf, target_sizes=target
    )[0]

    items = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = model.config.id2label[label_id.item()]
        if not include_parts and label not in _GARMENT_LABELS:
            continue
        x1, y1, x2, y2 = box.tolist()
        items.append(
            {
                "label": label,
                "score": float(score),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            }
        )
    items.sort(key=lambda d: d["score"], reverse=True)

    if nms_iou < 1.0:
        items = _per_class_nms(items, iou_thresh=nms_iou)

    return items[:max_items]


def crop_item(pil_image, item, pad=BBOX_PAD):
    """Apply 5 % padding (matching the offline gallery convention) and return
    the cropped PIL image for one detection."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image).convert("RGB")
    W, H = pil_image.size
    x1, y1, x2, y2 = item["bbox"]
    w, h = x2 - x1, y2 - y1
    xp = int(w * pad)
    yp = int(h * pad)
    return pil_image.crop(
        (max(0, x1 - xp), max(0, y1 - yp), min(W, x2 + xp), min(H, y2 + yp))
    )
