"""Online YOLO crop using the DeepFashion-tuned single-class detector
(`yolo/best_yolo8l.pt`, trained via `notebooks/yolo-final-proj.ipynb`).

This module exposes the detector as an **optional ablation** in
`demo_batch_eval.py --yolo`. The Streamlit demo no longer uses it; for the
interactive flow we use the Fashionpedia multi-class detector
(`src/clothing_detect.py`) instead, which supports per-item user choice.

Kept as a documented extra-credit artifact (per project update #1, YOLO
fine-tuning is optional but we did it). The top-1 highest-confidence
detection is returned; if YOLO finds nothing, callers fall back to the full
image."""
from PIL import Image
from ultralytics import YOLO

from .config import BBOX_PAD, YOLO_WEIGHTS


_yolo_model = None


def _get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    return _yolo_model


def detect_all(pil_image, conf=0.25):
    """Returns list of (x1, y1, x2, y2, confidence) for every detection
    (single-class YOLO, so class is always 'clothing'). Empty list if nothing found."""
    model = _get_model()
    results = model.predict(pil_image, conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes
    out = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        out.append((int(x1), int(y1), int(x2), int(y2), float(boxes.conf[i])))
    return out


def detect(pil_image, conf=0.25):
    """Returns (x1, y1, x2, y2, confidence, class_id) for the highest-confidence
    detection, or None. Kept for backward compatibility with demo_batch_eval.py."""
    dets = detect_all(pil_image, conf=conf)
    if not dets:
        return None
    best = max(dets, key=lambda d: d[4])
    return best[0], best[1], best[2], best[3], best[4], 0


def _union_bbox(dets, W, H):
    """Union of all detection bboxes; falls back to the full image."""
    if not dets:
        return 0, 0, W, H
    x1 = max(0, min(d[0] for d in dets))
    y1 = max(0, min(d[1] for d in dets))
    x2 = min(W, max(d[2] for d in dets))
    y2 = min(H, max(d[3] for d in dets))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, W, H
    return x1, y1, x2, y2


def body_region_crops(pil_image, conf=0.25, pad=BBOX_PAD):
    """Returns a dict {'upper': PIL, 'lower': PIL, 'full': PIL, 'person_bbox': (x1,y1,x2,y2),
    'detections': [...]} where each crop is the corresponding body region.

    upper = top  55% of the person bbox
    lower = bottom 55% of the person bbox
    full  = whole person bbox (with a small pad)
    """
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image).convert("RGB")
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    W, H = pil_image.size
    dets = detect_all(pil_image, conf=conf)
    px1, py1, px2, py2 = _union_bbox(dets, W, H)

    # Pad the person bbox a little for the full-body crop only
    pw, ph = px2 - px1, py2 - py1
    px_pad = int(pw * pad); py_pad = int(ph * pad)
    full_x1 = max(0, px1 - px_pad); full_y1 = max(0, py1 - py_pad)
    full_x2 = min(W, px2 + px_pad); full_y2 = min(H, py2 + py_pad)

    # Heuristic splits (overlap a little to keep waistline garments intact)
    upper_y2 = py1 + int(ph * 0.55)
    lower_y1 = py1 + int(ph * 0.45)

    crops = {
        "upper": pil_image.crop((px1, py1, px2, upper_y2)),
        "lower": pil_image.crop((px1, lower_y1, px2, py2)),
        "full":  pil_image.crop((full_x1, full_y1, full_x2, full_y2)),
        "person_bbox": (px1, py1, px2, py2),
        "detections": dets,
    }
    return crops


def yolo_crop(pil_image, conf=0.25, pad=BBOX_PAD, region="full"):
    """End-to-end: PIL image -> cropped PIL image for the chosen body region.

    region: 'upper' | 'lower' | 'full' (default 'full' = legacy behaviour but using
    the union of YOLO detections instead of just the top-1)."""
    crops = body_region_crops(pil_image, conf=conf, pad=pad)
    info = {
        "bbox": crops["person_bbox"],
        "n_detections": len(crops["detections"]),
        "region": region,
    }
    return crops[region], info
