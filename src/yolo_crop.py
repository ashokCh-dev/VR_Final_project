"""Online YOLO crop: highest-confidence clothing detection, fallback to full image."""
from PIL import Image
from ultralytics import YOLO

from .config import BBOX_PAD, YOLO_WEIGHTS


_yolo_model = None


def _get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    return _yolo_model


def detect(pil_image, conf=0.25):
    """Returns (x1, y1, x2, y2, confidence, class_id) for the most confident detection,
    or None if no box passes the confidence threshold."""
    model = _get_model()
    results = model.predict(pil_image, conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    best = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].tolist()
    return int(x1), int(y1), int(x2), int(y2), float(boxes.conf[best]), int(boxes.cls[best])


def yolo_crop(pil_image, conf=0.25, pad=BBOX_PAD):
    """End-to-end: PIL image -> cropped PIL image. Falls back to original if no detection.
    Returns (cropped_image, detection_dict_or_None)."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image).convert("RGB")
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    det = detect(pil_image, conf=conf)
    if det is None:
        return pil_image, None
    x1, y1, x2, y2, score, cls_id = det
    W, H = pil_image.size
    px = int((x2 - x1) * pad); py = int((y2 - y1) * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return pil_image, {"bbox": None, "score": score, "class_id": cls_id}
    return pil_image.crop((x1, y1, x2, y2)), {
        "bbox": (x1, y1, x2, y2),
        "score": score,
        "class_id": cls_id,
    }
