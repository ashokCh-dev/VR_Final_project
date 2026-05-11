"""Dataset parsing + bbox cropping shared across modules."""
import json
from pathlib import Path

from PIL import Image

from .config import BBOX_FILE, BBOX_PAD, CAPTIONS_FILE, SPLIT_FILE


def parse_split_file(path=SPLIT_FILE):
    with open(path) as f:
        lines = f.readlines()
    rows = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        raw = parts[0]
        img_name = raw[len("img/"):] if raw.startswith("img/") else raw
        rows.append({"image_name": img_name, "item_id": parts[1], "split": parts[2]})
    return rows


def parse_bbox_file(path=BBOX_FILE):
    with open(path) as f:
        lines = f.readlines()
    bboxes = {}
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        raw = parts[0]
        img_name = raw[len("img/"):] if raw.startswith("img/") else raw
        bboxes[img_name] = (int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]))
    return bboxes


def bbox_crop(img_or_path, bbox, pad=BBOX_PAD):
    """Crop with `pad` fractional padding. `img_or_path` may be a PIL image or a path."""
    pil = img_or_path if isinstance(img_or_path, Image.Image) else Image.open(img_or_path).convert("RGB")
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    W, H = pil.size
    x1, y1, x2, y2 = bbox
    px = int((x2 - x1) * pad); py = int((y2 - y1) * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)
    return pil.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else pil


def load_captions(path=CAPTIONS_FILE):
    return json.load(open(path))


def item_id_from_path(image_name):
    """DeepFashion convention: '.../id_00000123/...' -> 'id_00000123'."""
    for part in Path(image_name).parts:
        if part.startswith("id_"):
            return part
    return None
