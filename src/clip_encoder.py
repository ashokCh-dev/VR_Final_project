"""CLIP loading + fused (α·image + (1-α)·text) embedding."""
from typing import Sequence

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import CLIP_MODEL, CLIP_PRETRAIN, DEVICE


class ClipEncoder:
    def __init__(self, checkpoint_path=None, device=DEVICE):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAIN
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    @torch.no_grad()
    def encode_image(self, pil_or_list):
        """Single PIL image or a list of PIL images -> normalized embedding tensor on CPU."""
        if isinstance(pil_or_list, Image.Image):
            pil_or_list = [pil_or_list]
        tensors = torch.stack([self.preprocess(im) for im in pil_or_list]).to(self.device)
        emb = F.normalize(self.model.encode_image(tensors).float(), dim=-1)
        return emb

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]):
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(list(texts)).to(self.device)
        emb = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
        return emb

    @torch.no_grad()
    def fuse(self, img_emb, txt_emb, alpha):
        if alpha == 1.0:
            return F.normalize(img_emb, dim=-1)
        if alpha == 0.0:
            return F.normalize(txt_emb, dim=-1)
        return F.normalize(alpha * img_emb + (1.0 - alpha) * txt_emb, dim=-1)

    @torch.no_grad()
    def embed_query(self, pil_image, caption=None, alpha=1.0):
        """End-to-end: PIL image (+ optional caption) -> 1-D numpy embedding."""
        img_emb = self.encode_image(pil_image)
        if alpha < 1.0 and caption:
            txt_emb = self.encode_text(caption)
            emb = self.fuse(img_emb, txt_emb, alpha)
        else:
            emb = img_emb
        return emb.cpu().numpy().astype("float32")


@torch.no_grad()
def embed_rows(rows, captions, encoder, bbox_map, img_root, alpha,
               batch_size=64, desc="Embedding"):
    """Batched fused-embedding over a list of split rows. Used for gallery (re)builds."""
    from .data_utils import bbox_crop

    all_embs, all_ids, all_names = [], [], []
    for i in tqdm(range(0, len(rows), batch_size), desc=desc, leave=False):
        batch = rows[i:i + batch_size]
        crops, texts, ids, names = [], [], [], []
        for r in batch:
            p = img_root / r["image_name"]
            if not p.exists():
                continue
            try:
                bbox = bbox_map.get(r["image_name"])
                crop = bbox_crop(p, bbox) if bbox else Image.open(p).convert("RGB")
                crops.append(crop)
                texts.append(captions.get(r["image_name"], ""))
                ids.append(r["item_id"])
                names.append(r["image_name"])
            except Exception:
                continue
        if not crops:
            continue
        img_emb = encoder.encode_image(crops)
        txt_emb = encoder.encode_text(texts) if alpha < 1.0 else None
        fused = encoder.fuse(img_emb, txt_emb if txt_emb is not None else img_emb, alpha)
        all_embs.append(fused.cpu().numpy())
        all_ids.extend(ids); all_names.extend(names)
    import numpy as np
    return np.vstack(all_embs).astype("float32"), all_ids, all_names
