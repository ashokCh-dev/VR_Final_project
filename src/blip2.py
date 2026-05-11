"""BLIP-2 captioning + BLIP-1 ITM (image-text matching) for re-ranking.

We use BLIP-1 ITM for re-ranking because BLIP-2 doesn't expose a clean ITM head
on its public checkpoints; BLIP-1 ITM is the standard substitute in retrieval
papers. BLIP-2 (8-bit, opt-2.7b) is used only for caption generation of user
queries at inference time.
"""
from PIL import Image

from .config import BLIP_ITM_MODEL, BLIP2_CAPTION_MODEL, BLIP2_ITM_MODEL, DEVICE


# ── Caption (BLIP-2, 8-bit) ──────────────────────────────────────────────────
_caption_model = None
_caption_processor = None


def _load_captioner():
    global _caption_model, _caption_processor
    if _caption_model is not None:
        return _caption_model, _caption_processor

    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    _caption_processor = Blip2Processor.from_pretrained(BLIP2_CAPTION_MODEL)

    load_kwargs = {"torch_dtype": torch.float16}
    if DEVICE == "cuda":
        # 8-bit keeps BLIP-2 opt-2.7b under ~6 GB on the 4060
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            load_kwargs["device_map"] = "auto"

    _caption_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_CAPTION_MODEL, **load_kwargs
    )
    _caption_model.eval()
    return _caption_model, _caption_processor


def caption(pil_image: Image.Image, max_new_tokens: int = 30) -> str:
    model, processor = _load_captioner()
    inputs = processor(images=pil_image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


# ── ITM (BLIP-2 vit-g, with BLIP-1 fallback) ─────────────────────────────────
_itm_model = None
_itm_processor = None
_itm_kind = None  # "blip2" or "blip1"


def _load_itm(prefer="blip2"):
    """Load BLIP-2 ITM (Salesforce/blip2-itm-vit-g) by default.
    Falls back to BLIP-1 large COCO on load error."""
    global _itm_model, _itm_processor, _itm_kind
    if _itm_model is not None:
        return _itm_model, _itm_processor, _itm_kind

    import torch

    if prefer == "blip2":
        try:
            from transformers import AutoProcessor, Blip2ForImageTextRetrieval
            _itm_processor = AutoProcessor.from_pretrained(BLIP2_ITM_MODEL)
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            _itm_model = Blip2ForImageTextRetrieval.from_pretrained(
                BLIP2_ITM_MODEL, torch_dtype=dtype
            ).to(DEVICE).eval()
            _itm_kind = "blip2"
            return _itm_model, _itm_processor, _itm_kind
        except Exception as e:
            print(f"[blip2] BLIP-2 ITM failed to load ({e}); falling back to BLIP-1.")
            _itm_model = None
            _itm_processor = None

    from transformers import BlipForImageTextRetrieval, BlipProcessor
    _itm_processor = BlipProcessor.from_pretrained(BLIP_ITM_MODEL)
    _itm_model = BlipForImageTextRetrieval.from_pretrained(BLIP_ITM_MODEL)
    if DEVICE == "cuda":
        _itm_model = _itm_model.half()
    _itm_model = _itm_model.to(DEVICE).eval()
    _itm_kind = "blip1"
    return _itm_model, _itm_processor, _itm_kind


def itm_scores(pil_image: Image.Image, captions_list, batch_size: int = 16):
    """Returns the ITM positive-match probability for each (image, caption) pair.
    Higher = better match. Length matches `captions_list`. Batched for speed."""
    import torch

    if not captions_list:
        return []
    model, processor, kind = _load_itm()

    scores = [0.0] * len(captions_list)
    # Group indices by non-empty caption so we don't waste compute on blanks
    valid_idx = [i for i, c in enumerate(captions_list) if c]
    if not valid_idx:
        return scores

    valid_caps = [captions_list[i] for i in valid_idx]

    for start in range(0, len(valid_caps), batch_size):
        chunk_idx = valid_idx[start:start + batch_size]
        chunk_caps = valid_caps[start:start + batch_size]
        # Repeat the image so we batch (N, 3, H, W) with N=len(chunk_caps).
        # BLIP-2's processor manages text length internally via num_query_tokens;
        # BLIP-1 happily accepts max_length=64. Branch on kind to avoid a tripwire.
        if kind == "blip2":
            inputs = processor(
                images=[pil_image] * len(chunk_caps),
                text=chunk_caps,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
        else:
            inputs = processor(
                images=[pil_image] * len(chunk_caps),
                text=chunk_caps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(DEVICE)
        if DEVICE == "cuda":
            inputs = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in inputs.items()}

        with torch.no_grad():
            if kind == "blip2":
                # Blip2ForImageTextRetrieval.forward(use_image_text_matching_head=True) -> ITMOutput
                out = model(**inputs, use_image_text_matching_head=True)
                logits = out.logits_per_image if hasattr(out, "logits_per_image") else out.logits
                # Per HF docs: itm output shape (B, 2). softmax dim=-1, take "match" col.
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            else:
                out = model(**inputs, use_itm_head=True)
                probs = torch.softmax(out.itm_score.float(), dim=-1)[:, 1]

        for k_local, idx_global in enumerate(chunk_idx):
            scores[idx_global] = float(probs[k_local].item())

    return scores
