"""
Hard-negative CLIP fine-tuning, localized for RTX 4060 (8 GB VRAM).
Mirrors clip-hard-negative-mining.ipynb cells 4, 6, 8, 10, 12, 14.

Memory tuning vs. the original notebook:
  - FT_BATCH_SIZE 32 -> 8, GRAD_ACCUM_STEPS 4 (effective batch still 32)
  - HN_EMB_BATCH 128 -> 64 (mining pass)
  - Modern torch.amp API instead of deprecated torch.cuda.amp.*

Usage:
    python train_hn.py                            # default seed 83 -> clip_finetuned_hn.pt
    python train_hn.py --seed 588 --suffix _seed588   # -> clip_finetuned_hn_seed588.pt
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import hnswlib
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths (auto-detect Kaggle vs local via src.config) ──────────────────────
from src.config import (
    ARTIFACTS_DIR as OUTPUT_DIR,
    BBOX_FILE,
    BBOX_PAD as PAD,
    CAPTIONS_FILE,
    CLIP_MODEL,
    CLIP_PRETRAIN,
    DEVICE,
    EMB_DIM,
    IMG_ROOT,
    SPLIT_FILE,
)

SEEDS         = [83, 588, 527, 33]
ALPHA_VALUES  = [0.7, 0.5]

# Hard-negative mining
HN_POOL_SIZE  = 10
HN_EMB_BATCH  = 64
FETCH_K       = HN_POOL_SIZE * 5 + 1

# Fine-tuning hyperparameters (4060-tuned)
FT_EPOCHS          = 10
FT_BATCH_SIZE      = 8
GRAD_ACCUM_STEPS   = 4
FT_LR              = 1e-5
WEIGHT_DECAY       = 0.01
TEMPERATURE        = 0.07
TRIPLET_MARGIN     = 0.3
LAMBDA_TRIPLET     = 0.5
LAST_N_BLOCKS      = 4
WARMUP_EPOCHS      = 1
REFRESH_POOL_EVERY = 3
FT_SEED            = SEEDS[0]  # overridden by --seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=FT_SEED,
                   help="Random seed for torch / numpy / random / DataLoader / hard-neg sampling")
    p.add_argument("--suffix", default="",
                   help="Filename suffix for outputs (e.g. '_seed588'). "
                        "Empty -> overwrites clip_finetuned_hn.pt.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parse_split_file(path):
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


def parse_bbox_file(path):
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


def bbox_crop(img_path, bbox, pad=PAD):
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    x1, y1, x2, y2 = bbox
    px = int((x2 - x1) * pad); py = int((y2 - y1) * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)
    return pil.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else pil


@torch.no_grad()
def embed_rows(rows, model, preprocess, bbox_map, batch_size=HN_EMB_BATCH):
    model.eval()
    all_embs, all_ids, all_names = [], [], []
    for i in tqdm(range(0, len(rows), batch_size), desc="Embedding", leave=False):
        batch = rows[i:i + batch_size]
        crops, ids, names = [], [], []
        for r in batch:
            p = IMG_ROOT / r["image_name"]
            if not p.exists():
                continue
            try:
                bbox = bbox_map.get(r["image_name"])
                crop = bbox_crop(p, bbox) if bbox else Image.open(p).convert("RGB")
                crops.append(preprocess(crop))
                ids.append(r["item_id"])
                names.append(r["image_name"])
            except Exception:
                continue
        if not crops:
            continue
        t = torch.stack(crops).to(DEVICE)
        emb = F.normalize(model.encode_image(t).float(), dim=-1)
        all_embs.append(emb.cpu().numpy())
        all_ids.extend(ids); all_names.extend(names)
    return np.vstack(all_embs).astype(np.float32), all_ids, all_names


def build_hn_pool(embs, ids, names):
    id_arr = np.array(ids)
    idx = hnswlib.Index(space="cosine", dim=EMB_DIM)
    idx.init_index(max_elements=len(embs), ef_construction=200, M=32)
    idx.add_items(embs, list(range(len(embs))))
    idx.set_ef(100)
    labels, _ = idx.knn_query(embs, k=FETCH_K)
    pool = {}
    for i, name in enumerate(names):
        hard_negs = []
        for pos in labels[i]:
            if pos == i or id_arr[pos] == ids[i]:
                continue
            hard_negs.append(names[pos])
            if len(hard_negs) == HN_POOL_SIZE:
                break
        pool[name] = hard_negs
    return pool


class HardNegTripletDataset(Dataset):
    def __init__(self, rows, hard_neg_pool, transform, bbox_map):
        groups = defaultdict(list)
        for r in rows:
            groups[r["item_id"]].append(r["image_name"])
        self.items = [
            (iid, imgs) for iid, imgs in groups.items()
            if len(imgs) >= 2 and any(hard_neg_pool.get(n) for n in imgs)
        ]
        self.hard_neg_pool = hard_neg_pool
        self.transform = transform
        self.bbox_map = bbox_map

    def __len__(self):
        return len(self.items)

    def _load(self, name):
        p = IMG_ROOT / name
        bbox = self.bbox_map.get(name)
        crop = bbox_crop(p, bbox) if bbox else Image.open(p).convert("RGB")
        return self.transform(crop)

    def __getitem__(self, idx):
        _, imgs = self.items[idx]
        anc_name, pos_name = random.sample(imgs, 2)
        hn_pool = self.hard_neg_pool.get(anc_name) or self.hard_neg_pool.get(pos_name) or []
        try:
            anchor = self._load(anc_name)
            positive = self._load(pos_name)
            hard_neg = self._load(random.choice(hn_pool)) if hn_pool else anchor
        except Exception:
            anchor = positive = hard_neg = self._load(anc_name)
        return anchor, positive, hard_neg


def infonce_loss(z1, z2, temp=TEMPERATURE):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temp
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def main():
    args = parse_args()
    seed = args.seed
    suffix = args.suffix
    ckpt_name = f"clip_finetuned_hn{suffix}.pt"
    hist_name = f"training_history_hn{suffix}.json"
    print(f"Device: {DEVICE}  Seed: {seed}  Suffix: '{suffix}'")
    print(f"Will save -> {ckpt_name}, {hist_name}")
    if DEVICE == "cpu":
        print("WARNING: CUDA not available. Training on CPU will be unusably slow. "
              "Reboot into discrete-GPU mode and re-run.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    all_rows = parse_split_file(SPLIT_FILE)
    bbox_map = parse_bbox_file(BBOX_FILE)
    train_rows = [r for r in all_rows if r["split"] == "train"]
    print(f"Train: {len(train_rows)}  Bboxes: {len(bbox_map)}")

    # Frozen CLIP for initial mining pass
    print(f"\nLoading frozen CLIP {CLIP_MODEL} ({CLIP_PRETRAIN})...")
    clip_frozen, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN
    )
    clip_frozen = clip_frozen.to(DEVICE).eval()

    print("\nStep 1: Embedding train images with frozen CLIP...")
    train_embs, train_ids, train_names = embed_rows(
        train_rows, clip_frozen, clip_preprocess, bbox_map
    )
    print(f"  Embedded {len(train_names)} train images")

    print("\nStep 2: Mining initial hard negative pool...")
    hard_neg_pool = build_hn_pool(train_embs, train_ids, train_names)
    sizes = [len(v) for v in hard_neg_pool.values()]
    print(f"  Avg pool size: {np.mean(sizes):.1f}  Min: {min(sizes)}  Max: {max(sizes)}")

    # Free the frozen model before training to save VRAM
    del clip_frozen, train_embs
    torch.cuda.empty_cache()

    set_seed(seed)

    print(f"\nStep 3: Loading fresh CLIP for fine-tuning...")
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAIN)
    clip_model = clip_model.to(DEVICE)

    for p in clip_model.parameters():
        p.requires_grad = False
    for block in list(clip_model.visual.transformer.resblocks)[-LAST_N_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in clip_model.visual.ln_post.parameters():
        p.requires_grad = True
    if hasattr(clip_model.visual, "proj") and clip_model.visual.proj is not None:
        clip_model.visual.proj.requires_grad = True

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in clip_model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    hn_dataset = HardNegTripletDataset(train_rows, hard_neg_pool, clip_preprocess, bbox_map)
    print(f"  Triplet dataset: {len(hn_dataset)} unique items")

    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a, b: 1.0 - F.cosine_similarity(a, b),
        margin=TRIPLET_MARGIN, reduction="mean",
    )

    ft_loader = DataLoader(
        hn_dataset, batch_size=FT_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, clip_model.parameters()),
        lr=FT_LR, weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = len(ft_loader) // GRAD_ACCUM_STEPS
    total_steps = FT_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    def refresh_pool(model):
        print("  Refreshing hard negative pool...")
        embs, ids, names = embed_rows(train_rows, model, clip_preprocess, bbox_map)
        new_pool = build_hn_pool(embs, ids, names)
        hn_dataset.hard_neg_pool = new_pool
        del embs
        torch.cuda.empty_cache()

    clip_model.train()
    best_loss = float("inf")
    history = []

    print(f"\nStep 4: Training {FT_EPOCHS} epochs "
          f"(batch={FT_BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}, effective={FT_BATCH_SIZE*GRAD_ACCUM_STEPS})")

    for epoch in range(FT_EPOCHS):
        if epoch > 0 and epoch % REFRESH_POOL_EVERY == 0:
            refresh_pool(clip_model)
            clip_model.train()

        tot_loss = tot_nce = tot_tri = 0.0
        n_optim_steps = 0
        optimizer.zero_grad()

        for step, (anchor, positive, hard_neg) in enumerate(
            tqdm(ft_loader, desc=f"Epoch {epoch+1}/{FT_EPOCHS}", leave=False)
        ):
            anchor = anchor.to(DEVICE, non_blocking=True)
            positive = positive.to(DEVICE, non_blocking=True)
            hard_neg = hard_neg.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda"):
                z_a = F.normalize(clip_model.encode_image(anchor).float(), dim=-1)
                z_p = F.normalize(clip_model.encode_image(positive).float(), dim=-1)
                z_n = F.normalize(clip_model.encode_image(hard_neg).float(), dim=-1)
                loss_nce = infonce_loss(z_a, z_p)
                loss_tri = triplet_loss_fn(z_a, z_p, z_n)
                loss = (loss_nce + LAMBDA_TRIPLET * loss_tri) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, clip_model.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                n_optim_steps += 1

            tot_loss += loss.item() * GRAD_ACCUM_STEPS
            tot_nce += loss_nce.item()
            tot_tri += loss_tri.item()

        n = len(ft_loader)
        avg = tot_loss / n; avg_nce = tot_nce / n; avg_tri = tot_tri / n
        lr = scheduler.get_last_lr()[0]
        history.append({"epoch": epoch + 1, "loss": avg, "infonce": avg_nce,
                        "triplet": avg_tri, "lr": lr})
        print(f"Epoch {epoch+1:2d}/{FT_EPOCHS} | Total: {avg:.4f} | "
              f"InfoNCE: {avg_nce:.4f} | Triplet: {avg_tri:.4f} | LR: {lr:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save(clip_model.state_dict(), OUTPUT_DIR / ckpt_name)
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

        json.dump(history, open(OUTPUT_DIR / hist_name, "w"), indent=2)

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Saved → {OUTPUT_DIR / ckpt_name}")


if __name__ == "__main__":
    main()
