"""
Rebuild HNSW gallery indices for the C-conditions using a hard-neg checkpoint.

Usage:
  python rebuild_indices_hn.py                                  # uses clip_finetuned_hn.pt -> *_hn.bin/.json
  python rebuild_indices_hn.py --suffix _seed588               # uses clip_finetuned_hn_seed588.pt
  python rebuild_indices_hn.py --checkpoint path/to/x.pt --suffix _x   # explicit checkpoint
"""
import argparse
import time
from pathlib import Path

from src import config
from src.clip_encoder import ClipEncoder, embed_rows
from src.data_utils import load_captions, parse_bbox_file, parse_split_file
from src.retrieve import build_hnsw, save_hnsw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path. Defaults to artifacts/clip_finetuned_hn{suffix}.pt.")
    p.add_argument("--suffix", default="",
                   help="Filename suffix for both checkpoint (input) and indices (output). "
                        "Outputs always include '_hn' too: gallery_index_C_alpha07_hn{suffix}.bin")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {config.DEVICE}")
    ckpt = Path(args.checkpoint) if args.checkpoint else config.ARTIFACTS_DIR / f"clip_finetuned_hn{args.suffix}.pt"
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}. Train first via train_hn.py.")

    C_CONDITIONS = [
        (f"C_alpha0.7_hn{args.suffix}", 0.7,
         f"gallery_index_C_alpha07_hn{args.suffix}.bin",
         f"gallery_meta_C_alpha07_hn{args.suffix}.json"),
        (f"C_alpha0.5_hn{args.suffix}", 0.5,
         f"gallery_index_C_alpha05_hn{args.suffix}.bin",
         f"gallery_meta_C_alpha05_hn{args.suffix}.json"),
    ]

    print(f"Loading hard-neg CLIP: {ckpt}")
    encoder = ClipEncoder(checkpoint_path=ckpt)

    all_rows = parse_split_file()
    gallery_rows = [r for r in all_rows if r["split"] == "gallery"]
    bbox_map = parse_bbox_file()
    captions = load_captions()
    print(f"Gallery rows: {len(gallery_rows)}")

    for key, alpha, bin_name, meta_name in C_CONDITIONS:
        print(f"\n=== {key} (alpha={alpha}) ===")
        t0 = time.time()
        embs, ids, names = embed_rows(
            gallery_rows, captions, encoder, bbox_map, config.IMG_ROOT,
            alpha=alpha, batch_size=64, desc=f"Embedding gallery ({key})",
        )
        print(f"  Embedded: {embs.shape} in {time.time()-t0:.0f}s")

        idx = build_hnsw(embs)
        bin_path = config.ARTIFACTS_DIR / bin_name
        meta_path = config.ARTIFACTS_DIR / meta_name
        save_hnsw(idx, ids, names, bin_path, meta_path)
        print(f"  Saved -> {bin_path.name}, {meta_path.name}")

    print("\nDone. Update src/config.INDEX_FILES to point C_* at the _hn files when ready.")


if __name__ == "__main__":
    main()
