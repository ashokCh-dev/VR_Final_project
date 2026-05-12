"""Generate report figures from artifacts/.

Run from the project root:
    python make_figures.py

Produces:
    figures/loss_curve.png      — per-seed loss vs epoch
    figures/ablation_bars.png   — R@10 (hit/full) and mAP@10 across ablation conditions
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── Loss curves ─────────────────────────────────────────────────────────────
def loss_curve_plot():
    hist_files = sorted(ARTIFACTS.glob("training_history_hn*.json"))
    if not hist_files:
        print("No training histories found.")
        return

    seed_label = {
        "training_history_hn.json": "seed 83",
        "training_history_hn_seed527.json": "seed 527",
        "training_history_hn_seed33.json": "seed 33",
        "training_history_hn_seed588.json": "seed 588",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Left panel: total loss per seed
    ax = axes[0]
    for f in hist_files:
        label = seed_label.get(f.name, f.stem)
        history = json.load(open(f))
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        ax.plot(epochs, losses, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss (InfoNCE + λ·Triplet)")
    ax.set_title("Total training loss per seed")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # Right panel: components for the first (reference) seed only
    ax = axes[1]
    f0 = next((f for f in hist_files if f.name == "training_history_hn.json"), hist_files[0])
    history = json.load(open(f0))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["infonce"] for h in history], marker="o", linewidth=2,
            color="#1f77b4", label="InfoNCE")
    ax.plot(epochs, [h["triplet"] for h in history], marker="s", linewidth=2,
            color="#d62728", label="Triplet (cosine, m=0.3)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component loss")
    ax.set_title(f"Loss components — {seed_label.get(f0.name, f0.stem)}")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = FIGURES / "loss_curve.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Ablation bars ───────────────────────────────────────────────────────────
def ablation_bars_plot():
    conditions = [
        ("eval_A_alpha1.0.json",     "A\nfrozen α=1"),
        ("eval_B_alpha0.5.json",     "B\nfrozen α=0.5"),
        ("eval_B_alpha0.7.json",     "B\nfrozen α=0.7"),
        ("eval_C_alpha0.5.json",     "C\nFT α=0.5"),
        ("eval_C_alpha0.7.json",     "C\nFT α=0.7"),
        ("eval_C_alpha0.5_hn.json",  "C-HN\nα=0.5"),
        ("eval_C_alpha0.7_hn.json",  "C-HN\nα=0.7"),
    ]

    labels, r10_hit, r10_full, map10, r10_hit_err, r10_full_err, map10_err = [], [], [], [], [], [], []
    for fname, label in conditions:
        p = ARTIFACTS / fname
        if not p.exists():
            print(f"  skip (missing): {fname}")
            continue
        data = json.load(open(p))
        results = data["results"]
        boot = data.get("bootstrap", {})
        labels.append(label)
        r10_hit.append(results["Recall@10"])
        r10_full.append(results["Recall_full@10"])
        map10.append(results["mAP@10"])
        r10_hit_err.append(boot.get("Recall@10", {}).get("std", 0))
        r10_full_err.append(boot.get("Recall_full@10", {}).get("std", 0))
        map10_err.append(boot.get("mAP@10", {}).get("std", 0))

    if not labels:
        print("No ablation evals found.")
        return

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - width, r10_hit, width, yerr=r10_hit_err, capsize=3,
                label="Recall@10 (hit)", color="#1f77b4")
    b2 = ax.bar(x, r10_full, width, yerr=r10_full_err, capsize=3,
                label="Recall@10 (full)", color="#ff7f0e")
    b3 = ax.bar(x + width, map10, width, yerr=map10_err, capsize=3,
                label="mAP@10", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Metric value")
    ax.set_title("Ablation: Recall@10 (hit / full) and mAP@10 by condition\n"
                 "(mean ± std over 4 query-bootstrap seeds)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    # Value labels on top of bars
    for bars in (b1, b2, b3):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = FIGURES / "ablation_bars.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    loss_curve_plot()
    ablation_bars_plot()
