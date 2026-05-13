# Visual Product Search Engine
*Image-to-image retrieval on DeepFashion In-Shop with CLIP fine-tuning and BLIP-2 ITM re-ranking*

**Author:** Ashok &nbsp;|&nbsp; **Course:** Visual Recognition Project

---

## 1. Problem and motivation

Online-shoppers routinely spot a product in a photo but cannot describe it in words that match how the catalog has tagged it. Text search fails because of two mismatches:

- **Language mismatch.** Users type colloquial terms ("black hoodie"); catalogs use technical or marketing copy ("100% cotton tonal Sherpa-lined zip-up").
- **Inconsistent metadata.** Different sellers describe identical items differently, so keyword recall is unreliable.

A query-by-image system removes both barriers. The user uploads an image and the system returns visually + semantically similar products — no text required.

We use the **DeepFashion In-Shop Clothes Retrieval** dataset. Images with the same `item_id` are treated as the same product. The dataset comes pre-split into `train` (25,882 images), `gallery` (12,612), and `query` (14,218).

## 2. System architecture

```
─── Offline indexing (per gallery image) ───────────────────────────────────────
   bbox crop  →  CLIP image encoder
                 BLIP-2 captioner  →  CLIP text encoder
                                ⤷  fused v = α·φV(x̂) + (1−α)·φT(c),  ‖v‖=1
                 HNSW index (cosine)  ←──────────────────────────────────────

─── Online query ──────────────────────────────────────────────────────────────
   user image  →  YOLO crop  →  CLIP image encoder  →  HNSW search (top-K')
                                                          │
                                                          └─ BLIP-2 ITM
                                                             re-rank (optional)
                                                          ↓
                                                       top-K results
```

The two encoders share a CLIP ViT-L/14 backbone (`openai` pretrained). YOLO (v8-L, weights from a prior project) is used **only at query time** — the offline gallery uses the dataset's ground-truth bboxes (approach 1 of the project clarification). HNSW (`hnswlib`, `cosine`, M=32, ef_construction=200, ef=100) backs the ANN search. BLIP-2 (`Salesforce/blip2-itm-vit-g`) provides the image-text matching head for re-ranking.

Captions for the gallery are generated once by BLIP-2 (`Salesforce/blip2-opt-2.7b`) — short product descriptors like *"black floral print mini dress"* (38,494 captions covering train + gallery). Query images do not have pre-computed captions; this is by design — at inference time the system encodes the query with the visual branch only and searches against fused gallery vectors. The fusion coefficient α therefore controls how much text information enters the **gallery representation**.

## 3. Fine-tuning strategy

We fine-tune CLIP **only**; YOLO and BLIP-2 stay frozen.

| Choice | Value |
| --- | --- |
| Backbone | ViT-L/14 (`openai`) |
| Unfrozen | last 4 vision transformer blocks + `ln_post` + visual projection (51.2 M / 427.6 M = 12.0 % of params) |
| Text encoder | frozen |
| Optimizer | AdamW, lr = 1 × 10⁻⁵, weight decay = 0.01 |
| Schedule | 1-epoch linear warmup → cosine annealing |
| Epochs | 10 |
| Effective batch | 32 (micro-batch 8 × grad-accum 4) |
| AMP | `torch.amp` fp16 |
| Loss | $\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(z_a, z_p) + \lambda \cdot \mathcal{L}_{\text{tri}}(z_a, z_p, z_n)$, λ=0.5, τ=0.07, margin=0.3 (defined below) |
| Hard negatives | top-10 mined per anchor from a frozen-CLIP HNSW pass, pool refreshed every 3 epochs with the latest weights |
| Image augmentation | OpenCLIP default preprocessing (center crop 224 × 224, normalize) |

The **hard-negative pool** is the main lever vs. plain InfoNCE: we mine 10 hard negatives per training image from a 50-NN HNSW query (skipping same-`item_id` matches), then re-mine every 3 epochs using the *current* (already partially fine-tuned) encoder so the negatives stay informative.

For a mini-batch of *B* anchor-positive pairs $\{(z_a^{(i)}, z_p^{(i)})\}_{i=1}^B$ with all features L2-normalised, the symmetric InfoNCE term over the union $Z = \{z_a, z_p\}$ (size $2B$) is:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2B} \sum_{i=1}^{2B} \log \frac{\exp(z_i \cdot z_{\text{pos}(i)} / \tau)}{\sum_{j \neq i} \exp(z_i \cdot z_j / \tau)}$$

where $\text{pos}(i)$ is the in-batch positive (anchor↔positive pair). The triplet term with cosine-distance margin is:

$$\mathcal{L}_{\text{tri}} = \frac{1}{B} \sum_{i=1}^{B} \max\big(0,\ (1 - z_a^{(i)} \cdot z_p^{(i)}) - (1 - z_a^{(i)} \cdot z_n^{(i)}) + m\big), \quad m = 0.3$$

where $z_n^{(i)}$ is a hard negative sampled from the mined pool for anchor $i$. Both losses operate on the unit hypersphere ($\|z\| = 1$), so cosine similarity reduces to a dot product.

Training was performed on a single RTX 4060 Laptop (8 GB VRAM) — the small micro-batch + grad-accum is the memory adaptation versus the original Kaggle reference (batch=32). One run completes in ~50 minutes; loss falls **1.18 → 0.24** across 10 epochs (final InfoNCE 0.099, Triplet 0.273).

![Training loss curves — total loss across two independent training seeds (left) and per-component loss for seed 83 (right). The triplet loss plateaus near the margin (m=0.3) while InfoNCE drives the remaining decrease; the two seeds track to within ≈0.005 every epoch.](figures/loss_curve.png)

## 4. Evaluation protocol

- **Splits:** As provided. Train = 25,882, Gallery = 12,612, Query = 14,218. Ground truth: two images match iff they share `item_id`.
- **Metrics:** Recall@K, NDCG@K, mAP@K at **K ∈ {5, 10, 15}**.
- **Implementation:** Single source of truth at `src/metrics.py`.
- **Random component for mean ± std:**
  - For **C-HN α = 0.7 (headline)**: **four** full fine-tuning runs with seeds 83, 527, 33 (locally on the 4060) and 588 (on Kaggle). Different torch RNG perturbs DataLoader shuffle, hard-neg sampling order, and weight init via the LR schedule. Reported as straight mean ± std over the four point estimates.
  - For **C-HN α = 0.5**: three full fine-tuning runs (seeds 83, 527, 33 — seed 588 was only run for α=0.7 on Kaggle).
  - For **non-trained** conditions (A, B, C-vanilla): query-set bootstrap — resample 80% of the query set with replacement, four seeds [83, 588, 527, 33], report mean ± std of the per-bootstrap means.

This is per the project clarification: *"choose any random component, justify in viva."* The two random components in play are (a) training stochasticity (for C-HN, where it applies) and (b) query-distribution variance via bootstrap (for frozen and vanilla-FT conditions, where there is no training stochasticity to vary).

### 4.1 Metric definitions

For each query *q* with ground-truth `item_id` and retrieved list `R = (r_1, …, r_K)` ordered by descending cosine similarity, let

$$\text{hit}(r_i, q) = \begin{cases} 1 & \text{if } \text{item\_id}(r_i) = \text{item\_id}(q) \\ 0 & \text{otherwise} \end{cases}$$

Let *n_rel(q)* be the number of gallery images sharing *q*'s item_id (multiple images can match a single query, since DeepFashion stores ~2–5 views per item).

**Recall@K (hit-rate)** — what the project statement asks for: *"fraction of queries for which at least one relevant item is retrieved in the top-K results."* For each query, the per-query score is binary.

$$\text{Recall@K}_{\text{hit}}(q) = \mathbb{1}\Big[\,\textstyle\sum_{i=1}^{K} \text{hit}(r_i, q) \geq 1\Big] \quad ; \quad \text{Recall@K}_{\text{hit}} = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall@K}_{\text{hit}}(q)$$

**Recall@K (full / textbook)** — the IR-textbook definition (Manning et al.): fraction of *all* relevant items captured in the top-K. We report this as a secondary column for cross-paper comparability.

$$\text{Recall@K}_{\text{full}}(q) = \frac{\sum_{i=1}^{K} \text{hit}(r_i, q)}{n_{rel}(q)} \quad ; \quad \text{Recall@K}_{\text{full}} = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall@K}_{\text{full}}(q)$$

By construction $\text{Recall@K}_{\text{full}} \leq \text{Recall@K}_{\text{hit}}$. The two coincide only when every query has exactly one relevant gallery image. Most DeepFashion items have 2–5 gallery views, so the gap reflects how many of the views beyond the first one we manage to surface in the top-K.

**NDCG@K** — binary relevance, log-base-2 discount, normalised by the ideal DCG of `min(n_rel, K)` relevant items at top ranks:

$$\text{DCG@K}(q) = \sum_{i=1}^{K} \frac{\text{hit}(r_i, q)}{\log_2(i+1)} \qquad \text{IDCG@K}(q) = \sum_{i=1}^{\min(n_{rel}(q), K)} \frac{1}{\log_2(i+1)}$$

$$\text{NDCG@K}(q) = \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)} \quad ; \quad \text{NDCG@K} = \frac{1}{|Q|} \sum_q \text{NDCG@K}(q)$$

**mAP@K** — mean of per-query Average Precision at K. *AP@K* uses cumulative precision at each hit, normalised by `min(n_rel, K)`:

$$\text{AP@K}(q) = \frac{1}{\min(n_{rel}(q), K)} \sum_{i=1}^{K} \text{hit}(r_i, q) \cdot \frac{\textstyle\sum_{j=1}^{i} \text{hit}(r_j, q)}{i} \quad ; \quad \text{mAP@K} = \frac{1}{|Q|} \sum_q \text{AP@K}(q)$$

The `min(n_rel, k)` denominator is the *only* way an upper bound of 1 is guaranteed. The original notebook used `k` here, which inflated mAP whenever `n_rel < k` (e.g. mAP@10 = 1.67 for condition C). All numbers in this report use the corrected denominator.

## 5. Ablation conditions

Three settings isolate the contribution of each component, plus one re-ranking ablation:

| ID | α | Encoder | Captions | Notes |
| --- | --- | --- | --- | --- |
| **A**         | 1.0 | frozen CLIP | — | vision-only baseline |
| **B**         | 0.7 / 0.5 | frozen CLIP | BLIP-2 | measures gain from captions, no fine-tune |
| **C**         | 0.7 / 0.5 | InfoNCE FT | BLIP-2 | vanilla fine-tune |
| **C-HN**      | 0.7 / 0.5 | InfoNCE + Triplet + HN-mining FT | BLIP-2 | **headline model** |
| **C-HN + ITM**| 0.7 | same as C-HN | BLIP-2 | online BLIP-2 ITM re-rank, blend(ITM=0.2, ANN=0.8) |

## 6. Results

Headline numbers reported as **mean ± std**. All evaluations use the full 14,218-query set.

![Ablation summary across the 7 conditions. Bars show R@10 (hit-rate, full-Recall) and mAP@10 with bootstrap-std error bars (mostly imperceptible). The dominant lift comes from fine-tuning (A/B → C); hard-neg mining and α=0.7 over α=0.5 each add ~1 pp on top.](figures/ablation_bars.png)

### Full ablation table

All numbers are query-bootstrap mean ± std over 4 seeds (resample 80% with replacement, seeds 83/588/527/33). **Recall@K (hit)** is the project-statement variant; **Recall@K (full)** is the textbook Manning-et-al variant.

| Condition | R@10 (hit) | R@10 (full) | NDCG@10 | mAP@10 | R@15 (hit) | R@15 (full) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A   (frozen, α=1.0) †      | 0.569 ± 0.009 | 0.238 ± 0.003 | 0.248 ± 0.004 | 0.175 ± 0.003 | 0.603 ± 0.008 | 0.262 ± 0.003 |
| B   (frozen, α=0.7) †      | 0.591 ± 0.010 | 0.259 ± 0.004 | 0.264 ± 0.004 | 0.189 ± 0.003 | 0.631 ± 0.008 | 0.287 ± 0.004 |
| B   (frozen, α=0.5) †      | 0.583 ± 0.005 | 0.258 ± 0.002 | 0.253 ± 0.003 | 0.180 ± 0.003 | 0.624 ± 0.006 | 0.288 ± 0.003 |
| C   (InfoNCE FT, α=0.7) †  | 0.879 ± 0.003 | 0.561 ± 0.002 | 0.570 ± 0.002 | 0.471 ± 0.002 | 0.901 ± 0.002 | 0.607 ± 0.002 |
| C   (InfoNCE FT, α=0.5) †  | 0.866 ± 0.004 | 0.540 ± 0.003 | 0.544 ± 0.003 | 0.445 ± 0.003 | 0.892 ± 0.003 | 0.587 ± 0.002 |
| C-HN (α=0.5) ‡             | 0.870 ± 0.002 | 0.549 ± 0.002 | 0.553 ± 0.001 | 0.454 ± 0.001 | 0.893 ± 0.002 | 0.596 ± 0.001 |
| **C-HN (α=0.7) — headline ‡** | **0.881 ± 0.003** | **0.568 ± 0.001** | **0.578 ± 0.003** | **0.480 ± 0.003** | **0.903 ± 0.002** | **0.613 ± 0.001** |
| C-HN (α=0.7) + BLIP-2 ITM blend(0.2) ◇ | 0.881 | 0.564 | 0.551 | 0.449 | 0.907 | 0.617 |

> † = 4-seed query bootstrap (single trained model, query subsamples). ‡ = mean ± std over **independent training runs**: 4 for α=0.7 (seeds 83/527/33/588), 3 for α=0.5 (seeds 83/527/33). ◇ = single point estimate on the seed-83 model (full-set Kaggle run, no bootstrap or multi-seed for ITM).

### 6.1.5 α sweep (C-HN, seed 83 model)

Following the α=0.7 vs α=0.5 contrast, we extended the sweep up the α range using the seed-83 C-HN model to see where the optimum lies. All four α values use the **same trained checkpoint**; only the gallery-side fusion ratio differs. Bootstrap mean ± std over 4 query-subsamples.

| α | R@10 (hit) | R@10 (full) | NDCG@10 | mAP@10 |
| --- | ---: | ---: | ---: | ---: |
| 0.5  | 0.872 ± 0.004 | 0.549 ± 0.003 | 0.554 ± 0.003 | 0.454 ± 0.003 |
| 0.7  | 0.881 ± 0.004 | 0.567 ± 0.003 | 0.578 ± 0.004 | 0.479 ± 0.004 |
| 0.85 | **0.884 ± 0.004** | **0.568 ± 0.003** | **0.581 ± 0.004** | **0.482 ± 0.003** |
| 0.9  | **0.884 ± 0.004** | **0.569 ± 0.003** | **0.581 ± 0.004** | **0.483 ± 0.004** |

The curve rises from 0.5 → 0.85 then **plateaus**. The 0.85/0.9 gap above 0.7 is tiny (~0.3 pp on R@10, ~0.4 pp on mAP@10) and falls inside the 4-training-seed std (R@10 std = 0.0034 from §6.1), so we cannot call α=0.85 statistically distinct from α=0.7 — but the monotone-then-plateau shape is consistent and matches the noisy-caption hypothesis: more weight on the visual channel helps until the text contribution becomes negligible.

We kept the headline at α=0.7 because (a) it was the value we trained multiple seeds for, giving the strongest variance evidence; (b) the gain from going higher is within noise; and (c) shifting the headline post-hoc on a single-seed sweep would over-fit to seed 83.

> **Seed stability of the headline.** Across four independent fine-tuning runs (different seeds for weight init, DataLoader shuffle, and hard-neg sampling), R@10 spans **0.8785–0.8862** and mAP@10 spans **0.4772–0.4836**. The 4-seed std is well under 1 pp for every metric, confirming the result is not a lucky-seed artifact. Individual per-seed numbers (for transparency):
>
> | Seed | R@10 | NDCG@10 | mAP@10 | Where |
> |---|---:|---:|---:|---|
> | 83  | 0.8796 | 0.5775 | 0.4793 | local (RTX 4060) |
> | 527 | 0.8815 | 0.5783 | 0.4800 | local |
> | 33  | 0.8785 | 0.5755 | 0.4772 | local |
> | 588 | 0.8862 | 0.5823 | 0.4836 | Kaggle T4 |

### 6.1 Notes on the metric implementation

The original research notebook reported impossible mAP values (e.g. mAP@10 = 1.67) in condition C. The cause was two coupled bugs in the AP computation: the per-query denominator was `k` instead of `min(n_relevant, k)`, and the top-K candidate list was over-fetched. We re-implemented metrics from scratch (`src/metrics.py`) and verified all reported mAP values now lie in [0, 1]. The fixed implementation is the single source of truth across every condition in this report.

### 6.2 Key findings

**1. Fine-tuning is the dominant lever.** Going from A (R@10 = 0.569 hit / 0.238 full) to C (vanilla, R@10 = 0.879 hit / 0.561 full) gives **+31 pp Recall@10 (hit)** and **+30 pp mAP@10**. Captions on a frozen encoder (A → B) give only **+2 pp R@10**.

**2. Hard-negative mining is consistent but small.** C → C-HN gives a marginal **+0.2 pp R@10 (hit)** (0.879 → 0.881) and **+0.8 pp mAP@10** (0.471 → 0.479) at α = 0.7. At α = 0.5 the pattern repeats (+0.5 pp R@10, +0.9 pp mAP@10). Hard-neg mining is not the main win in this task; the bulk of the improvement comes from contrastive fine-tuning itself.

**3. Higher α is better up to ~0.85, then plateaus.** A four-point α sweep on the seed-83 C-HN model (§6.1.5) shows R@10 rising 0.872 → 0.881 → 0.884 → 0.884 as α moves 0.5 → 0.7 → 0.85 → 0.9, then flattening. Captions help (the curve is above the α=1.0 frozen baseline) but the visual channel deserves most of the weight — consistent with BLIP-2 producing short product descriptors that pack many catalog images into a small text vocabulary, so heavy text weight introduces noise.

**4. The hit-rate vs full-Recall gap reveals an in-list ranking problem.** Hit-rate R@10 (0.881) and full R@10 (0.567) differ by a factor of ~1.6× for our headline model. This means we routinely find at least one correct match in top-10 (88% of queries) but recover only ~57% of *all* matches when items have multiple gallery views. The frozen baseline has a ~2.4× ratio (0.569 vs 0.238), so fine-tuning narrows but does not close this gap — most of our remaining mAP headroom is here.

**5. BLIP-2 ITM re-ranking is a *negative* result.** Implementation uses `Salesforce/blip2-itm-vit-g` (BLIP-2's image-text retrieval checkpoint with the actual ITM head); BLIP-1 (`blip-itm-large-coco`) is wired in as a fallback for OOM/load-failure cases but is never used in the reported numbers. Pure-ITM reordering of the top-50 candidates was catastrophic on a 200-query probe (R@10 0.78, NDCG@10 0.42) — the short product captions don't differentiate among visually similar candidates, so the ranking becomes near-random. A low-weight blend (`combined = 0.8·ANN + 0.2·ITM`) recovers to ≈baseline on the full 14k set: R@10 essentially unchanged (0.881 vs 0.881), but **NDCG@10 still drops 3 pp (0.578 → 0.551)** and **mAP@10 drops 3 pp (0.479 → 0.449)**. ITM doesn't add useful signal here. We document the implementation but do not include it in the headline model.

**6. Variance is tight on both axes.** Query-bootstrap std across 4 subsamples is ~0.3–0.5 pp on every metric. **Training-seed std across 4 independent fine-tuning runs** for the headline (C-HN α=0.7) is also under 1 pp on every metric (R@10 std = 0.0034, mAP@10 std = 0.0026). Both far below the inter-condition gaps, so the ablation ordering is statistically meaningful on both axes of variance.

## 7. Streamlit demo

`app.py` provides the interactive demo required by the deliverable. Flow:

1. User uploads an image.
2. **YOLO** (yolov8-l) crops the dominant garment; the user sees the crop next to the original.
3. **Confirm / Re-crop** buttons gate the rest of the pipeline (re-crop loosens YOLO confidence and re-runs).
4. On confirm: the cropped image is encoded by the **fine-tuned CLIP** (hard-neg checkpoint), HNSW returns top-K candidates, and (optional sidebar checkbox) BLIP-2 ITM re-ranks them.
5. The top-K results are shown as a grid with similarity score and `item_id`.

Sidebar controls expose the condition (A / B / C / C-HN), α, K, and the rerank toggle, so the demo doubles as a live A/B/C ablation.

## 8. Limitations and what we'd do next

- **Two seeds, not three.** The third fine-tuning seed (527) hit the Kaggle 12 h kernel cap. With a 24 h kernel or two parallel sessions we'd close this out.
- **No query-side captioning.** We deliberately don't caption queries at inference (faster, doesn't assume internet access). A pre-cached BLIP-2 caption + text-text similarity might beat ITM since the search would be in caption-space rather than mixed image-text-space. This is the obvious next thing to try given the ITM finding.
- **Gallery uses ground-truth bbox, queries use YOLO.** Per the project clarification this is approved, but a unified YOLO pipeline (using the fine-tuned detector from `notebooks/yolo-final-proj.ipynb` for both sides) would eliminate the train/serve distribution mismatch at some compute cost.
- **CLIP ViT-L/14 is heavy on 8 GB VRAM.** A ViT-B/32 backbone would train faster and be deployable on smaller machines at a 2–4 pp metric cost — worth profiling for production.
- **Captions are short.** Going from product-name captions (BLIP-2 default) to attribute-rich captions ("black, floral, A-line, mini, summer") would likely give a bigger lift than further fine-tuning. We can fine-tune BLIP-2 on DeepFashion's `list_description_inshop.json` to produce such captions — but per the project clarification we cannot use that file directly.

## 9. Conclusions

A simple recipe — fine-tune the last 4 blocks of CLIP ViT-L/14 with InfoNCE + Triplet on hard negatives, fuse with BLIP-2 captions at α = 0.7, search HNSW — gives **Recall@10 (hit) = 0.881 ± 0.003**, **Recall@10 (full) = 0.568 ± 0.001**, **NDCG@10 = 0.578 ± 0.003** and **mAP@10 = 0.480 ± 0.003** on DeepFashion In-Shop — a **+31 pp Recall@10 (hit) / +30 pp mAP@10** lift over the frozen-CLIP baseline. The headline is the **mean ± std over four independent fine-tuning runs** with seeds 83, 527, 33 (local RTX 4060) and 588 (Kaggle T4), so the variance reflects real training stochasticity, not just query-set resampling.

The most actionable finding is the *negative* one: the problem statement's BLIP-2 ITM re-rank step does not help in this short-caption regime and consistently degrades ranking metrics by ~3 pp. We recommend dropping it from a production pipeline and instead investing the same compute in better captions or a larger ANN candidate pool.

## Appendix A — Reproducibility

All numbers in this report come from `artifacts/eval_*.json` and from the Kaggle kernel output. The `--bootstrap 4` flag of `demo_batch_eval.py` produces both point estimates and bootstrap mean ± std with seeds [83, 588, 527, 33] and an 80 % resample fraction. The CLIP hard-neg checkpoint is `clip_finetuned_hn.pt`; commit hashes for the code: see `git log` on the [GitHub repo](https://github.com/ashokCh-dev/VR_Final_project).
