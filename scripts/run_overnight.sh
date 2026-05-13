#!/bin/bash
# Overnight pipeline:
#   1. Full-set BLIP-2 ITM rerank eval on C_alpha0.7_hn  (~5h)
#   2. Hard-neg training with seed 588                    (~1h)
#   3. Rebuild C-indices for seed 588                     (~5min)
#   4. Eval seed 588                                       (~2min)
#   5. Hard-neg training with seed 527                    (~1h)
#   6. Rebuild + eval seed 527
#
# Each step logs to its own file in artifacts/ and continues only if the previous
# one succeeded. Total budget ~7-8 hours.

set -uo pipefail
cd /home/ashok_ubun/studies_ubun/VR_final_proj

PY=/home/ashok_ubun/anaconda3/bin/python
ART=artifacts

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$ART/overnight.log"; }

log "==== Stage 1/3 — BLIP-2 ITM rerank, full query set ===="
$PY -u demo_batch_eval.py \
  --condition C_alpha0.7_hn \
  --rerank --rerank_mode blend --itm_weight 0.2 \
  --out "$ART/eval_C_hn_itm_blend02_full.json" \
  > "$ART/overnight_stage1_itm.log" 2>&1
STAGE1_RC=$?
log "Stage 1 rc=$STAGE1_RC"

log "==== Stage 2a/3 — Train hard-neg seed 588 ===="
$PY -u train_hn.py --seed 588 --suffix _seed588 \
  > "$ART/overnight_stage2a_train588.log" 2>&1
STAGE2A_RC=$?
log "Stage 2a rc=$STAGE2A_RC"

if [ $STAGE2A_RC -eq 0 ]; then
  log "==== Stage 2b/3 — Rebuild + eval seed 588 ===="
  $PY -u rebuild_indices_hn.py --suffix _seed588 \
    > "$ART/overnight_stage2b_rebuild588.log" 2>&1
  $PY -u demo_batch_eval.py \
    --condition C_alpha0.7_hn_seed588 \
    --bin_path  "$ART/gallery_index_C_alpha07_hn_seed588.bin" \
    --meta_path "$ART/gallery_meta_C_alpha07_hn_seed588.json" \
    --checkpoint "$ART/clip_finetuned_hn_seed588.pt" \
    --alpha 0.7 --model_type ft_hn \
    --out "$ART/eval_C_hn_seed588.json" \
    >> "$ART/overnight_stage2b_rebuild588.log" 2>&1
  log "Stage 2b done"
fi

log "==== Stage 3a/3 — Train hard-neg seed 527 ===="
$PY -u train_hn.py --seed 527 --suffix _seed527 \
  > "$ART/overnight_stage3a_train527.log" 2>&1
STAGE3A_RC=$?
log "Stage 3a rc=$STAGE3A_RC"

if [ $STAGE3A_RC -eq 0 ]; then
  log "==== Stage 3b/3 — Rebuild + eval seed 527 ===="
  $PY -u rebuild_indices_hn.py --suffix _seed527 \
    > "$ART/overnight_stage3b_rebuild527.log" 2>&1
  $PY -u demo_batch_eval.py \
    --condition C_alpha0.7_hn_seed527 \
    --bin_path  "$ART/gallery_index_C_alpha07_hn_seed527.bin" \
    --meta_path "$ART/gallery_meta_C_alpha07_hn_seed527.json" \
    --checkpoint "$ART/clip_finetuned_hn_seed527.pt" \
    --alpha 0.7 --model_type ft_hn \
    --out "$ART/eval_C_hn_seed527.json" \
    >> "$ART/overnight_stage3b_rebuild527.log" 2>&1
  log "Stage 3b done"
fi

log "==== Overnight pipeline complete ===="
log "Result files:"
ls -la "$ART"/eval_C_hn*.json 2>&1 | tee -a "$ART/overnight.log"
