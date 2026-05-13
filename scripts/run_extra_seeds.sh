#!/bin/bash
# Train C_alpha0.7_hn for seeds 527 and 33, rebuild indices, eval with --bootstrap 4.
# Sequential; total ~2.5 h on RTX 4060.

set -uo pipefail
cd /home/ashok_ubun/studies_ubun/VR_final_proj

PY=/home/ashok_ubun/anaconda3/bin/python
ART=artifacts

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$ART/extra_seeds.log"; }

for SEED in 527 33; do
  log "==== Seed $SEED — training ===="
  $PY -u train_hn.py --seed $SEED --suffix _seed${SEED} \
    > "$ART/extra_seed${SEED}_train.log" 2>&1

  log "==== Seed $SEED — rebuilding C indices ===="
  $PY -u rebuild_indices_hn.py --suffix _seed${SEED} \
    > "$ART/extra_seed${SEED}_rebuild.log" 2>&1

  log "==== Seed $SEED — eval (alpha=0.7) ===="
  $PY -u demo_batch_eval.py \
    --condition C_alpha0.7_hn_seed${SEED} \
    --bin_path  "$ART/gallery_index_C_alpha07_hn_seed${SEED}.bin" \
    --meta_path "$ART/gallery_meta_C_alpha07_hn_seed${SEED}.json" \
    --checkpoint "$ART/clip_finetuned_hn_seed${SEED}.pt" \
    --alpha 0.7 --model_type ft_hn --bootstrap 4 \
    --out "$ART/eval_C_alpha0.7_hn_seed${SEED}.json" \
    > "$ART/extra_seed${SEED}_eval07.log" 2>&1

  log "==== Seed $SEED — eval (alpha=0.5) ===="
  $PY -u demo_batch_eval.py \
    --condition C_alpha0.5_hn_seed${SEED} \
    --bin_path  "$ART/gallery_index_C_alpha05_hn_seed${SEED}.bin" \
    --meta_path "$ART/gallery_meta_C_alpha05_hn_seed${SEED}.json" \
    --checkpoint "$ART/clip_finetuned_hn_seed${SEED}.pt" \
    --alpha 0.5 --model_type ft_hn --bootstrap 4 \
    --out "$ART/eval_C_alpha0.5_hn_seed${SEED}.json" \
    > "$ART/extra_seed${SEED}_eval05.log" 2>&1
done

log "==== Done ===="
ls -la "$ART"/eval_C_alpha*_hn_seed*.json 2>&1 | tee -a "$ART/extra_seeds.log"
