#!/usr/bin/env bash
# Phase 4: generality check on dair-ai/emotion (6 classes) with the AG News protocol.
#   reflective_p12            reflective GEPA, 12 proposals, minibatch acceptance
#   reflective_heldout_p30    strongest matched baseline (30 proposals, held-out acceptance)
#   beam_lam0.3               beam-search ADMM (rho 0.1, top-p 0.99, B=4, fluency 0.3), held-out acceptance
# Selection split, minibatch 8, seeds 0-9. GPUs 4-7 only; one sequential queue per GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase4_emotion
GPUS=(4 5 6 7)
COMMON="--config configs/emotion.yaml --train artifacts/data_emotion/train.json \
  --validation artifacts/data_emotion/validation.json --reflection-minibatch-size 8 --selection-split"
BEAM="--proposals 12 --proposer admm --admm-projection context_beam --admm-top-p 0.99 --admm-rho 0.1 \
  --admm-outer-iterations 20 --admm-beam-width 4 --admm-fluency-weight 0.3 --admm-acceptance heldout"
mkdir -p "$OUT"

JOBS=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
  JOBS+=("beam_lam0.3_s$seed|--run-seed $seed $BEAM")
  JOBS+=("reflective_p12_s$seed|--run-seed $seed --proposals 12")
  JOBS+=("reflective_heldout_p30_s$seed|--run-seed $seed --proposals 30 --reflection-acceptance heldout")
done

run_queue() {  # gpu job...
  local gpu=$1; shift
  for job in "$@"; do
    local name=${job%%|*} args=${job#*|}
    CUDA_VISIBLE_DEVICES=$gpu $PY -m gepa_pilot.run_gepa $COMMON $args --run-dir "$OUT/$name" \
      > "$OUT/$name.log" 2>&1
  done
}

for slot in "${!GPUS[@]}"; do
  queue=()
  for i in "${!JOBS[@]}"; do (( i % ${#GPUS[@]} == slot )) && queue+=("${JOBS[$i]}"); done
  run_queue "${GPUS[$slot]}" "${queue[@]}" &
done
wait
grep -H "DONE\|Traceback" "$OUT"/*.log
