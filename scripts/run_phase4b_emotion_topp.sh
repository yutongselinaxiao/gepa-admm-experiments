#!/usr/bin/env bash
# Phase 4b: top-p ADMM proposer on emotion (fills the gap left by Phase 4; AG News analogue: phase2/admm_topp).
#   admm_topp   ADMM proposer, context top-p projection (rho 0.1, top-p 0.99), held-out acceptance
# Selection split, minibatch 8, seeds 0-9. GPUs 4-7 only; one sequential queue per GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase4_emotion
GPUS=(4 5 6 7)
COMMON="--config configs/emotion.yaml --train artifacts/data_emotion/train.json \
  --validation artifacts/data_emotion/validation.json --reflection-minibatch-size 8 --selection-split"
TOPP="--proposals 12 --proposer admm --admm-projection context_top_p --admm-top-p 0.99 --admm-rho 0.1 \
  --admm-outer-iterations 20 --admm-acceptance heldout"
mkdir -p "$OUT"

JOBS=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
  JOBS+=("admm_topp_s$seed|--run-seed $seed $TOPP")
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
