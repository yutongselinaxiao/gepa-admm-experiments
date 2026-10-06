#!/usr/bin/env bash
# Phase 3: compute- and acceptance-matched reflective baselines for beam-search ADMM.
#   A reflective_p30           30 proposals (~ADMM wall time), GEPA minibatch acceptance
#   B reflective_heldout_p12   12 proposals, held-out acceptance (same test as ADMM)
#   C reflective_heldout_p30   30 proposals, held-out acceptance
# Selection split, minibatch 8, seeds 0-9. GPUs 4-7 only; one sequential queue per GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase3
GPUS=(4 5 6 7)
COMMON="--reflection-minibatch-size 8 --selection-split"
mkdir -p "$OUT"

JOBS=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
  JOBS+=("reflective_p30_s$seed|--run-seed $seed --proposals 30")
  JOBS+=("reflective_heldout_p12_s$seed|--run-seed $seed --proposals 12 --reflection-acceptance heldout")
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
