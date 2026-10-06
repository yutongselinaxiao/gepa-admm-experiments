#!/usr/bin/env bash
# Phase 2b: seeds 3-9 for reflective and top-p ADMM alone (Phase 2 had seeds 0-2).
# All runs: selection split (GEPA sees 256 val examples; 256 reserved for reporting),
# 12 proposals, minibatch 8, 3 run seeds. GPUs 4-7 only; one sequential queue per GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase2
GPUS=(4 5 6 7)
COMMON="--proposals 12 --reflection-minibatch-size 8 --selection-split"
TOPP="--admm-projection context_top_p --admm-top-p 0.99 --admm-rho 0.1 --admm-outer-iterations 20"
mkdir -p "$OUT"

JOBS=()
for seed in 3 4 5 6 7 8 9; do
  JOBS+=("reflective_s$seed|--run-seed $seed")
  JOBS+=("admm_topp_s$seed|--run-seed $seed --proposer admm $TOPP --admm-acceptance heldout")
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
