#!/usr/bin/env bash
# Phase 2c: ADMM with beam-search Z-step (context_beam, B=4), fluency weight 0.3 and 1.0.
# All runs: selection split (GEPA sees 256 val examples; 256 reserved for reporting),
# 12 proposals, minibatch 8, 3 run seeds. GPUs 4-7 only; one sequential queue per GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase2
GPUS=(4 5 6 7)
COMMON="--proposals 12 --reflection-minibatch-size 8 --selection-split"
BEAM="--proposer admm --admm-projection context_beam --admm-top-p 0.99 --admm-rho 0.1 --admm-outer-iterations 20 --admm-beam-width 4 --admm-acceptance heldout"
mkdir -p "$OUT"

JOBS=()
for seed in 0 1 2; do for lam in 0.3 1.0; do
  JOBS+=("beam_lam${lam}_s$seed|--run-seed $seed $BEAM --admm-fluency-weight $lam")
done; done

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
