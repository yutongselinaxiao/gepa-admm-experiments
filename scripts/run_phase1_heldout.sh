#!/usr/bin/env bash
# Phase 1b: ADMM proposer with held-out acceptance (instead of GEPA's minibatch rule).
# Same settings as run_phase1_main.sh. GPUs 4-7 only; jobs queue in waves of 4.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase1_heldout
GPUS=(4 5 6 7)
mkdir -p "$OUT"

JOBS=()
for rho in 0.3 1; do for seed in 0 1 2; do JOBS+=("$rho $seed"); done; done

for i in "${!JOBS[@]}"; do
  read -r rho seed <<<"${JOBS[$i]}"
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  name="admm_rho${rho}_s${seed}"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m gepa_pilot.run_gepa --proposals 12 --reflection-minibatch-size 8 \
    --run-seed "$seed" --proposer admm --admm-rho "$rho" --admm-outer-iterations 20 \
    --admm-acceptance heldout --run-dir "$OUT/$name" > "$OUT/$name.log" 2>&1 &
  if (( (i + 1) % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
grep -H "DONE\|Traceback" "$OUT"/*.log
