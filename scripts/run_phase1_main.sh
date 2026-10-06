#!/usr/bin/env bash
# Phase 1 main comparison: GEPA reflective vs. GEPA with the ADMM proposer.
# 12 proposals, reflection minibatch 8, 3 run seeds per arm, one job per GPU.
# Reflective seed 0 is the pilot run artifacts/gepa_confirmation_mb8.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/phase1
mkdir -p "$OUT"
COMMON="--proposals 12 --reflection-minibatch-size 8"

launch() {  # gpu name args...
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY -m gepa_pilot.run_gepa $COMMON --run-dir "$OUT/$name" "$@" \
    > "$OUT/$name.log" 2>&1 &
}

launch 0 reflective_s1 --run-seed 1
launch 1 reflective_s2 --run-seed 2
launch 2 admm_rho0.3_s0 --run-seed 0 --proposer admm --admm-rho 0.3 --admm-outer-iterations 20
launch 3 admm_rho0.3_s1 --run-seed 1 --proposer admm --admm-rho 0.3 --admm-outer-iterations 20
launch 4 admm_rho0.3_s2 --run-seed 2 --proposer admm --admm-rho 0.3 --admm-outer-iterations 20
launch 5 admm_rho1_s0 --run-seed 0 --proposer admm --admm-rho 1 --admm-outer-iterations 20
launch 6 admm_rho1_s1 --run-seed 1 --proposer admm --admm-rho 1 --admm-outer-iterations 20
launch 7 admm_rho1_s2 --run-seed 2 --proposer admm --admm-rho 1 --admm-outer-iterations 20
wait
grep -H "DONE\|Traceback" "$OUT"/*.log
