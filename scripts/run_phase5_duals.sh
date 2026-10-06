#!/usr/bin/env bash
# Phase 5: beam-search ADMM with per-class reward constraints (dual step 1.0, dual batch 64),
# otherwise identical to beam_lam0.3 (Phase 2 for AG News, Phase 4 for emotion).
# Selection split, 12 proposals, minibatch 8, seeds 0-9. GPUs 4-7 only.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
GPUS=(4 5 6 7)
BEAM="--proposals 12 --reflection-minibatch-size 8 --selection-split --proposer admm \
  --admm-projection context_beam --admm-top-p 0.99 --admm-rho 0.1 --admm-outer-iterations 20 \
  --admm-beam-width 4 --admm-fluency-weight 0.3 --admm-acceptance heldout \
  --admm-dual-step 1.0 --admm-dual-batch-size 64"
EMOTION="--config configs/emotion.yaml --train artifacts/data_emotion/train.json \
  --validation artifacts/data_emotion/validation.json"
mkdir -p artifacts/phase5_duals

JOBS=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
  JOBS+=("agnews_beam_dual_s$seed|--run-seed $seed")
  JOBS+=("emotion_beam_dual_s$seed|--run-seed $seed $EMOTION")
done

run_queue() {  # gpu job...
  local gpu=$1; shift
  for job in "$@"; do
    local name=${job%%|*} args=${job#*|}
    CUDA_VISIBLE_DEVICES=$gpu $PY -m gepa_pilot.run_gepa $BEAM $args \
      --run-dir "artifacts/phase5_duals/$name" > "artifacts/phase5_duals/$name.log" 2>&1
  done
}

for slot in "${!GPUS[@]}"; do
  queue=()
  for i in "${!JOBS[@]}"; do (( i % ${#GPUS[@]} == slot )) && queue+=("${JOBS[$i]}"); done
  run_queue "${GPUS[$slot]}" "${queue[@]}" &
done
wait
grep -H "DONE\|Traceback" artifacts/phase5_duals/*.log
