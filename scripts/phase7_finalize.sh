#!/usr/bin/env bash
# Phase 7 follow-up. failure_watcher.py follows the running batch and reruns any job that failed
# with a GPU error (OOM, CUDA/cuDNN) in artifacts/phase7_rerun/, outside artifacts/phase7/, so the
# main pipeline's 2k re-scoring never sees a half-finished run. Once the watcher and the main
# batch (STATUS) are both finished, successful reruns are moved into artifacts/phase7/ and all
# Phase 7 runs are re-scored on the 2,000-example sets (cached prompts are not re-scored).
# Writes artifacts/phase7/STATUS_FINAL.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
$PY scripts/failure_watcher.py artifacts/phase7/jobs.jsonl --rerun-dir artifacts/phase7_rerun \
  --failed-dir artifacts/phase7_failed --min-free-mb 25000 --max-per-gpu 1 --retries 5 --retry-wait 300
until [ -f artifacts/phase7/STATUS ]; do sleep 60; done
moved=""
for name in $($PY -c "import json; r=json.load(open('artifacts/phase7_rerun/watcher.status.json'))['results']; print(' '.join(n for n, o in r.items() if o == 'done'))"); do
  if [ -f "artifacts/phase7_rerun/$name/gepa_result.json" ]; then
    mv "artifacts/phase7_rerun/$name" "artifacts/phase7_rerun/$name.log" artifacts/phase7/
    moved="$moved $name"
  fi
done
for task in agnews emotion; do
  runs=$(ls -d artifacts/phase7/${task}_*_s* 2>/dev/null | grep -v '\.log$' | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE7_FINAL $(date); reruns moved in:${moved:- none}; watcher: $(cat artifacts/phase7_rerun/watcher.status.json | tr -d '\n ')" > artifacts/phase7/STATUS_FINAL
