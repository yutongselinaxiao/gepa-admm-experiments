#!/usr/bin/env bash
# Phase 7 follow-up: reruns failed jobs outside artifacts/phase7/ (so the main pipeline's 2k
# re-scoring never sees a half-finished run), then, once both the main batch (STATUS) and the
# rerun queue are finished, moves successful reruns into artifacts/phase7/ and re-scores on the
# 2,000-example sets (cached prompts are not re-scored). Writes artifacts/phase7/STATUS_FINAL.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
$PY scripts/gpu_queue.py artifacts/phase7_failed/rerun.jsonl --min-free-mb 18000 --max-per-gpu 1 --settle 30
until [ -f artifacts/phase7/STATUS ]; do sleep 60; done
moved=""
for run in artifacts/phase7_rerun/*/; do
  run=${run%/}; name=$(basename "$run")
  if grep -q "\"$name\": \"done\"" artifacts/phase7_failed/rerun.jsonl.status.json && [ -f "$run/gepa_result.json" ]; then
    mv "$run" artifacts/phase7/ && mv "artifacts/phase7_rerun/$name.log" artifacts/phase7/ 2>/dev/null
    moved="$moved $name"
  fi
done
for task in agnews emotion; do
  runs=$(ls -d artifacts/phase7/${task}_*_s* 2>/dev/null | grep -v '\.log$' | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE7_FINAL $(date); reruns moved in:${moved:- none}" > artifacts/phase7/STATUS_FINAL
