#!/usr/bin/env bash
# Phase 8: full GEPA runs of gated variable-length ADMM (beam projection, independent S' acceptance),
# 10 seeds x 2 tasks. (Multi-length with gradient-chosen insertion moved to Phase 10, after the
# Phase 9 controls.) Waits until Phase 7 is fully finished (STATUS_FINAL) so the two
# batches do not compete for memory on the shared GPUs, then runs through the queue (GPU errors,
# including ones GEPA swallows, are retried) and re-scores all Phase 8 runs on the 2k report sets.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
until [ -f artifacts/phase7/STATUS_FINAL ]; do sleep 120; done
$PY scripts/gpu_queue.py artifacts/phase8/jobs.jsonl --min-free-mb 25000 --max-per-gpu 2 --settle 90 --retries 5
for task in agnews emotion; do
  runs=$(for d in artifacts/phase8/${task}_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE8_COMPLETE $(date)" > artifacts/phase8/STATUS
