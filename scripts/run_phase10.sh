#!/usr/bin/env bash
# Phase 10: multi-length ADMM with gradient-chosen insertion (beam projection, independent S' acceptance),
# 10 seeds x 2 tasks; offsets 0 -2 +2. Lowest priority: starts after Phase 9 (STATUS) and can be
# cancelled (tmux kill-session -t gepa_phase10) without affecting other phases. Re-scores on the 2k sets.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
until [ -f artifacts/phase9/STATUS ]; do sleep 120; done
$PY scripts/gpu_queue.py artifacts/phase10/jobs.jsonl --min-free-mb 25000 --max-per-gpu 2 --settle 90 --retries 5
for task in agnews emotion; do
  runs=$(for d in artifacts/phase10/${task}_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE10_COMPLETE $(date)" > artifacts/phase10/STATUS
