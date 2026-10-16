#!/usr/bin/env bash
# Phase 9: controls for the variable-length ADMM arms (beam projection, independent S' acceptance),
# 10 seeds x 2 tasks. multistart: 3 fixed-length ADMM runs per proposal from the same parent, pooled
# and chosen on S (same pool size and compute as multi-length, but no length change). beam_p36: the
# fair fixed-length beam ADMM proposer with 36 instead of 12 GEPA proposals (the same ~3x compute
# spent on more GEPA proposals instead). Starts after Phase 8 (STATUS); re-scores on the 2k sets.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
until [ -f artifacts/phase8/STATUS ]; do sleep 120; done
$PY scripts/gpu_queue.py artifacts/phase9/jobs.jsonl --min-free-mb 25000 --max-per-gpu 2 --settle 90 --retries 5
for task in agnews emotion; do
  runs=$(for d in artifacts/phase9/${task}_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE9_COMPLETE $(date)" > artifacts/phase9/STATUS
