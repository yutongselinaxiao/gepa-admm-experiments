#!/usr/bin/env bash
# Phase 7: fair reflect-then-polish (beam polish, final child judged vs GEPA parent on S') and
# multi-length beam ADMM proposer (offsets 0, -2, +2; S' acceptance). 10 seeds x 2 tasks.
# Runs through the memory-aware queue on shared GPUs 4-7, then re-scores the new runs on the
# 2,000-example report sets (after the Stage 0 re-scoring has released artifacts/report2k/*.json).
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
$PY scripts/gpu_queue.py artifacts/phase7/jobs.jsonl --min-free-mb 18000 --max-per-gpu 2 --settle 90
while tmux has-session -t gepa_stage0 2>/dev/null; do sleep 60; done
for task in agnews emotion; do
  runs=$(ls -d artifacts/phase7/${task}_*_s* 2>/dev/null | grep -v '\.log$' | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE7_COMPLETE $(date)" > artifacts/phase7/STATUS
