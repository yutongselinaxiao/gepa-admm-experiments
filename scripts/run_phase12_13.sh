#!/usr/bin/env bash
# Phases 12-13, run as one batch after Phase 11 (Yahoo) finishes:
#   12: AG News answered with label names (multi-token answers; seed instruction ~80% vs ~30% with letters)
#   13: GoEmotions, multi-label via binary relevance (8 emotions; report metric = example / micro / macro F1)
# All eight arms, 10 seeds each: A reflective (12), B matched reflective (30, held-out judge), C fair beam ADMM,
# E fair reflect-then-polish, G gated, F multi-length with gradient insertion, I multi-start (3), H fixed beam
# with 36 proposals. Queue on GPUs 4-7 (GPU errors retried), then 2k re-scoring.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
until [ -f artifacts/phase11_yahoo/STATUS ]; do sleep 120; done
$PY scripts/gpu_queue.py artifacts/phase12_13/jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
for task in agnews_names goemotions; do
  runs=$(for d in artifacts/phase12_13/${task}_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task $task --extra-runs $runs
done
echo "PHASE12_13_COMPLETE $(date)" > artifacts/phase12_13/STATUS
