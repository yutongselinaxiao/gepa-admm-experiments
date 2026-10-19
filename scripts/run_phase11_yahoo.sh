#!/usr/bin/env bash
# Phase 11: a third classification task, Yahoo Answers topics (question titles, 10 classes; seed
# instruction 19.5% on validation). All eight arms used on AG News and emotion, 10 seeds each:
# A reflective (12), B matched reflective (30, held-out judge), C fair beam ADMM, E fair reflect-then-
# polish, G gated, F multi-length with gradient insertion, I multi-start (3), H fixed beam with 36
# proposals. Same protocol as Phases 6-10. Queue on GPUs 4-7 (GPU errors retried), then 2k re-scoring.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
$PY scripts/gpu_queue.py artifacts/phase11_yahoo/jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
runs=$(for d in artifacts/phase11_yahoo/yahoo_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task yahoo --extra-runs $runs
echo "PHASE11_COMPLETE $(date)" > artifacts/phase11_yahoo/STATUS
