#!/usr/bin/env bash
# Phase 14: GoEmotions rerun with the corrected multi-label design. One example = one comment, scored on all
# 8 emotions; GEPA's score, S/S' selection and acceptance = example F1, ADMM's loss = 1 - soft F1, and the
# optimization data has the natural positive rate (Phase 13 optimized balanced yes/no pairs, which pushed
# prompts toward "yes"). Core arms A, B, C, G, F x 10 seeds. Re-scored into a NEW report file, because the
# old one caches prompt scores for the old pair-based report set.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
# Smoke first (one proposal, 2 ADMM rounds per proposer): the main batch starts only if all four succeed.
$PY scripts/gpu_queue.py artifacts/phase14_goemotions_f1/smoke_jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
if ! $PY -c "import json,sys; r=json.load(open('artifacts/phase14_goemotions_f1/smoke_jobs.jsonl.status.json'))['results']; sys.exit(0 if len(r)==4 and all(v=='done' for v in r.values()) else 1)"; then
  echo "SMOKE FAILED $(date)" > artifacts/phase14_goemotions_f1/STATUS; exit 1
fi
$PY scripts/gpu_queue.py artifacts/phase14_goemotions_f1/jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
runs=$(for d in artifacts/phase14_goemotions_f1/goemotions_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task goemotions \
  --output artifacts/report2k/goemotions_f1.json --extra-runs $runs
echo "PHASE14_COMPLETE $(date)" > artifacts/phase14_goemotions_f1/STATUS
