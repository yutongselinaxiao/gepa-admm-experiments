#!/usr/bin/env bash
# Phase 14b: rerun the GoEmotions ADMM arms (C, G, F x 10 seeds) after the out-of-memory fix (gradient
# accumulation in multi-label ADMM). The reflection arms A, B from Phase 14 are unaffected and kept; the old
# ADMM run folders are in superseded_unchunked/. Smoke-gated; then all Phase 14 runs are re-scored into the
# comment-level report file.
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
D=artifacts/phase14_goemotions_f1
$PY scripts/gpu_queue.py $D/smoke2_jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
if ! $PY -c "import json,sys; r=json.load(open('$D/smoke2_jobs.jsonl.status.json'))['results']; sys.exit(0 if len(r)==3 and all(v=='done' for v in r.values()) else 1)"; then
  echo "SMOKE FAILED $(date)" > $D/STATUS; exit 1
fi
$PY scripts/gpu_queue.py $D/admm_jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
runs=$(for d in $D/goemotions_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task goemotions \
  --output artifacts/report2k/goemotions_f1.json --extra-runs $runs
echo "PHASE14_COMPLETE $(date)" > $D/STATUS
