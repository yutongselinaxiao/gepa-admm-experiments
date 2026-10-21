#!/usr/bin/env bash
# Phase 15: GoEmotions (comment-level F1) controls and polish, 10 seeds each: E reflect-then-polish,
# I multi-start (3 ADMM runs per proposal), H fixed-length ADMM with 36 proposals. I and H match F's ~3x
# compute, so F vs. I/H tests whether F's micro/macro-F1 gain is more than extra compute. Runs go into the
# Phase 14 folder; smoke-gated; then all GoEmotions runs are re-scored (cached prompts are reused).
set -u
cd "$(dirname "$0")/.."
PY=/dataMeR2/yutong/envs/myenv/bin/python
D=artifacts/phase14_goemotions_f1
$PY scripts/gpu_queue.py $D/smoke3_jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
if ! $PY -c "import json,sys; r=json.load(open('$D/smoke3_jobs.jsonl.status.json'))['results']; sys.exit(0 if len(r)==3 and all(v=='done' for v in r.values()) else 1)"; then
  echo "SMOKE FAILED $(date)" > $D/STATUS_15; exit 1
fi
$PY scripts/gpu_queue.py $D/controls_jobs.jsonl --min-free-mb 20000 --max-per-gpu 3 --settle 60 --retries 5
runs=$(for d in $D/goemotions_*_s*/; do [ -f "$d/gepa_result.json" ] && echo "${d%/}"; done | tr '\n' ' ')
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.report_set rescore --task goemotions \
  --output artifacts/report2k/goemotions_f1.json --extra-runs $runs
echo "PHASE15_COMPLETE $(date)" > $D/STATUS_15
