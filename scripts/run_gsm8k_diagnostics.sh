#!/usr/bin/env bash
# GSM8K surrogate diagnostics: rank stage sharded over GPUs 4-7 (30 instructions, 200
# training problems), then the gradient stage on GPU 4. Official test split never loaded.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
PY=/dataMeR2/yutong/envs/myenv/bin/python
OUT=artifacts/gsm8k_diagnostics
mkdir -p "$OUT"
GPUS=(4 5 6 7)
for slot in 0 1 2 3; do
  idx=$(seq $slot 4 29 | tr '\n' ' ')
  CUDA_VISIBLE_DEVICES=${GPUS[$slot]} $PY -m gepa_pilot.gsm8k_diagnostics --stage rank \
    --instruction-indices $idx --shard-name "s$slot" --output-dir "$OUT" > "$OUT/rank_s$slot.log" 2>&1 &
done
wait
$PY -c "from gepa_pilot.gsm8k_diagnostics import merge_rank; import json; print(json.dumps(merge_rank('$OUT'), indent=2))" \
  > "$OUT/rank_summary.log" 2>&1
CUDA_VISIBLE_DEVICES=4 $PY -m gepa_pilot.gsm8k_diagnostics --stage grad --output-dir "$OUT" > "$OUT/grad.log" 2>&1
grep -hE "Traceback|GRAD" "$OUT"/*.log
