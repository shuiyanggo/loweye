#!/usr/bin/env bash
# Read-only: current pipeline state, smoke parity evidence, and training progress.
set -uo pipefail
L=/root/loweye/logs
B=/root/autodl-tmp/loweye
echo "=== pipeline status (fire_smoke) ==="
tail -3 "$L/fire_smoke_pipeline_status.json" 2>/dev/null || echo missing
echo "=== pipeline status (crop_pest) ==="
tail -3 "$L/crop_pest_pipeline_status.json" 2>/dev/null || echo missing
echo "=== smoke marker lines (crop_pest) ==="
grep -n -E 'onnx_verification|AssertionError|RuntimeError|Mismatched elements|Max absolute difference|Traceback|non-zero exit' "$L/smoke_crop_pest.log" 2>/dev/null | tail -20 || echo none
echo "=== smoke marker lines (fire_smoke) ==="
grep -n -E 'onnx_verification|AssertionError|RuntimeError|Mismatched elements|Max absolute difference|Traceback|non-zero exit' "$L/smoke_fire_smoke.log" 2>/dev/null | tail -20 || echo none
echo "=== epochs logged ==="
for t in fire_smoke crop_pest; do
  f="$B/$t/${t}_epochs.jsonl"
  if [ -f "$f" ]; then echo "$t lines=$(wc -l < "$f")"; tail -1 "$f"; else echo "$t no epochs file"; fi
done
echo "=== pipeline log tail ==="
tail -4 "$L/inspection_pipeline.log"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo "=== trainers running ==="
pgrep -af train_inspection_fcos.py || echo none
