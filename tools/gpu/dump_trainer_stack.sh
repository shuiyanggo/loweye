#!/usr/bin/env bash
# Decisive check: is the trainer healthy inside evaluate(), or wedged?
# Reading the source proved the inner training loop prints nothing until the
# epoch row (which is flushed AND fsynced), so "0 epochs" plus "no GPU" is
# consistent with being inside evaluate(). py-spy turns that inference into a
# fact by showing the live Python stack.
set -u
PID=$(pgrep -f 'train_inspection_fcos.py --manifest' | head -1)
echo "=== clock ==="
date; date +%s
echo "=== artifact mtimes ==="
stat -c '%y %10s %n' /root/autodl-tmp/loweye/crop_pest/torch_cache/hub/checkpoints/*.pth 2>&1
stat -c '%y %10s %n' /root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl \
                    /root/autodl-tmp/loweye/crop_pest/crop_pest_run.json 2>&1
echo "=== pipeline log tail ==="
tail -c 500 /root/loweye/logs/inspection_pipeline.log
echo "=== py-spy install if missing ==="
if ! command -v py-spy >/dev/null 2>&1 && [ ! -x /root/miniconda3/bin/py-spy ]; then
  /root/miniconda3/bin/pip install --quiet py-spy 2>&1 | tail -3
fi
PYSPY=$(command -v py-spy || echo /root/miniconda3/bin/py-spy)
ls -l "$PYSPY" 2>&1
echo "=== py-spy dump (top 45 lines) ==="
timeout 90 "$PYSPY" dump --pid "$PID" 2>&1 | head -45
echo "=== epochs / gpu / cpu now ==="
wc -l /root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl 2>&1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps -o pid,etime,time,pcpu,stat -p "$PID"
