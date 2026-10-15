#!/usr/bin/env bash
# Read-only look at whether the crop_pest trainer is progressing or wedged.
# The earlier snapshot showed 1 process and 0% GPU, which is the expected shape
# of validation (dataloader workers exit at the end of a training epoch) but is
# also the shape of a hang, so measure log growth instead of guessing.
set -u
LOG=/root/loweye/logs/train_crop_pest.log
OUT=/root/autodl-tmp/loweye/crop_pest

echo "=== date ==="
date
echo "=== pipeline status ==="
for task in crop_pest fire_smoke; do
  echo "--- $task ---"
  tail -c 300 /root/loweye/logs/${task}_pipeline_status.json 2>/dev/null
  echo
done
echo "=== log growth over 25s ==="
stat -c 'size=%s mtime=%y' "$LOG" 2>/dev/null
sleep 25
stat -c 'size=%s mtime=%y' "$LOG" 2>/dev/null
echo "=== train log tail (2500 bytes, CR-normalised) ==="
tail -c 2500 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -12
echo "=== out dir ==="
ls -l "$OUT" 2>&1
echo "=== epochs jsonl ==="
wc -l "$OUT"/crop_pest_epochs.jsonl 2>&1
echo "=== processes ==="
ps -eo pid,etime,time,pcpu,rss,stat,args | grep -E 'train_inspection_fcos|run_inspection_tasks' | grep -v grep
echo "=== gpu (3 samples, 2s apart) ==="
for _ in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 2; done
