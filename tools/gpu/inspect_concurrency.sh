#!/usr/bin/env bash
# Safety check: exactly one coordinator, exactly one trainer, one watcher.
# Two coordinators would mean two trainers writing the same checkpoint files, so this
# is worth confirming after installing the auto-relaunch watcher.
set -u
echo "--- relevant processes ---"
ps -eo pid,ppid,etime,pcpu,rss,args | grep -E 'run_inspection_tasks|train_inspection_fcos|auto_relaunch|06_launch_pipeline' | grep -v grep
echo "--- counts ---"
echo "coordinators=$(pgrep -fc 'run_inspection_tasks.py' || true)"
echo "trainers=$(pgrep -fc 'train_inspection_fcos.py' || true)"
echo "watchers=$(pgrep -fc 'auto_relaunch_when_idle.py' || true)"
echo "--- progress ---"
for t in crop_pest fire_smoke; do
  f=/root/autodl-tmp/loweye/$t/${t}_epochs.jsonl
  if [ -f "$f" ]; then
    echo "$t epochs=$(wc -l < "$f") last=$(date -r "$f" '+%H:%M:%S')"
  else
    echo "$t epochs=none"
  fi
done
echo "--- pipeline status ---"
for t in fire_smoke crop_pest; do
  echo "$t: $(cat /root/loweye/logs/${t}_pipeline_status.json 2>/dev/null || echo missing)"
done
echo "--- watcher log ---"
tail -3 /root/loweye/logs/auto_relaunch.log 2>/dev/null || echo "no watcher log"
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
