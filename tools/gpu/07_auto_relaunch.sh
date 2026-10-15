#!/usr/bin/env bash
# Start the dead-end watcher once, detached, so it survives independently of any
# workstation session. Single instance guarded by its own pidfile.
set -euo pipefail
ROOT=/root/loweye
PIDF=$ROOT/logs/auto_relaunch.pid
LOG=$ROOT/logs/auto_relaunch.log
mkdir -p "$ROOT/logs"
if [ -f "$PIDF" ]; then
  OLD=$(cat "$PIDF" 2>/dev/null || true)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "auto-relaunch watcher already running: pid=$OLD"
    exit 0
  fi
  echo "stale watcher pidfile: pid=${OLD:-empty}"
fi
cd "$ROOT"
nohup /root/miniconda3/bin/python -u "$ROOT/tools/gpu/auto_relaunch_when_idle.py" >> "$LOG" 2>&1 &
NEW=$!
echo "$NEW" > "$PIDF"
sleep 4
if ! kill -0 "$NEW" 2>/dev/null; then
  echo "watcher failed to stay up"
  tail -20 "$LOG"
  exit 1
fi
echo "watcher_pid=$NEW"
tail -3 "$LOG"
