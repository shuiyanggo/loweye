#!/usr/bin/env bash
# Idempotent launcher for the two-task inspection pipeline. Detached, logged,
# single-instance guarded by a pidfile. Never touches CPLID artifacts.
set -euo pipefail
ROOT=/root/loweye
LOG=$ROOT/logs/inspection_pipeline.log
PIDF=$ROOT/logs/inspection_pipeline.pid
PY=/root/miniconda3/bin/python
mkdir -p "$ROOT/logs"
if [ -f "$PIDF" ]; then
  OLD=$(cat "$PIDF" 2>/dev/null || true)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "already running: pid=$OLD"
    exit 0
  fi
  echo "stale pidfile: pid=${OLD:-empty}"
fi
cd "$ROOT"
nohup "$PY" -u "$ROOT/tools/gpu/run_inspection_tasks.py" >> "$LOG" 2>&1 &
NEW=$!
echo "$NEW" > "$PIDF"
sleep 5
if ! kill -0 "$NEW" 2>/dev/null; then
  echo "launcher failed: process exited early"
  tail -20 "$LOG"
  exit 1
fi
echo "pipeline_pid=$NEW"
echo "--- tail $LOG ---"
tail -6 "$LOG"
