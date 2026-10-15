#!/usr/bin/env bash
set -euo pipefail
cd /root/loweye
for spec in '2847 tools/fetch_dfire.py' '2947 tools/fetch_ip102.py'; do
  read -r pid expected <<< "$spec"
  if [ -r "/proc/$pid/cmdline" ]; then
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$cmd" != *"$expected"* ]]; then echo "PID mismatch $pid $cmd"; exit 1; fi
    kill "$pid"
    echo "Stopped verified old download $pid"
  fi
done
nohup /root/miniconda3/bin/python -u tools/fetch_dfire.py > logs/fetch_dfire.log 2>&1 < /dev/null &
echo "dfire_download_pid=$!"
nohup /root/miniconda3/bin/python -u tools/fetch_ip102.py > logs/fetch_ip102.log 2>&1 < /dev/null &
echo "ip102_download_pid=$!"
nohup /root/miniconda3/bin/python -u tools/gpu/run_inspection_tasks.py > logs/inspection_pipeline.log 2>&1 < /dev/null &
echo "pipeline_pid=$!"
