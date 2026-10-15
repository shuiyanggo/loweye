#!/usr/bin/env bash
set -euo pipefail
cd /root/loweye
cmd=$(tr '\0' ' ' < /proc/3622/cmdline)
[[ "$cmd" == *tools/gpu/run_inspection_tasks.py* ]]
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/root/loweye/logs/fire_smoke_pipeline_status.json')
assert json.loads(p.read_text())['phase']=='waiting_for_verified_download'
PY
kill 3622
nohup /root/miniconda3/bin/python -u tools/gpu/run_inspection_tasks.py > logs/inspection_pipeline.log 2>&1 < /dev/null &
echo "pipeline_pid=$!"
