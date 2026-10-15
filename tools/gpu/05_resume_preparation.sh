#!/usr/bin/env bash
# Resume the failed inspection coordinator only; never stop download workers.
set -euo pipefail
cd /root/loweye
mkdir -p logs
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

if Path('/proc/3988').exists():
    raise SystemExit('Refusing resume: old coordinator PID 3988 still exists')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        arguments = (proc / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        continue  # Process exited between enumeration and inspection.
    except PermissionError as exc:
        raise SystemExit(f'Cannot inspect process {proc.name}: {exc}')
    if any(b'train_inspection_fcos.py' in argument or b'run_inspection_tasks.py' in argument
           for argument in arguments):
        raise SystemExit(f'Refusing resume: inspection trainer/coordinator active at PID {proc.name}')

data = Path('/root/loweye/data/external/ip102')
receipt = json.loads((data / 'source_license.json').read_text())
transport = json.loads((data / 'ip102_detection_v1.transport.json').read_text())
if not receipt.get('sha256') or not receipt.get('archive_bytes'):
    raise SystemExit('Missing verified IP102 download receipt')
if transport.get('md5') != '85869d5c3c686e936c0dde780b208b01':
    raise SystemExit('Pinned IP102 source archive MD5 mismatch')
if transport.get('sha256') != receipt['sha256']:
    raise SystemExit('IP102 source receipt/transport SHA256 mismatch')
print('Resume preflight passed; download workers and CPLID artifacts untouched', flush=True)
PY
nohup /root/miniconda3/bin/python -u tools/gpu/run_inspection_tasks.py >> logs/inspection_pipeline.log 2>&1 < /dev/null &
pipeline_pid=$!
printf '%s\n' "$pipeline_pid" > logs/inspection_pipeline.pid
printf 'pipeline_pid=%s\n' "$pipeline_pid"
