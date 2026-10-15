#!/usr/bin/env bash
# Read-only: dig into the crop_pest smoke-training failure. No writes.
set -u
echo "=== smoke log: matched error lines ==="
grep -n -i -E "error|traceback|exception|assert|raise|failed|Warning" /root/loweye/logs/smoke_crop_pest.log | tail -40
echo "=== smoke log tail 70 ==="
tail -70 /root/loweye/logs/smoke_crop_pest.log
echo "=== smoke log head 25 ==="
head -25 /root/loweye/logs/smoke_crop_pest.log
echo "=== pipeline log tail 45 ==="
tail -45 /root/loweye/logs/inspection_pipeline.log
echo "=== smoke out dirs ==="
ls -lhR /root/autodl-tmp/loweye 2>/dev/null | head -50
echo "=== manifest summary ==="
/root/miniconda3/bin/python - <<'PY'
import json
d = json.load(open('/root/loweye/data/crop_pest_manifest.json'))
for k, v in d.items():
    if isinstance(v, list):
        print(k, 'list len', len(v), 'first', v[:2] if v else None)
    else:
        print(k, repr(v)[:300])
PY
