#!/usr/bin/env bash
# GPU rehearsal of the D-Fire smoke gate the coordinator will run once crop_pest
# finishes. Scratch output paths only: the coordinator's own fire_smoke_smoke /
# models_smoke stay untouched, so its result can never be confused with this one.
# Purpose: a freshly initialised 2-class head must produce a non-empty ONNX
# verification, and the coordinator treats an empty one as a hard failure for that
# task. Discovering it at 01:00 would waste the rest of the night; discovering it
# now leaves time to fix whatever makes it empty.
set -u
ROOT=/root/loweye
BIG=/root/autodl-tmp/loweye
STAGE=$BIG/_presmoke
cd "$ROOT" || exit 1
echo "=== start $(date '+%F %T')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
/root/miniconda3/bin/python tools/gpu/train_inspection_fcos.py \
  --manifest "$STAGE/fire_smoke_manifest.json" \
  --task fire_smoke \
  --size 640 \
  --batch 4 \
  --out "$STAGE/fire_smoke_smoke" \
  --final-dir "$STAGE/models_smoke" \
  --smoke --workers 0
echo "smoke_exit=$?"
echo "=== artifacts"
ls -la "$STAGE/models_smoke" 2>/dev/null
ls -la "$STAGE/fire_smoke_smoke" 2>/dev/null
echo "=== verification"
/root/miniconda3/bin/python -X utf8 - "$STAGE/models_smoke/fire_smoke_fcos_metadata.json" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    print('METADATA_MISSING')
    raise SystemExit(0)
d = json.loads(p.read_text())
print('classes', d.get('classes'))
print('input', (d.get('preprocessing') or {}).get('input'))
print('onnx', json.dumps(d.get('onnx'), ensure_ascii=False)[:1200])
print('empty_detections', ((d.get('onnx') or {}).get('verification') or {}).get('empty_detections'))
print('test_status', d.get('test_status'))
PYEOF
echo "=== done $(date '+%F %T')"
