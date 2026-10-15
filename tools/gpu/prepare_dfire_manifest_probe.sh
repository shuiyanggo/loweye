#!/usr/bin/env bash
# CPU-only rehearsal of the D-Fire manifest derivation that the coordinator will run
# once crop_pest finishes. Nothing here uses the coordinator's manifest path and no
# GPU process is started: it only answers "do D-Fire's YOLO labels survive
# --quarantine-invalid on this host", which would otherwise only be discovered after
# the GPU had already gone idle on a failure.
set -u
ROOT=/root/loweye
BIG=/root/autodl-tmp/loweye
STAGE=$BIG/_presmoke
MANIFEST=$STAGE/fire_smoke_manifest.json
mkdir -p "$STAGE"
cd "$ROOT" || exit 1
echo "=== start $(date '+%F %T') host=$(hostname)"
echo "=== dataset root"
ls "$ROOT/data/external/dfire" | head -20
echo "=== prepare"
/root/miniconda3/bin/python tools/prepare_inspection_data.py \
  --root "$ROOT/data/external/dfire" \
  --dataset dfire \
  --output "$MANIFEST" \
  --classes smoke fire \
  --quarantine-invalid
echo "prepare_exit=$?"
ls -la "$STAGE"
echo "=== manifest summary"
/root/miniconda3/bin/python -X utf8 - "$MANIFEST" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    print('MANIFEST_MISSING')
    raise SystemExit(0)
d = json.loads(p.read_text())
print('status', d.get('status'))
print('classes', d.get('classes'))
print('keys', sorted(d.keys()))
for split in ('train', 'val'):
    value = d.get(split)
    print(split, len(value) if isinstance(value, list) else value)
audit = d.get('source_group_audit') or {}
print('cross_split_groups', len(audit.get('cross_split_groups') or []))
print('audit_keys', sorted(audit.keys()))
for key in ('quarantine', 'quarantined', 'excluded', 'audit', 'source_audit'):
    if key in d:
        print(key, json.dumps(d[key], ensure_ascii=False)[:600])
PYEOF
echo "=== done $(date '+%F %T')"
