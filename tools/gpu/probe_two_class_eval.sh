#!/usr/bin/env bash
# Exercise the independent ONNX evaluator against a 2-class, 640 px artifact.
#
# The evaluator has only ever run against crop_pest's 102-class ONNX, so nothing has
# checked its class-count handling on fire_smoke's shape. This uses the probe ONNX
# produced by probe_fire_smoke_gate.sh plus the D-Fire probe manifest -- no real
# deliverable path is touched. Six test images are enough to prove the code path.
set -u
BIG=/root/autodl-tmp/loweye
PROBE_DATA=/root/loweye/logs/probe_data
mkdir -p "$PROBE_DATA"
cp /root/loweye/logs/fire_smoke_manifest_probe2.json "$PROBE_DATA/fire_smoke_manifest.json"
LOWEYE_DATA_DIR=$PROBE_DATA \
LOWEYE_ONNX_DIR=$BIG/models_probe \
LOWEYE_EVAL_OUT=/root/loweye/logs/fire_smoke_onnx_probe.json \
LOWEYE_EVAL_LIMIT=6 \
  bash /root/loweye/tools/gpu/evaluate_final_models.sh fire_smoke
echo "--- 2-class evaluator report ---"
/root/miniconda3/bin/python - <<'PY'
import json, pathlib
path = pathlib.Path('/root/loweye/logs/fire_smoke_onnx_probe.json')
if not path.is_file():
    print('no report produced'); raise SystemExit(1)
report = json.loads(path.read_text(encoding='utf-8'))
protocol = report.get('protocol', {})
coco = report.get('coco', {})
operating = report.get('operating_point', {}).get('overall', {})
print(json.dumps({
    'top_level_keys': sorted(report),
    'protocol': protocol,
    'coco_map': coco.get('map'),
    'coco_map50': coco.get('map50'),
    'operating_point': operating,
    'providers': report.get('runtime', {}).get('session_providers'),
    'onnx_sha256_16': str(report.get('artifact', {}).get('onnx_sha256', ''))[:16],
    'non_evaluable_classes': len(report.get('counts', {}).get('non_evaluable_classes', [])),
}, ensure_ascii=False, indent=1)[:3000])
PY
