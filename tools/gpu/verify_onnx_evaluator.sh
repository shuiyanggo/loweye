#!/usr/bin/env bash
# Plumbing check for the independent ONNX evaluator, against the smoke artifact.
#
# Formal training owns the GPU while this runs, and the host's onnxruntime is
# the CPU build, so keep the thread count low and the image count tiny. The
# point is to prove the evaluator runs end to end, that its letterbox matches
# the training dataset exactly, and that the COCO path produces finite numbers
# -- before it is trusted with the 3798-image official test split.
set -u
PY=/root/miniconda3/bin/python
ONNX=/root/autodl-tmp/loweye/models_smoke/crop_pest_fcos_800.onnx
MANIFEST=/root/loweye/data/crop_pest_manifest.json
OUT=/root/loweye/logs/crop_pest_onnx_evaluator_check.json

echo "=== artifacts ==="
ls -l "$ONNX" "$MANIFEST" 2>&1
echo "=== onnxruntime build ==="
$PY -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())" 2>&1 | tail -2
echo "=== evaluator: 8 val images, letterbox parity check ==="
$PY /root/loweye/tools/gpu/evaluate_onnx_detector.py \
  --onnx "$ONNX" --manifest "$MANIFEST" --split val --limit 8 \
  --check-preprocessing 8 --threads 4 --progress-every 4 --out "$OUT" 2>&1 | tail -25
echo "=== evaluator report summary ==="
$PY - <<'PYEOF' 2>&1 | tail -20
import json
path = '/root/loweye/logs/crop_pest_onnx_evaluator_check.json'
try:
    d = json.load(open(path))
except Exception as exc:
    print('no report:', type(exc).__name__, exc)
else:
    print(json.dumps({
        'map': d['coco']['map'], 'map50': d['coco']['map50'],
        'images': d['protocol']['images'],
        'gt_boxes': d['counts']['ground_truth_boxes'],
        'detections_at_floor': d['counts']['detections_at_floor'],
        'letterbox_parity': d['preprocessing_parity'],
        'seconds_per_image': d['timing']['seconds_per_image'],
        'non_evaluable_classes': len(d['counts']['non_evaluable_classes']),
        'session_providers': d['runtime']['session_providers'],
    }, ensure_ascii=False, indent=1))
PYEOF
echo "=== training progress ==="
if [ -f /root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl ]; then
  wc -l /root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl
  tail -2 /root/autodl-tmp/loweye/crop_pest/crop_pest_epochs.jsonl
else
  echo "no epochs file yet"
fi
echo "--- train log tail ---"
tail -c 500 /root/loweye/logs/train_crop_pest.log 2>/dev/null
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- trainers ---"
pgrep -fc train_inspection_fcos.py
