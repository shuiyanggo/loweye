#!/usr/bin/env bash
# Independent evaluation of the shipped ONNX artifacts on the official test split.
#
# The trainer already scored its own PyTorch module on the test split once and
# wrote an immutable receipt; this runs the *delivered file* through a different
# runtime (onnxruntime) and a different metric implementation (pycocotools
# COCOeval called directly, not through torchmetrics). Both numbers are reported;
# a gap between them means the export changed behaviour.
#
#   bash evaluate_final_models.sh                  # both tasks, test split
#   bash evaluate_final_models.sh crop_pest        # one task
#
# Environment overrides exist so the harness itself can be exercised against the
# smoke artifacts before the real ones exist:
#   LOWEYE_ONNX_DIR   directory holding {task}_fcos_{size}.onnx
#   LOWEYE_DATA_DIR   directory holding {task}_manifest.json
#   LOWEYE_EVAL_OUT   report path
#   LOWEYE_EVAL_LIMIT image count limit
set -u
PY=/root/miniconda3/bin/python
EVALUATOR=/root/loweye/tools/gpu/evaluate_onnx_detector.py
DATA="${LOWEYE_DATA_DIR:-/root/loweye/data}"
BIG=/root/autodl-tmp/loweye
LOGS=/root/loweye/logs
ONNX_DIR="${LOWEYE_ONNX_DIR:-$BIG/models}"
WANT="${1:-crop_pest fire_smoke}"

resolve_manifest() {
  local base="$DATA/$1_manifest.json"
  if [ -f "${base%.json}_group_safe_manifest.json" ]; then
    echo "${base%.json}_group_safe_manifest.json"
  else
    echo "$base"
  fi
}

evaluate() {
  local task="$1" size="$2"
  local onnx="$ONNX_DIR/${task}_fcos_${size}.onnx"
  local manifest out
  manifest=$(resolve_manifest "$task")
  out="${LOWEYE_EVAL_OUT:-$LOGS/${task}_onnx_test.json}"
  echo "=== $task (size $size) ==="
  if [ ! -f "$onnx" ]; then echo "SKIP: $onnx does not exist yet"; return 1; fi
  if [ ! -f "$manifest" ]; then echo "SKIP: $manifest does not exist yet"; return 1; fi
  if [ -z "${LOWEYE_EVAL_LIMIT:-}" ] && [ -f "$out" ]; then
    echo "already evaluated: $out (delete it to re-run)"
    return 0
  fi
  local extra=()
  [ -n "${LOWEYE_EVAL_LIMIT:-}" ] && extra+=(--limit "$LOWEYE_EVAL_LIMIT")
  echo "onnx:     $onnx"
  echo "manifest: $manifest"
  echo "out:      $out"
  # 线程数实测（bjb2，crop_pest 800px，16 张 test 图）：16 线程 4.9 秒、32 线程 14.5 秒、96 线程 78.2 秒。
  # 容器里 ORT 线程数超过实际可用 CPU 配额会互相抢，越多越慢，所以默认取 16 而不是 nproc。
  "$PY" "$EVALUATOR" --onnx "$onnx" --manifest "$manifest" --split test \
    --threads "${LOWEYE_EVAL_THREADS:-16}" --progress-every 250 --out "$out" "${extra[@]+"${extra[@]}"}"
}

status=0
for task in $WANT; do
  case "$task" in
    crop_pest) evaluate crop_pest 800 || status=1 ;;
    fire_smoke) evaluate fire_smoke 640 || status=1 ;;
    *) echo "unknown task: $task"; status=1 ;;
  esac
done

echo "=== reports ==="
ls -l "$LOGS"/*_onnx_test.json 2>&1
echo "=== trainer-side test receipts (independent of this run) ==="
"$PY" - <<'PYEOF'
import json, pathlib
big = pathlib.Path('/root/autodl-tmp/loweye')
for task in ('crop_pest', 'fire_smoke'):
    path = big / task / f'{task}_test.json'
    if not path.is_file():
        print(f'{task}: no trainer test receipt at {path}')
        continue
    receipt = json.loads(path.read_text(encoding='utf-8'))
    metrics = receipt.get('metrics', {})
    overall = metrics.get('operating_point', {}).get('overall', {})
    print(json.dumps({'task': task, 'evaluations': receipt.get('evaluations'),
                      'checkpoint_sha256': receipt.get('checkpoint_sha256', '')[:16],
                      'map': metrics.get('map'), 'map50': metrics.get('map50'),
                      'recall': overall.get('recall'),
                      'false_alarms_per_image': overall.get('false_alarms_per_image')},
                     ensure_ascii=False))
PYEOF
echo "=== evaluator reports (onnxruntime + pycocotools) ==="
"$PY" - <<'PYEOF'
import json, pathlib
for path in sorted(pathlib.Path('/root/loweye/logs').glob('*_onnx_test.json')):
    report = json.loads(path.read_text(encoding='utf-8'))
    coco = report['coco']
    overall = report['operating_point']['overall']
    print(json.dumps({'task': path.name.split('_onnx_test')[0],
                      'split': report['protocol']['split'], 'images': report['protocol']['images'],
                      'map': coco.get('map'), 'map50': coco.get('map50'),
                      'recall': overall.get('recall'),
                      'false_alarms_per_image': overall.get('false_alarms_per_image'),
                      'seconds': report['timing']['seconds'],
                      'providers': report['runtime']['session_providers'],
                      'onnx_sha256': report['artifact']['onnx_sha256'][:16],
                      'non_evaluable_classes': len(report['counts']['non_evaluable_classes'])},
                     ensure_ascii=False))
PYEOF
exit "$status"
