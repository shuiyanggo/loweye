#!/bin/bash
PY=/root/miniconda3/bin/python3
echo "=== 冒烟日志 ==="
cat /root/autodl-tmp/loweye/smoke/train_fcos_log.json | $PY -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['log'], ensure_ascii=False, indent=1)); print('best', d['best'])"
echo "=== 装 onnx / onnxruntime（阿里云源）==="
timeout 600 $PY -m pip install -q onnx onnxruntime 2>&1 | tail -3
$PY -c "import onnx, onnxruntime; print('onnx', onnx.__version__, '| onnxruntime', onnxruntime.__version__)"
