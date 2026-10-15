#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/bin/python
mkdir -p /root/loweye/logs /root/loweye/tools/gpu /root/autodl-tmp/loweye/fire_smoke /root/autodl-tmp/loweye/crop_pest
# Keep existing torch/torchvision and CPLID artifacts untouched.
$PY -m pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple torchmetrics==1.8.2 pycocotools==2.0.10 lightning-utilities==0.15.2
$PY - <<'PY'
from importlib.metadata import version
for name in ('torch','torchvision','onnx','onnxruntime','torchmetrics','pycocotools','lightning-utilities'):
    print(name, version(name), flush=True)
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
print('COCO evaluator import OK')
PY
