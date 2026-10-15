#!/bin/bash
# 冒烟测试：1 轮 + 24 张图，验证脚本能不能跑通（含 ONNX 导出）
PY=/root/miniconda3/bin/python3
cd /root/loweye
export TORCH_HOME=/root/autodl-tmp/torch_home
export PYTHONUNBUFFERED=1
$PY tools/gpu/train_cplid_fcos.py \
  --data /root/loweye/data/external/cplid \
  --split /root/loweye/data/cplid_split.json \
  --out /root/autodl-tmp/loweye/smoke \
  --epochs 1 --limit-train 24 --batch 2 --workers 4 --size 1024
echo "冒烟退出码 $?"
ls -l /root/autodl-tmp/loweye/smoke
