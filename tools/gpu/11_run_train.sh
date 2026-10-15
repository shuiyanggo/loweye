#!/bin/bash
# 正式训练：FCOS-R50-FPN / CPLID 两类 / 1024 输入 / 30 epoch，后台跑
PY=/root/miniconda3/bin/python3
cd /root/loweye
export TORCH_HOME=/root/autodl-tmp/torch_home
export PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/loweye/models /root/loweye/logs

if pgrep -f train_cplid_fcos.py >/dev/null; then
  echo "已有训练在跑，PID $(pgrep -f train_cplid_fcos.py | tr '\n' ' ')"
  exit 0
fi

nohup $PY tools/gpu/train_cplid_fcos.py \
  --data /root/loweye/data/external/cplid \
  --split /root/loweye/data/cplid_split.json \
  --out /root/autodl-tmp/loweye/models \
  --epochs 30 --batch 6 --size 1024 --workers 12 \
  > logs/train_fcos.log 2>&1 &
sleep 5
echo "训练已后台启动，PID $(pgrep -f train_cplid_fcos.py | tr '\n' ' ')"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
