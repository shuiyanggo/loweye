#!/bin/bash
# 解包数据 + 后台启动 FCOS 微调
PY=/root/miniconda3/bin/python3
mkdir -p /root/loweye/data/external/cplid /root/autodl-tmp/loweye/models /root/loweye/logs /root/loweye/tools/gpu

echo "=== 远端 torchvision 期望的权重文件名 ==="
$PY -c "from torchvision.models.detection import FCOS_ResNet50_FPN_Weights as F; print(F.COCO_V1.url.rsplit('/',1)[-1])"
ls -l /root/autodl-tmp/torch_home/hub/checkpoints/ 2>/dev/null || echo "(权重目录还不存在)"

if [ -f /root/autodl-tmp/cplid.tar ]; then
  echo "=== 解包 CPLID ==="
  tar -xf /root/autodl-tmp/cplid.tar -C /root/loweye/data/external/cplid
  echo "图片 $(find /root/loweye/data/external/cplid -name '*.jpg' | wc -l) 张，标注 $(find /root/loweye/data/external/cplid -name '*.xml' | wc -l) 个"
else
  echo "缺 /root/autodl-tmp/cplid.tar"
fi

if [ -s /root/loweye/tools/gpu/train_cplid_fcos.py ] && [ -s /root/loweye/data/cplid_split.json ]; then
  echo "=== 后台启动训练 ==="
  cd /root/loweye
  export TORCH_HOME=/root/autodl-tmp/torch_home
  nohup $PY tools/gpu/train_cplid_fcos.py \
      --data /root/loweye/data/external/cplid \
      --split /root/loweye/data/cplid_split.json \
      --out /root/autodl-tmp/loweye/models \
      --epochs 24 --batch 4 --size 1024 --workers 12 \
      > logs/train_fcos.log 2>&1 &
  echo "训练已后台启动，PID $!"
else
  echo "训练脚本或划分文件还没上传，先跳过启动"
fi
