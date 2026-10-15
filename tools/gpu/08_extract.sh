#!/bin/bash
PY=/root/miniconda3/bin/python3
echo "=== 远端 torchvision 期望的权重文件名 ==="
$PY -c "from torchvision.models.detection import FCOS_ResNet50_FPN_Weights as F; print(F.COCO_V1.url.rsplit('/',1)[-1])"
echo "=== 解包 CPLID ==="
rm -rf /root/loweye/data/external/cplid
mkdir -p /root/loweye/data/external/cplid
tar -xf /root/autodl-tmp/cplid.tar -C /root/loweye/data/external/cplid
echo "图片 $(find /root/loweye/data/external/cplid -name '*.jpg' | wc -l) 张"
echo "标注 $(find /root/loweye/data/external/cplid -name '*.xml' | wc -l) 个"
du -sh /root/loweye/data/external/cplid
echo "=== 权重缓存目录 ==="
ls -l /root/autodl-tmp/torch_home/hub/checkpoints/ 2>/dev/null || echo "(还没有权重，等上传)"
