#!/bin/bash
PY=/root/miniconda3/bin/python3
echo "=== CPLID 下载进度 ==="
tail -2 /root/loweye/logs/cplid_download.log 2>/dev/null | tr '\r' '\n' | tail -2
echo "文件数: $(find /root/loweye/data/external/cplid -type f 2>/dev/null | wc -l)"
du -sh /root/loweye/data/external/cplid 2>/dev/null
echo "=== 权重站可达性 ==="
for url in https://download.pytorch.org/models/fcos_resnet50_fpn_coco-bf2d0c1e.pth https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth; do
  code=$(timeout 12 curl -s -o /dev/null -w '%{http_code}' -I "$url" 2>/dev/null)
  echo "$(basename $url) -> ${code:-超时}"
done
echo "=== torchvision 版本与可用权重枚举 ==="
$PY - <<'PYEOF'
import torchvision
from torchvision.models import detection as D
print("torchvision", torchvision.__version__)
names = [n for n in dir(D) if n.endswith("_Weights") or "Weights" in n]
print("权重枚举类:", [n for n in names if "Weights" in n][:12])
PYEOF
echo "=== 检查结束 ==="
