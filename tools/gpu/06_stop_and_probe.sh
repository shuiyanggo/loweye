#!/bin/bash
echo "=== 停掉慢速下载 ==="
pkill -f fetch_cplid.py && echo "已停止" || echo "没有在跑"
sleep 1
echo "=== FCOS 权重用 Range GET 试（HEAD 可能被 CDN 拒）==="
for url in https://download.pytorch.org/models/fcos_resnet50_fpn_coco-bf2d0c1e.pth \
           https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-73cbd019.pth; do
  out=$(timeout 20 curl -s -r 0-1023 -o /tmp/_probe.bin -w '%{http_code} %{size_download}' "$url" 2>/dev/null)
  echo "$(basename $url) -> $out"
done
rm -f /tmp/_probe.bin
echo "=== 磁盘与目录 ==="
df -h /root/autodl-tmp | tail -1
ls /root/loweye
echo "=== 检查结束 ==="
