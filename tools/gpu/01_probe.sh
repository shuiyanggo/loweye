#!/bin/bash
# 远端环境探测：python、torch、CUDA、网络、可写数据盘
PY=/root/miniconda3/bin/python3
echo "=== python ==="
$PY -V
echo "=== torch ==="
timeout 200 $PY - <<'PYEOF'
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as exc:
    print("torch 不可用:", type(exc).__name__, exc)
try:
    import torchvision
    print("torchvision", torchvision.__version__)
except Exception as exc:
    print("torchvision 不可用:", type(exc).__name__, exc)
PYEOF
echo "=== 网络（各 10 秒上限）==="
for url in https://pypi.org/simple/ https://mirrors.aliyun.com/pypi/simple/ https://hf-mirror.com https://github.com; do
  code=$(timeout 10 curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
  echo "$url -> ${code:-超时/失败}"
done
echo "=== pip 源 ==="
$PY -m pip config list 2>/dev/null || echo "(无 pip 配置)"
echo "=== 磁盘 ==="
df -h /root/autodl-tmp / 2>/dev/null | tail -3
echo "=== 探测结束 ==="
