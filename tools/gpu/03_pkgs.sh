#!/bin/bash
# 远端依赖可行性探测：mmdet/mmcv 能不能装、openmmlab 轮子站通不通
PY=/root/miniconda3/bin/python3
echo "=== 版本可用性（各 60 秒上限）==="
for pkg in mmengine mmcv mmdet; do
  echo "--- $pkg ---"
  timeout 60 $PY -m pip index versions "$pkg" 2>&1 | head -2
done
echo "=== 站点可达性 ==="
for url in https://download.openmmlab.com/mmcv/dist/index.html https://mirrors.aliyun.com/pypi/simple/ https://pypi.tuna.tsinghua.edu.cn/simple/; do
  code=$(timeout 10 curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
  echo "$url -> ${code:-超时}"
done
echo "=== 已装的相关包 ==="
$PY -m pip list 2>/dev/null | grep -iE 'mmcv|mmdet|mmengine|torch|onnx|opencv|numpy|pycoco|pillow'
echo "=== CUDA 工具链（编译 mmcv 用）==="
command -v nvcc && nvcc --version | tail -1 || echo "无 nvcc"
echo "=== 探测结束 ==="
