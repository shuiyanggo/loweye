#!/bin/bash
# 查 openmmlab 有没有匹配 torch 2.8 / cu128 / python3.12 的预编译 mmcv 轮子
BASE=https://download.openmmlab.com/mmcv/dist
for combo in cu128/torch2.8.0 cu126/torch2.8.0 cu124/torch2.6.0 cu121/torch2.2.0 cu118/torch2.0.0; do
  code=$(timeout 10 curl -s -o /dev/null -w '%{http_code}' "$BASE/$combo/index.html" 2>/dev/null)
  echo "$combo -> ${code:-超时}"
done
echo "=== cu128/torch2.8.0 的 cp312 轮子 ==="
timeout 20 curl -s "$BASE/cu128/torch2.8.0/index.html" 2>/dev/null | grep -oE 'mmcv-[0-9.]+-cp312[^"<]*' | sort -u | head -8
echo "=== 顺带看看 cu126/torch2.8.0 ==="
timeout 20 curl -s "$BASE/cu126/torch2.8.0/index.html" 2>/dev/null | grep -oE 'mmcv-[0-9.]+-cp312[^"<]*' | sort -u | head -8
echo "=== 检查结束 ==="
