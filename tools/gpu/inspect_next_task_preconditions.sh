#!/usr/bin/env bash
# 只读：核对「下一个会被协调器选中的任务」的前置条件。
# 协调器 tools/gpu/run_inspection_tasks.py 第 85-104 行会在选中任务后先校验
# data/external/<dataset>/source_license.json 与 <archive>.transport.json 的哈希是否等于钉死的 md5，
# 不符就直接抛 "Pinned source archive hash mismatch" 并把该任务记为 failed。
# fire_smoke 从未被选中过，所以这条校验对它尚未跑过一次 —— 这里先把它跑一遍，别等到 GPU 空转才发现。
set -u
cd /root/loweye
PIN_DFIRE=9745236834d65441af996cb6da6a4fde
PIN_IP102=85869d5c3c686e936c0dde780b208b01

echo "== 任务相位（logs/<task>_pipeline_status.json 原文）"
for t in fire_smoke crop_pest; do
  echo "-- $t"
  cat "logs/${t}_pipeline_status.json" 2>/dev/null || echo MISSING
  echo
done

echo "== fire_smoke / dfire 前置"
D=data/external/dfire
ls -la "$D" 2>/dev/null | head -12
echo "-- source_license.json"
cat "$D/source_license.json" 2>/dev/null || echo MISSING
echo "-- dfire_v1.transport.json（应含 md5=$PIN_DFIRE）"
cat "$D/dfire_v1.transport.json" 2>/dev/null || echo MISSING
echo "-- 解包文件计数与体积"
find "$D" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | wc -l
du -sh "$D" 2>/dev/null

echo "== crop_pest / ip102 前置（已跑过校验，这里只看现状）"
I=data/external/ip102
cat "$I/source_license.json" 2>/dev/null | head -c 400 || echo MISSING
echo
echo "-- ip102_v1.transport.json（应含 md5=$PIN_IP102）"
cat "$I/ip102_detection_v1.transport.json" 2>/dev/null | head -c 400 || echo MISSING
echo

echo "== manifest 与类别表"
ls -l data/fire_smoke_manifest.json data/crop_pest_manifest.json data/ip102_classes.txt 2>/dev/null
wc -l data/ip102_classes.txt 2>/dev/null

echo "== 磁盘"
df -h /root /root/autodl-tmp 2>/dev/null | tail -3

echo "== GPU / 训练进程"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -af 'run_inspection_tasks|train_inspection_fcos' | head -4
date
