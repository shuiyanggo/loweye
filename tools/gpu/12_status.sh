#!/bin/bash
echo "=== 训练日志（最后 8 行）==="
tail -8 /root/loweye/logs/train_fcos.log
echo "=== 进程 ==="
pgrep -af train_cplid_fcos.py | head -3
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== 产物 ==="
ls -l /root/autodl-tmp/loweye/models/ 2>/dev/null | tail -5
