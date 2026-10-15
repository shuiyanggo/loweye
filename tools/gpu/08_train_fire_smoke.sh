#!/usr/bin/env bash
set -euo pipefail
cd /root/loweye
export TORCH_HOME=/root/autodl-tmp/torch_home
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
PY=/root/miniconda3/bin/python
MODE=${1:-smoke}
BATCH=${BATCH:-8}
mkdir -p logs /root/autodl-tmp/loweye/fire_smoke /root/autodl-tmp/loweye/models
COMMON=(--manifest /root/loweye/data/dfire_split.json --task fire_smoke --out /root/autodl-tmp/loweye/fire_smoke --final-dir /root/autodl-tmp/loweye/models --batch "$BATCH" --size 640 --workers 8 --seed 42)
if [[ "$MODE" == smoke ]]; then
  exec "$PY" -u tools/gpu/train_inspection_fcos.py "${COMMON[@]}" --epochs 1 --limit-train 64 --limit-val 32 --smoke
elif [[ "$MODE" == full ]]; then
  nohup "$PY" -u tools/gpu/train_inspection_fcos.py "${COMMON[@]}" --epochs 100 > logs/train_fire_smoke.log 2>&1 < /dev/null &
  echo "$!" > logs/train_fire_smoke.pid
  echo "TRAIN_FIRE_SMOKE_PID=$!"
else
  echo 'mode must be smoke or full' >&2; exit 2
fi
