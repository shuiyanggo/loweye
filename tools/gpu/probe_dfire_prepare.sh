#!/usr/bin/env bash
# Pre-validate the D-Fire data preparation now, on CPU, while the GPU trains.
#
# The coordinator runs this same command (strict mode, --classes smoke fire) after
# crop_pest finishes -- i.e. roughly three hours from now. The IP102 path needed
# four attempts before it passed, so running the D-Fire path early against a
# *probe* output path means a strict-mode data problem surfaces now, with time to
# fix it, instead of after the GPU has gone idle.
#
# This never writes the coordinator's own manifest path.
set -u
OUT=/root/loweye/logs/fire_smoke_manifest_probe.json
LOG=/root/loweye/logs/probe_dfire_prepare.log
rm -f "$OUT"
echo "started $(date)" > "$LOG"
nohup /root/miniconda3/bin/python -u /root/loweye/tools/prepare_inspection_data.py \
  --root /root/loweye/data/external/dfire \
  --dataset dfire \
  --output "$OUT" \
  --classes smoke fire >> "$LOG" 2>&1 &
echo "pid=$!"
sleep 6
echo "--- log head ---"
tail -5 "$LOG"
