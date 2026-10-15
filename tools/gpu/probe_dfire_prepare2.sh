#!/usr/bin/env bash
# D-Fire probe 2: the same preparation the coordinator runs, plus --quarantine-invalid.
#
# Probe 1 (probe_dfire_prepare.sh, strict defaults) proved this source cannot ever
# produce a manifest under the strict converter: D-Fire's YOLO labels contain boxes
# the converter rejects ("nonfinite/out-of-range coordinates or empty box",
# "box extends beyond image"). The IP102 branch already uses --quarantine-invalid for
# the same class of source defect, so the question is whether the same switch yields a
# usable manifest here and how many samples it excludes.
#
# Runs synchronously so one call returns the verdict; summarize_prepare_errors.py
# prints counts only, and probe 1's error report is kept untouched as evidence.
set -u
OUT=/root/loweye/logs/fire_smoke_manifest_probe2.json
LOG=/root/loweye/logs/probe_dfire_prepare2.log
rm -f "$OUT"
echo "started $(date)" > "$LOG"
/root/miniconda3/bin/python -u /root/loweye/tools/prepare_inspection_data.py \
  --root /root/loweye/data/external/dfire \
  --dataset dfire \
  --output "$OUT" \
  --classes smoke fire \
  --quarantine-invalid >> "$LOG" 2>&1
echo "prepare_exit=$?"
echo "--- probe 1 (strict) failure classification ---"
/root/miniconda3/bin/python /root/loweye/tools/gpu/summarize_prepare_errors.py \
  /root/loweye/logs/fire_smoke_manifest_probe.json
echo "--- probe 2 (quarantine) outcome ---"
/root/miniconda3/bin/python /root/loweye/tools/gpu/summarize_prepare_errors.py "$OUT"
echo "--- prepare log tail ---"
tail -4 "$LOG"
