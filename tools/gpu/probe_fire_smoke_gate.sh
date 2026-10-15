#!/usr/bin/env bash
# Probe the fire_smoke smoke gate now, on the live GPU, while crop_pest training continues.
#
# Why: the smoke gate is the trainer's single hard gate. crop_pest already passed it, but
# fire_smoke differs in three ways that have never been exercised end to end:
#   1) a 2-class head instead of 102 (fewer classes => flatter score landscape => the
#      tie-aware box assertion may see zero unambiguous ranks);
#   2) the manifest produced by the D-Fire quarantine fix (probe 2);
#   3) size 640 instead of 800.
# If any of those fails, the coordinator would hit it roughly three hours from now with
# the GPU already idle. Running it now costs about a minute of a GPU that is merely
# 4.2 GB into 24 GB.
#
# Isolation: writes only to its own probe directories, so the coordinator's real
# fire_smoke/ and models/ paths are untouched, and the monitor watches different paths.
# TORCH_HOME is pre-set to crop_pest's existing cache because the trainer's
# os.environ.setdefault would otherwise re-download the 124 MB COCO weights.
set -u
BIG=/root/autodl-tmp/loweye
OUT=$BIG/fire_smoke_probe
FINAL=$BIG/models_probe
LOG=/root/loweye/logs/probe_fire_smoke_gate.log
# The first probe run pointed at fire_smoke_manifest_probe.json, which is the FAILURE
# report from the strict attempt (prepare stamps failures at its --output path) -- it has
# no "classes" key, so the trainer raised "classes must be a nonempty unique string list"
# and told us nothing about the gate. guard against that here.
MANIFEST=/root/loweye/logs/fire_smoke_manifest_probe2.json
/root/miniconda3/bin/python - "$MANIFEST" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data.get("usable_for_training") is not False, f"probe manifest is a failure report: {data.get('status')}"
assert data.get("classes"), "probe manifest has no classes"
print("probe manifest ok: classes=%s train=%d val=%d test=%d" % (
    data["classes"], len(data["train"]), len(data["val"]), len(data["test"])))
PY
if [ $? -ne 0 ]; then
  echo "probe manifest is not a training manifest; aborting"
  exit 2
fi
rm -rf "$OUT" "$FINAL"
mkdir -p "$OUT" "$FINAL"
echo "probe start $(date)" | tee "$LOG"
TORCH_HOME=$BIG/crop_pest/torch_cache \
  /root/miniconda3/bin/python -u /root/loweye/tools/gpu/train_inspection_fcos.py \
  --manifest "$MANIFEST" \
  --task fire_smoke --size 640 --batch 4 --workers 2 \
  --out "$OUT" --final-dir "$FINAL" --epochs 24 --smoke >> "$LOG" 2>&1
echo "probe exit=$?" | tee -a "$LOG"
echo "--- evidence lines ---"
grep -h -E 'onnx_export_evidence|onnx_verification|AssertionError|Error|Traceback' "$LOG" | tail -20
echo "--- produced ---"
ls -la "$FINAL" "$OUT" 2>/dev/null
