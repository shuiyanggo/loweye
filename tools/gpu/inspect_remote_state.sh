#!/usr/bin/env bash
# Read-only remote state inspection. No writes, no training, no deletion.
set -u
echo "=== date/uptime ==="; date; uptime
echo "=== disk ==="; df -h /root /root/autodl-tmp 2>/dev/null
echo "=== gpu ==="; nvidia-smi 2>/dev/null | head -15
echo "=== /root/loweye root ==="; ls -la /root/loweye/ 2>/dev/null | head -30
echo "=== data ==="; ls -lh /root/loweye/data/ 2>/dev/null
echo "=== external (recursive, top) ==="; ls -lhR /root/loweye/data/external/ 2>/dev/null | head -80
echo "=== models ==="; ls -lhR /root/loweye/models 2>/dev/null | head -40
echo "=== logs (newest first) ==="; ls -lht /root/loweye/logs/ 2>/dev/null | head -40
echo "=== pipeline status json ==="
for f in /root/loweye/logs/*_pipeline_status.json; do
  [ -e "$f" ] || continue
  echo "--- $f"; cat "$f"; echo
done
echo "=== running python processes ==="; ps aux | grep python | grep -v grep | head -20
echo "=== tail prepare_crop_pest.log ==="; tail -8 /root/loweye/logs/prepare_crop_pest.log 2>/dev/null
echo "=== tail prepare_fire_smoke.log ==="; tail -8 /root/loweye/logs/prepare_fire_smoke.log 2>/dev/null
echo "=== tail fetch logs ==="
tail -3 /root/loweye/logs/fetch_dfire.log 2>/dev/null
tail -3 /root/loweye/logs/fetch_ip102.log 2>/dev/null
echo "=== manifest files ==="; ls -lh /root/loweye/data/*manifest*.json 2>/dev/null
