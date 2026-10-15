#!/usr/bin/env bash
# Full-fidelity look at the crop_pest trainer: the earlier tail was ambiguous
# (a 124 MB download bar plus a metric warning, no epoch bar), and the log is
# append-mode across restarts, so the tail alone cannot say which run wrote what.
set -u
LOG=/root/loweye/logs/train_crop_pest.log
OUT=/root/autodl-tmp/loweye/crop_pest
PID=$(pgrep -f 'train_inspection_fcos.py --manifest' | head -1)

echo "=== date / uptime of trainer ==="
date
ps -o pid,lstart,etime,time,pcpu,stat,nlwp -p "${PID:-0}" 2>&1
echo "=== full train log (CR-normalised, numbered) ==="
tr '\r' '\n' < "$LOG" | grep -v '^ *$' | cat -n | tail -40
echo "=== log line count / size ==="
stat -c 'bytes=%s mtime=%y' "$LOG"
tr '\r' '\n' < "$LOG" | wc -l
echo "=== torch_cache contents (did the 124M download land?) ==="
find "$OUT/torch_cache" -type f -printf '%s\t%p\n' 2>&1 | head -20
find /root/.cache/torch -maxdepth 3 -type f -printf '%s\t%p\n' 2>/dev/null | head -10
echo "=== is anything else downloading? ==="
pgrep -af 'curl|wget|download_transport|fetch_' | head -10
echo "=== process io growth over 20s ==="
cat /proc/$PID/io 2>&1
sleep 20
cat /proc/$PID/io 2>&1
echo "=== process status ==="
head -8 /proc/$PID/status 2>&1
cat /proc/$PID/wchan 2>&1; echo
echo "=== threads ==="
ls /proc/$PID/task 2>/dev/null | wc -l
echo "=== open file descriptors (non-lib) ==="
ls -l /proc/$PID/fd 2>/dev/null | grep -vE 'socket|/dev/|pipe:|anon_inode' | tail -12
echo "=== network connections ==="
ss -tnp 2>/dev/null | grep -E 'python|ESTAB' | head -10
echo "=== py-spy available? ==="
which py-spy 2>&1 || echo "no py-spy"
echo "=== gpu samples ==="
for _ in 1 2 3 4; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 3; done
echo "=== out dir now ==="
ls -l "$OUT" 2>&1
echo "=== coordinator ==="
tails=$(pgrep -af 'run_inspection_tasks.py' | head -3); echo "$tails"
