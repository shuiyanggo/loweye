#!/usr/bin/env bash
# 查看某个任务的冒烟门控是否真的导出了 ONNX、导出校验（parity）证据是什么、
# 以及产物落在哪里。只读，不改任何文件。
#
# 用法（远端）: bash tools/gpu/inspect_export_evidence.sh crop_pest [800]
set -u

TASK="${1:-crop_pest}"
LOGS=/root/loweye/logs
MODELS=/root/autodl-tmp/loweye/models

echo "== 任务: ${TASK}"
echo "== 冒烟日志"
ls -l --time-style=+%m-%d_%H:%M "${LOGS}/smoke_${TASK}.log" 2>/dev/null || echo "（无冒烟日志）"
echo "== 冒烟日志尾部"
tail -12 "${LOGS}/smoke_${TASK}.log" 2>/dev/null || true
echo "== 导出校验证据"
grep -o 'onnx_export_evidence.*' "${LOGS}/smoke_${TASK}.log" 2>/dev/null | tail -1 | cut -c1-700 || true
echo
echo "== 日志里出现的 onnx 路径"
grep -o '/root[^ ]*\.onnx' "${LOGS}/smoke_${TASK}.log" 2>/dev/null | sort -u || true
echo "== models 目录"
ls -l --time-style=+%m-%d_%H:%M "${MODELS}" 2>/dev/null || echo "（无 models 目录）"
echo "== 全盘找该任务的 onnx（最近 5 个）"
find /root -maxdepth 5 -name "*${TASK}*.onnx" -printf '%TY-%Tm-%Td_%TH:%TM %10s %p\n' 2>/dev/null | sort | tail -5
echo "== 冒烟脚手架目录"
find /root/autodl-tmp/loweye -maxdepth 1 -name '*smoke*' -printf '%TY-%Tm-%Td_%TH:%TM %10s %p\n' 2>/dev/null | sort | tail -5
