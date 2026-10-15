"""远端 FCOS 训练收尾：轮询产物、就绪后回收到本机。

这是 CPLID 绝缘子交付时期的收尾脚本（等 cplid_fcos_1024.onnx 就绪后回收 ONNX / best.pth / 训练日志）。
巡检管线（crop_pest、fire_smoke）不走这里：产物回收看 tools/gpu/monitor_inspection.py，
独立评测与本地落盘看 tools/gpu/finish_inspection_delivery.py。

端点必须由环境变量给出——本会话里实例换过三次，写死的旧地址只会连到已退役的机器：
    $env:LOWEYE_GPU_HOST='connect.<区域>.seetacloud.com'
    $env:LOWEYE_GPU_PORT='<端口>'
    $env:LOWEYE_GPU_PASS='<密码>'
    python -m tools.gpu.await_and_fetch
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
REMOTE_MODELS = "/root/autodl-tmp/loweye/models"
REMOTE_LOG = "/root/loweye/logs/train_fcos.log"


def endpoint_from_env() -> tuple[str, str]:
    """缺任何一个环境变量就立刻退出，不退回某个默认端点。"""
    host = os.environ.get("LOWEYE_GPU_HOST")
    port = os.environ.get("LOWEYE_GPU_PORT")
    if not host or not port:
        raise SystemExit("缺少环境变量 LOWEYE_GPU_HOST / LOWEYE_GPU_PORT（AutoDL 地址与端口）")
    if not os.environ.get("LOWEYE_GPU_PASS"):
        raise SystemExit("缺少环境变量 LOWEYE_GPU_PASS（AutoDL root 密码）")
    return host, port


def remote(host: str, port: str, action: str, **kw: str) -> str:
    cmd = [str(PY), "-m", "tools.remote_gpu", action, "--host", host, "--port", port, "--timeout", "180"]
    for key, value in kw.items():
        cmd += [f"--{key}", value]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    host, port = endpoint_from_env()
    for round_no in range(45):
        status = remote(host, port, "run", cmd=f"tail -1 {REMOTE_LOG}; ls -l {REMOTE_MODELS}/ 2>/dev/null | tail -6")
        tail = " / ".join(line.strip() for line in status.strip().splitlines() if line.strip())[-420:]
        print(f"[{round_no:02d}] {tail}", flush=True)
        if "cplid_fcos_1024.onnx" in status:
            print("ONNX 已就绪，开始回收", flush=True)
            for remote_path, local in (
                (f"{REMOTE_MODELS}/cplid_fcos_1024.onnx", ROOT / "models" / "detector" / "cplid_fcos_1024.onnx"),
                (f"{REMOTE_MODELS}/cplid_fcos_best.pth", ROOT / "models" / "detector" / "cplid_fcos_best.pth"),
                (REMOTE_LOG, ROOT / "out" / "train_fcos_gpu_log.jsonl"),
            ):
                print(f"下载 {remote_path}", flush=True)
                print(remote(host, port, "download", remote=remote_path, local=str(local))[-300:], flush=True)
            print("回收完成", flush=True)
            return 0
        time.sleep(60)
    print("超时未拿到 ONNX，需人工检查远端", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
