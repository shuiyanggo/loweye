"""本机侧看门狗：实例一重启，主机内的进程全死，流水线不会自己回来。

已发生两次实例重启，每次都丢掉数小时训练。主机内任何看门狗都会跟着一起死，
所以这个脚本跑在本机：每 5 分钟连一次主机，如果协调器不在、且还有任务没完成，
就通过 06_launch_pipeline.sh 重新拉起。

安全性来自两层：
1. 重新拉起走的是协调器本身，它带 --resume-last，续的是最后一个完成轮次，
   manifest / checkpoint / 日志都在数据盘上活着；
2. 06_launch_pipeline.sh 在 pidfile 指向活进程时拒绝启动（already running + exit 0），
   所以不会出现两个协调器同时写同一批 checkpoint。

环境变量：LOWEYE_GPU_HOST / LOWEYE_GPU_PORT / LOWEYE_GPU_PASS
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta

import paramiko

LOGS = "/root/loweye/logs"
TASKS = ("crop_pest", "fire_smoke")
POLL_SECONDS = 300
DEADLINE_HOURS = 72


def log(payload: dict) -> None:
    payload["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def connect() -> paramiko.SSHClient:
    host = os.environ.get("LOWEYE_GPU_HOST", "connect.bjb2.seetacloud.com")
    port = int(os.environ.get("LOWEYE_GPU_PORT", "26387"))
    password = os.environ.get("LOWEYE_GPU_PASS")
    if not password:
        raise SystemExit("缺少环境变量 LOWEYE_GPU_PASS（GPU 机器密码）")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username="root",
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str) -> str:
    _, stdout, _ = client.exec_command(command, timeout=180)
    return stdout.read().decode("utf-8", "replace")


def probe(client: paramiko.SSHClient) -> dict:
    phases: dict[str, str] = {}
    for task in TASKS:
        raw = run(client, f"grep -o '\"phase\": \"[^\"]*\"' {LOGS}/{task}_pipeline_status.json 2>/dev/null")
        chunks = raw.strip().split('"')
        phases[task] = chunks[-2] if len(chunks) >= 2 else "unknown"
    return {"phases": phases}


def main() -> int:
    deadline = datetime.now() + timedelta(hours=DEADLINE_HOURS)
    while datetime.now() < deadline:
        client = None
        try:
            client = connect()
            state = probe(client)
            if all(state["phases"][task] == "complete" for task in TASKS):
                log({"action": "exit", "reason": "both tasks complete", **state})
                return 0
            # 是否已有活着的协调器，交给启动脚本自己判断（它用 pidfile + 进程存活把守）。
            # 不要在这里用 pgrep 猜：exec_command 的包装 shell 命令行里就含脚本名，会多算一个。
            out = run(client, "bash /root/loweye/tools/gpu/06_launch_pipeline.sh")
            tail = out.strip()[-300:]
            if "already running" in out:
                log({"action": "wait", "reason": "launcher guard: coordinator alive", "out": tail, **state})
            elif "pipeline_pid=" in out:
                log({"action": "relaunch", "out": tail, **state})
            else:
                log({"action": "relaunch_failed", "out": tail, **state})
        except Exception as exc:
            log({"action": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if client is not None:
                client.close()
        time.sleep(POLL_SECONDS)
    log({"action": "exit", "reason": "deadline reached"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
