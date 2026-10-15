"""本机侧的远端训练进度查看器（只读）。

用 paramiko 连已授权的 AutoDL 实例，取回两个任务的阶段文件、逐轮指标与非有限跳过数，
**在本机解析**（远端不需要任何解释器：非交互 shell 的 PATH 里没有 python3，实测
`bash: line 1: python3: command not found`），并可选写出一行 JSON 便于逐轮对比。
不修改远端任何文件。

凭据与环境变量（与 tools/remote_gpu.py 一致，密码不写进仓库，必须由环境变量提供）：
    LOWEYE_GPU_PASS（必给）、LOWEYE_GPU_HOST / LOWEYE_GPU_PORT（可给，默认本会话授权的实例）

用法:
    python -m tools.check_remote_training
    python -m tools.check_remote_training --json out/inspection_gpu/_progress.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.remote_gpu import connect, run as gpu_run  # noqa: E402 - 复用既有的连接与取流实现

TASKS = ("crop_pest", "fire_smoke")
REMOTE = "/root/autodl-tmp/loweye"
LOGS = "/root/loweye/logs"
# 本会话实例作为兜底默认，但环境变量优先：换实例时只改环境变量，不必改代码。
DEFAULT_HOST = os.environ.get("LOWEYE_GPU_HOST", "connect.bjb2.seetacloud.com")
DEFAULT_PORT = int(os.environ.get("LOWEYE_GPU_PORT", "26387"))

LIVE_CMD = (
    "date +'%F %T'; uptime | tr -s ' ' | cut -d, -f1; "
    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
    "echo trainer_procs=$(pgrep -fc train_inspection_fcos.py); "
    "echo coordinator_pid=$(cat /root/loweye/logs/inspection_pipeline.pid 2>/dev/null); "
    "echo remote_python=$(command -v python3 || echo none); "
    "echo conda_python=$(ls /root/miniconda3/bin/python 2>/dev/null || echo none)"
)


def run(client, cmd: str, timeout: float = 180.0) -> str:
    """跑一条远端命令，返回合并后的文本。

    复用 tools/remote_gpu.run：它用独立线程抽干 stderr，否则远端往 stderr 写满通道窗口
    会把命令挂死（第一次 probe 就是那么超时的）。
    """
    code, text = gpu_run(client, cmd, timeout=timeout, stream=False)
    text = text.strip()
    return text if code == 0 else (text + "\n[exit %d]" % code)


def read_file(client, path: str) -> str:
    """取回一个远端小文件；不存在时返回空串。"""
    return run(client, "cat %s 2>/dev/null" % path)


def count_lines(client, path: str) -> int:
    """用 wc -l 数行数。

    不要先 head -c 截断再数行——逐轮记录文件已经 263 KB，截到 200 KB 会把第 8–10 轮
    当成不存在（本脚本第一版就是这样误报 rows=7、末轮=epoch 7 的）。
    """
    text = run(client, "wc -l < %s 2>/dev/null || echo 0" % path).strip().split()
    return int(text[0]) if text and text[0].isdigit() else 0


def parse_status(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001 - 半截 JSON 原样保留供人工核对
        return {"_unparsed": text[-400:], "_error": str(exc)}


def parse_epochs(text: str, tail: int = 3) -> tuple[int, list[dict]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001 - 半截行（head 截断）忽略
            continue
    return len(rows), rows[-tail:]


def summarize(task: str, status_text: str, epochs_text: str, epoch_rows: int | None = None) -> dict:
    status = parse_status(status_text) or {}
    count, rows = parse_epochs(epochs_text)
    last = rows[-1] if rows else {}
    command = status.get("command")
    parts = command if isinstance(command, list) else (command or "").split()
    limit = parts[parts.index("--nonfinite-limit") + 1] if "--nonfinite-limit" in parts else None
    return {
        "phase": status.get("phase"),
        "updated_at": status.get("updated_at") or status.get("updated"),
        "nonfinite_limit": limit,
        "epoch_rows": epoch_rows if epoch_rows is not None else count,
        "last_epochs": [
            {
                "epoch": r.get("epoch"),
                "loss": round(r["loss"], 4) if isinstance(r.get("loss"), (int, float)) else r.get("loss"),
                "map": r.get("map"),
                "map50": r.get("map50"),
                "nonfinite_batches": r.get("nonfinite_batches"),
                "epoch_seconds": r.get("epoch_seconds"),
            }
            for r in rows
        ],
        "status_raw": status if not status.get("phase") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="只读查看远端两个任务的训练进度")
    parser.add_argument("--json", default=None, help="把结果写到该文件（一行 JSON）")
    parser.add_argument("--host", default=None, help="默认取 LOWEYE_GPU_HOST，再退到本会话授权的实例")
    parser.add_argument("--port", type=int, default=None, help="默认取 LOWEYE_GPU_PORT")
    args = parser.parse_args()

    host = args.host or DEFAULT_HOST
    port = int(args.port or DEFAULT_PORT)
    client = connect(host, port)
    try:
        live = run(client, LIVE_CMD)
        report = {"live": live.splitlines(), "tasks": {}, "events": {}}
        for task in TASKS:
            status_text = read_file(client, "%s/%s_pipeline_status.json" % (LOGS, task))
            epochs_path = "%s/%s/%s_epochs.jsonl" % (REMOTE, task, task)
            rows = count_lines(client, epochs_path)
            epochs_text = run(client, "tail -n 4 %s 2>/dev/null" % epochs_path)
            report["tasks"][task] = summarize(task, status_text, epochs_text, rows)
        for task in TASKS:
            report["events"][task] = run(
                client,
                "tail -n 1 %s/train_%s.log 2>/dev/null | cut -c1-260; "
                "grep -o 'nonfinite_batches[^,}]*' %s/train_%s.log 2>/dev/null | tail -1"
                % (LOGS, task, LOGS, task),
            )
    finally:
        client.close()

    print("--- 现场")
    for line in report["live"]:
        print("   ", line)
    print("--- 任务")
    for task, item in report["tasks"].items():
        print("%-10s phase=%-30s rows=%-3d limit=%s" % (task, item["phase"], item["epoch_rows"], item["nonfinite_limit"]))
        for row in item["last_epochs"]:
            print(
                "           epoch=%-3s loss=%-9s map=%-9s map50=%-9s nonfinite=%-4s %ss"
                % (row["epoch"], row["loss"], row["map"], row["map50"], row["nonfinite_batches"], row["epoch_seconds"])
            )
        if item["status_raw"]:
            print("           状态文件未按预期解析:", json.dumps(item["status_raw"], ensure_ascii=False)[:300])
    print("--- 最新事件")
    for task, text in report["events"].items():
        print("   == %s" % task)
        print("   " + text.replace("\n", "\n   "))

    if args.json:
        dest = Path(args.json)
        if not dest.is_absolute():
            dest = PROJECT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        print("已写出:", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
