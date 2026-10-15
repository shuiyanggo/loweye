"""在租来的 GPU 机器上执行训练：连接、探测、上传、跑、取回结果。

密码不写进仓库：从环境变量 `LOWEYE_GPU_PASS` 读，主机与端口走命令行参数。
本机没有 sshpass，OpenSSH 也没法在后台任务里交互输密码，所以用 paramiko。

用法：
    $env:LOWEYE_GPU_PASS='...'
    python -m tools.remote_gpu probe      --host connect.<区域>.seetacloud.com --port <端口>
    python -m tools.remote_gpu upload     --host ... --port ... --local X --remote Y
    python -m tools.remote_gpu run        --host ... --port ... --cmd "..."
    python -m tools.remote_gpu download   --host ... --port ... --remote X --local Y
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
USER = "root"


def connect(host: str, port: int, *, timeout: float = 30.0):
    import paramiko

    password = os.environ.get("LOWEYE_GPU_PASS")
    if not password:
        raise SystemExit("缺少环境变量 LOWEYE_GPU_PASS（GPU 机器密码）")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=USER, password=password,
                   timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
    return client


def run(client, cmd: str, *, timeout: float = 1800.0, stream: bool = True) -> tuple[int, str]:
    """跑一条远端命令。

    stderr 单独用线程抽干：paramiko 里 stderr 是另一个流，只读 stdout 时远端一旦往
    stderr 写多了就会把通道窗口写满、整条命令挂住（第一次 probe 就是这么超时的）。
    """
    import threading

    import codecs

    _, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    chunks: list[str] = []
    err_lines: list[str] = []

    def pump(recv, sink: list[str], prefix: str) -> None:
        # Decode the byte stream incrementally. paramiko's own readline decodes
        # each buffered chunk as a whole, so a buffer boundary that lands inside
        # a multi-byte character raises UnicodeDecodeError and kills the call --
        # which is exactly what happened while streaming a training progress bar.
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            data = recv(65536)
            if not data:
                break
            text = decoder.decode(data)
            if not text:
                continue
            sink.append(text)
            if stream:
                sys.stdout.write(prefix + text)
                sys.stdout.flush()
        tail = decoder.decode(b"", True)
        if tail:
            sink.append(tail)
            if stream:
                sys.stdout.write(prefix + tail)
                sys.stdout.flush()

    thread = threading.Thread(
        target=pump, args=(stderr.channel.recv_stderr, err_lines, "[err] "), daemon=True
    )
    thread.start()
    try:
        pump(stdout.channel.recv, chunks, "")
    finally:
        thread.join(timeout=5.0)
    code = stdout.channel.recv_exit_status()
    if err_lines:
        chunks.append("\n[stderr]\n" + "".join(err_lines))
    return code, "".join(chunks)


def sftp_put(client, local: Path, remote: str) -> None:
    sftp = client.open_sftp()
    try:
        sftp.put(str(local), remote)
    finally:
        sftp.close()


def sftp_get(client, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote, str(local))
    finally:
        sftp.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("probe", "upload", "push", "download", "run", "mkdir", "md5"))
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cmd", default=None)
    ap.add_argument("--script", default=None, help="本地 shell 脚本，上传后远端执行（省掉引号地狱）")
    ap.add_argument("--local", default=None)
    ap.add_argument("--remote", default=None)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    client = connect(args.host, args.port)
    try:
        if args.action == "probe":
            steps = (
                ("系统", "uname -sr; nproc; free -g | sed -n '1,2p'"),
                ("GPU", "nvidia-smi --query-gpu=name,memory.total,driver_version "
                        "--format=csv,noheader 2>/dev/null || echo 'no nvidia-smi'"),
                ("python", "which python python3 2>/dev/null; python3 -V"),
                ("torch", "timeout 150 python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' "
                          "2>&1 | tail -3"),
                ("磁盘", "df -h / /root /root/autodl-tmp 2>/dev/null | tail -4"),
                ("网络", "timeout 12 curl -s -o /dev/null -w 'pypi:%{http_code} ' https://pypi.org/simple/; "
                        "timeout 12 curl -s -o /dev/null -w 'hf-mirror:%{http_code} ' https://hf-mirror.com; "
                        "timeout 12 curl -s -o /dev/null -w 'github:%{http_code}\\n' https://github.com"),
                ("已装包", "pip list 2>/dev/null | grep -iE 'torch|onnx|mmdet|yolox|mmcv|opencv|numpy|pillow' | head -20"),
            )
            for name, cmd in steps:
                print(f"--- {name} ---", flush=True)
                try:
                    run(client, cmd, timeout=200)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [该步超时/失败] {type(exc).__name__}: {exc}", flush=True)
            return 0

        if args.action == "run":
            if args.script:
                local = Path(args.script)
                text = local.read_text(encoding="utf-8").replace("\r\n", "\n")
                remote = "/root/_dsh_cmd.sh"
                sftp = client.open_sftp()
                try:
                    with sftp.open(remote, "w") as handle:
                        handle.write(text)
                finally:
                    sftp.close()
                print(f"== 远端执行 {local.name}（{len(text)} 字符）==", flush=True)
                code, _ = run(client, f"bash {remote}", timeout=args.timeout)
                return code
            if not args.cmd:
                raise SystemExit("run 需要 --cmd 或 --script")
            code, _ = run(client, args.cmd, timeout=args.timeout)
            return code

        if args.action == "mkdir":
            if not args.remote:
                raise SystemExit("mkdir 需要 --remote")
            run(client, f"mkdir -p {args.remote}", timeout=60)
            return 0

        if args.action == "upload":
            local = Path(args.local)
            if local.is_dir():
                sftp = client.open_sftp()
                try:
                    for path in sorted(local.rglob("*")):
                        if path.is_file():
                            rel = path.relative_to(local).as_posix()
                            target = f"{args.remote.rstrip('/')}/{rel}"
                            parent = target.rsplit("/", 1)[0]
                            try:
                                sftp.stat(parent)
                            except IOError:
                                run(client, f"mkdir -p {parent}", timeout=60)
                            sftp.put(str(path), target)
                            print(f"  ↑ {rel}（{path.stat().st_size / 1024:.0f} KB）")
                finally:
                    sftp.close()
            else:
                run(client, f"mkdir -p {args.remote.rsplit('/', 1)[0]}", timeout=60)
                sftp_put(client, local, args.remote)
                print(f"  ↑ {local.name} → {args.remote}")
            return 0

        if args.action == "push":
            # 可续传的大文件上传：sftp.put 在网络抖动时会「不报错但少写」，
            # 实测 419 MB 的 tar 静默少了 26 MB，所以这里自己分块追加并校验字节数。
            local = Path(args.local)
            remote = args.remote
            total = local.stat().st_size
            sftp = client.open_sftp()
            try:
                try:
                    have = sftp.stat(remote).st_size
                except IOError:
                    have = 0
                if have > total:
                    print(f"  远端比本地大（{have} > {total}），删掉重来", flush=True)
                    run(client, f"rm -f {remote}", timeout=60)
                    have = 0
                if have == total:
                    print(f"  远端已是完整文件（{total} 字节），跳过", flush=True)
                    return 0
                print(f"  续传：本地 {total} 字节，远端 {have}，从 {have} 开始", flush=True)
                chunk = 1 << 20
                sent = have
                with local.open("rb") as src, sftp.open(remote, "ab") as dst:
                    src.seek(have)
                    while True:
                        data = src.read(chunk)
                        if not data:
                            break
                        dst.write(data)
                        sent += len(data)
                        if (sent - have) % (16 << 20) < chunk:
                            print(f"    {sent}/{total}（{sent * 100 // total}%）", flush=True)
                size = sftp.stat(remote).st_size
                print(f"  完成：远端 {size}/{total} 字节", flush=True)
                return 0 if size == total else 1
            finally:
                sftp.close()

        if args.action == "md5":
            code, _ = run(client, f"md5sum {args.remote}", timeout=600)
            return code

        if args.action == "download":
            sftp_get(client, args.remote, Path(args.local))
            print(f"  ↓ {args.remote} → {args.local}")
            return 0
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
