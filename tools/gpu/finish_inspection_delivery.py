"""Recover-then-evaluate-then-report: the last mile of the AutoDL delivery.

The paid host produces best/last/ONNX plus a trainer test receipt; monitor
recovery already pulls those down with SHA256 verification. Two things still have
to happen, and both are easy to get wrong by hand:

1. the *independent* ONNX evaluation must run where pycocotools exists (the host,
   because the local `.venv` has onnxruntime but not pycocotools), and the report
   it writes must be provably about the same ONNX bytes that were recovered;
2. the Markdown report must be regenerated from the recovered artifacts, not from
   numbers copied out of a log.

So this script:
  * refuses to run until the selected tasks have a recovery receipt;
  * runs the host-side evaluator for any task whose local ONNX report is missing or
    whose recorded ONNX SHA256 does not match the recovered artifact;
  * asserts the SHA256 chain (recovery receipt == ONNX report) and fails loudly if
    it does not hold;
  * then invokes tools/write_model_report.py.

It is idempotent: a task whose local ONNX report already carries the recovered
digest is skipped. The remote path/limit arguments exist so this exact code path can
be rehearsed against small probe artifacts before the real ones exist.

Environment: `LOWEYE_GPU_PASS` must hold the host password (never stored in the repo).

Usage:
    python -m tools.gpu.finish_inspection_delivery --host H --port P
    python -m tools.gpu.finish_inspection_delivery --host H --port P --only crop_pest
    # rehearsal against probe artifacts:
    python -m tools.gpu.finish_inspection_delivery --host H --port P --only fire_smoke \
        --out-dir out/inspection_gpu/_dryrun --limit 8 \
        --remote-onnx /root/autodl-tmp/loweye/models_probe/fire_smoke_fcos_640.onnx \
        --remote-manifest /root/loweye/logs/fire_smoke_manifest_probe2.json \
        --remote-report /root/loweye/logs/_dryrun_fire_smoke_onnx_eval.json --skip-report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT / "out" / "inspection_gpu"
# Two different trees on the host, both verified on the bjb2 instance (2026-10-14):
#   HOST_ROOT  /root/loweye            the coordinator's ROOT (run_inspection_tasks.py:95):
#                                      it writes data/{task}_manifest.json and logs/*.log here
#   HOST_BIG   /root/autodl-tmp/loweye checkpoints, data zips and the trained models/
# A single REMOTE constant used to point the manifest at /root/autodl-tmp/loweye/data/,
# which does not exist on the instance: the default-path evaluation would have died with
# "missing manifest: ...". The rehearsal with --remote-manifest hid it; this is why the
# default path is what gets rehearsed.
HOST_ROOT = "/root/loweye"
HOST_BIG = "/root/autodl-tmp/loweye"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
EVALUATOR = "/root/loweye/tools/gpu/evaluate_onnx_detector.py"
TASKS = (
    ("fire_smoke", "dfire", 640),
    ("crop_pest", "ip102", 800),
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Remote:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = str(port)
        if not os.environ.get("LOWEYE_GPU_PASS"):
            raise SystemExit("missing LOWEYE_GPU_PASS (the GPU host password)")

    def call(self, *arguments: str) -> int:
        command = [sys.executable, "-m", "tools.remote_gpu", *arguments,
                   "--host", self.host, "--port", self.port]
        return subprocess.call(command, cwd=str(PROJECT))

    def run_script(self, body: str) -> int:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "finish_step.sh"
            script.write_text(body, encoding="utf-8")
            return self.call("run", "--script", str(script))

    def download(self, remote: str, local: Path) -> int:
        local.parent.mkdir(parents=True, exist_ok=True)
        return self.call("download", "--remote", remote, "--local", str(local))


def recovered_digest(out_dir: Path, task: str) -> str | None:
    """ONNX SHA256 recorded by the recovery receipt for this task."""
    receipt = out_dir / f"{task}_recovery.json"
    if not receipt.is_file():
        return None
    payload = load_json(receipt)
    if isinstance(payload, dict):
        # monitor_inspection writes a list; tolerate a single corrected record rather
        # than dying on someone's hand-edited receipt.
        payload = [payload]
    for entry in payload:
        if isinstance(entry, dict) and str(entry.get("remote", "")).endswith(".onnx"):
            return entry.get("sha256")
    return None


def evaluate_body(*, onnx: str, manifest: str, group_safe: str | None, report: str,
                  limit: int, threads: int) -> str:
    limit_flag = f" --limit {limit}" if limit else ""
    choose = f'if [ -f "{group_safe}" ]; then MAN="{group_safe}"; fi\n' if group_safe else ""
    # --check-preprocessing compares the evaluator's own preprocessing against the
    # trainer's InspectionDataset elementwise. tools/write_model_report.py renders that
    # parity result as a row of the delivered document, so leaving the flag out used to
    # put an avoidable "file missing" gap in the evidence table. 8 samples cost seconds.
    return f"""set -u
MAN="{manifest}"
{choose}if [ ! -f "{onnx}" ]; then echo "missing onnx: {onnx}"; exit 4; fi
if [ ! -f "$MAN" ]; then echo "missing manifest: $MAN"; exit 4; fi
echo "onnx={onnx}"
echo "manifest=$MAN"
{REMOTE_PYTHON} {EVALUATOR} --onnx "{onnx}" --manifest "$MAN" --split test \\
    --out "{report}" --threads {threads}{limit_flag} --check-preprocessing 8 --progress-every 500
echo "eval_exit=$?"
sha256sum "{onnx}"
test -f "{report}" || exit 5
"""


def resolve_remote_paths(arguments, task: str, size: int) -> tuple[str, str, str | None, str]:
    onnx = arguments.remote_onnx or f"{HOST_BIG}/models/{task}_fcos_{size}.onnx"
    report = arguments.remote_report or f"{HOST_ROOT}/logs/{task}_onnx_eval.json"
    if arguments.remote_manifest:
        return onnx, arguments.remote_manifest, None, report
    # The coordinator swaps in the group-safe manifest whenever the source-group
    # audit found cross-split leakage, so the evaluation has to follow the same rule.
    return (onnx, f"{HOST_ROOT}/data/{task}_manifest.json",
            f"{HOST_ROOT}/data/{task}_group_safe_manifest.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description="host-side final evaluation + report regeneration")
    parser.add_argument("--host", required=True,
                        help="GPU gateway host; required because the gateway changes between sessions")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--only", default=None, help="limit to one task")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N test images")
    parser.add_argument("--threads", type=int, default=16,
                        help="ONNX Runtime 线程数；实测 16 线程 0.31 秒/张、32 线程 0.91、96 线程 4.9（容器 CPU 配额远小于 nproc）")
    parser.add_argument("--remote-onnx", default=None, help="override the ONNX path (rehearsal)")
    parser.add_argument("--remote-manifest", default=None, help="override the manifest path (rehearsal)")
    parser.add_argument("--remote-report", default=None, help="override the report path (rehearsal)")
    parser.add_argument("--skip-report", action="store_true", help="evaluate only, do not regenerate")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir

    selected = [item for item in TASKS if arguments.only in (None, item[0])]
    if not selected:
        raise SystemExit(f"unknown task: {arguments.only}")

    expected, pending = {}, []
    for task, _dataset, _size in selected:
        digest = recovered_digest(out_dir, task)
        if digest is None:
            pending.append(task)
        expected[task] = digest
    if pending:
        print(f"not recovered yet, nothing to finish: {', '.join(pending)}", flush=True)
        print("wait for tools/gpu/monitor_inspection.py to write the recovery receipt", flush=True)
        return 3

    remote = Remote(arguments.host, arguments.port)
    exit_code = 0
    for task, _dataset, size in selected:
        report = out_dir / f"{task}_onnx_eval.json"
        wanted = expected[task]
        if report.is_file():
            recorded = load_json(report).get("artifact", {}).get("onnx_sha256")
            if recorded == wanted:
                print(f"{task}: local ONNX report already matches {wanted[:12]}… — skipped", flush=True)
                continue
            print(f"{task}: local report digest {str(recorded)[:12]}… != recovered "
                  f"{wanted[:12]}… — re-evaluating", flush=True)

        onnx, manifest, group_safe, remote_report = resolve_remote_paths(arguments, task, size)
        print(f"{task}: running the independent ONNX evaluation on the host", flush=True)
        body = evaluate_body(onnx=onnx, manifest=manifest, group_safe=group_safe, report=remote_report,
                             limit=arguments.limit, threads=arguments.threads)
        if remote.run_script(body) != 0:
            print(f"{task}: could not run the host evaluation", flush=True)
            exit_code = 1
            continue
        if remote.download(remote_report, report) != 0 or not report.is_file():
            print(f"{task}: could not download {remote_report}", flush=True)
            exit_code = 1
            continue
        recorded = load_json(report).get("artifact", {}).get("onnx_sha256")
        if recorded != wanted:
            print(f"{task}: SHA256 CHAIN BROKEN — evaluation report says {recorded}, "
                  f"recovery receipt says {wanted}", flush=True)
            exit_code = 1
            continue
        payload = load_json(report)
        print(f"{task}: independent eval ok — {payload.get('protocol', {}).get('images')} images, "
              f"map {payload.get('coco', {}).get('map')}, map50 {payload.get('coco', {}).get('map50')}, "
              f"onnx {str(recorded)[:12]}… (matches the recovered artifact)", flush=True)

    if exit_code or arguments.skip_report:
        return exit_code

    print("regenerating the Markdown report from the recovered artifacts", flush=True)
    return subprocess.call([sys.executable, "-m", "tools.write_model_report"], cwd=str(PROJECT))


if __name__ == "__main__":
    sys.exit(main())
