#!/usr/bin/env python3
"""未改动证明：新增检测模型是否碰过既有绝缘子交付物。

out/insulator_baseline.json 记录 models/detector/ 下既有产物在扩展训练开始前的
字节数与 SHA-256。本脚本对交付验收做「未改动 / 未缺失」的确定性核对，并列出本次
新增的文件（允许新增，不允许改动或缺失）。

不需要 GPU，不联网，不写除 --json-out 之外的任何文件。

用法：
    python -m tools.verify_insulator_baseline
    python -m tools.verify_insulator_baseline --baseline out/insulator_baseline.json --dir models/detector
    python -m tools.verify_insulator_baseline --json-out out/inspection_gpu/insulator_baseline_check.json

退出码：0 全部一致（允许新增文件）；3 有改动或缺失；2 用法或基线文件问题。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT / "out" / "insulator_baseline.json"
DEFAULT_DIR = PROJECT / "models" / "detector"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IGNORED_NAMES = {".gitkeep", "Thumbs.db"}


class BaselineError(Exception):
    """基线文件缺失、不是合法 JSON，或结构不符合约定。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline(path: Path) -> dict:
    """读取并校验基线结构；任何不符合约定处都抛 BaselineError（调用方转成退出码 2）。"""
    if not path.exists():
        raise BaselineError(f"baseline file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"baseline is not valid JSON: {path} ({error})") from error
    if not isinstance(payload, dict):
        raise BaselineError("baseline root must be an object")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise BaselineError("baseline needs a nonempty files list")
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise BaselineError(f"files[{index}] must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise BaselineError(f"files[{index}].name must be a plain file name")
        if not SHA256_RE.match(str(entry.get("sha256", "")).lower()):
            raise BaselineError(f"files[{index}].sha256 must be 64 hex characters")
        if "bytes" in entry and not isinstance(entry["bytes"], int):
            raise BaselineError(f"files[{index}].bytes must be an integer when present")
    return payload


def compare(payload: dict, directory: Path) -> dict:
    """逐条目核对字节与 SHA-256，并把目录里多出来的文件列为允许的新增。"""
    rows: list[dict] = []
    recorded: set[str] = set()
    for entry in payload["files"]:
        name = entry["name"]
        recorded.add(name)
        expected_sha = str(entry["sha256"]).lower()
        row = {
            "name": name,
            "expected_bytes": entry.get("bytes"),
            "expected_sha256": expected_sha,
            "actual_bytes": None,
            "actual_sha256": None,
        }
        target = directory / name
        if not target.exists():
            row["status"] = "missing"
        else:
            size = target.stat().st_size
            digest = sha256_file(target)
            row["actual_bytes"] = size
            row["actual_sha256"] = digest
            same = digest == expected_sha
            if isinstance(entry.get("bytes"), int):
                same = same and size == entry["bytes"]
            row["status"] = "unchanged" if same else "modified"
        rows.append(row)
    new_files: list[dict] = []
    if directory.exists():
        for item in sorted(directory.iterdir()):
            if item.is_file() and item.name not in recorded and item.name not in IGNORED_NAMES:
                new_files.append({"name": item.name, "bytes": item.stat().st_size})
    changed = [row for row in rows if row["status"] != "unchanged"]
    return {
        "directory": str(directory),
        "baseline_entries": len(rows),
        "unchanged": len(rows) - len(changed),
        "rows": rows,
        "changed": changed,
        "new_files": new_files,
        "ok": not changed,
    }


def render(result: dict) -> list[str]:
    lines = [f"directory: {result['directory']}"]
    for row in result["rows"]:
        actual = (
            f"{row['actual_bytes']} bytes / {str(row['actual_sha256'])[:16]}..."
            if row["actual_sha256"]
            else "absent"
        )
        lines.append(
            f"  {row['status']:<9} {row['name']}"
            f"  expected {row['expected_bytes']} bytes / {row['expected_sha256'][:16]}..."
            f"  actual {actual}"
        )
    for item in result["new_files"]:
        lines.append(f"  new       {item['name']}  {item['bytes']} bytes (allowed)")
    lines.append(
        f"unchanged {result['unchanged']}/{result['baseline_entries']} recorded files;"
        f" changed {len(result['changed'])}; new {len(result['new_files'])}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="证明既有绝缘子交付物未被改动、未被替换")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="基线 JSON 路径")
    parser.add_argument("--dir", dest="directory", type=Path, default=DEFAULT_DIR, help="待核对的模型目录")
    parser.add_argument("--json-out", type=Path, default=None, help="可选：把核对结果写成 JSON")
    args = parser.parse_args(argv)
    try:
        payload = load_baseline(args.baseline)
    except BaselineError as error:
        print(f"baseline error: {error}")
        return 2
    result = compare(payload, args.directory)
    print(f"baseline: {args.baseline}")
    for line in render(result):
        print(line)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json: {args.json_out}")
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
