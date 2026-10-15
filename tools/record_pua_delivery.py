"""把一次「有可见 PUA 输出的交付」追加进 out/pua_feedback.jsonl。

为什么单独写个脚本，而不是命令行拼一行 JSON：
- 记录内容含中文与引号，走 PowerShell 传参会被壳层吃掉引号（本轮已经踩过一次）；
- Windows PowerShell 5.1 的 `Add-Content -Encoding UTF8` 会写 BOM，追加到 JSONL 中段会让第二行以
  BOM 开头，之后任何 `json.loads` 读这一行都会报错；
- 因此统一由 Python 以 `encoding="utf-8"`（无 BOM）追加，并保证文件以单个换行结尾、每行一个合法 JSON。

用法：
    python tools/record_pua_delivery.py --record-json out/_pua_record.json
    python tools/record_pua_delivery.py --record-json out/_pua_record.json --ledger out/pua_feedback.jsonl --print
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = PROJECT / "out" / "pua_feedback.jsonl"

REQUIRED_KEYS = ("ts", "user_instruction", "task_summary")


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{path}: 需要一条记录对象或非空数组")
    for item in payload:
        if not isinstance(item, dict):
            raise SystemExit(f"{path}: 数组元素必须是对象")
        missing = [key for key in REQUIRED_KEYS if not str(item.get(key, "")).strip()]
        if missing:
            raise SystemExit(f"{path}: 记录缺少字段 {missing}")
        if "rating" in item and item["rating"] is not None:
            raise SystemExit(f"{path}: 用户要求不记录评分，rating 必须为 null")
    return payload


def append_records(ledger: Path, records: list[dict]) -> dict:
    """把记录追加进台账，返回校验结果（行数、坏行、是否被写入 BOM）。

    上一次写入若恰好没有以换行结尾（手改、编辑器截断都可能），直接追加会把两条记录粘成一行，
    于是整行都不再是合法 JSON —— 这里先补一个换行再写。
    """
    ledger.parent.mkdir(parents=True, exist_ok=True)
    before = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
    lines_before = len([line for line in before.splitlines() if line.strip()])

    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        if before and not before.endswith("\n"):
            handle.write("\n")
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    after = ledger.read_text(encoding="utf-8")
    lines_after = len([line for line in after.splitlines() if line.strip()])
    # 每行都必须是合法 JSON，防止把坏行留在台账里
    bad = [
        index
        for index, line in enumerate(after.splitlines(), start=1)
        if line.strip() and not _is_json(line)
    ]
    return {"ledger": str(ledger), "lines_before": lines_before,
            "lines_after": lines_after, "bad_lines": bad,
            "bom": after.startswith("\ufeff")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-json", required=True, help="含一条记录或记录数组的 JSON 文件")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--print", dest="show", action="store_true", help="追加后回显最后一行")
    arguments = parser.parse_args()

    records = load_records(Path(arguments.record_json))
    summary = append_records(Path(arguments.ledger), records)
    print(json.dumps(summary, ensure_ascii=False))
    if arguments.show:
        print(Path(arguments.ledger).read_text(encoding="utf-8").splitlines()[-1])
    return 1 if summary["bad_lines"] else 0


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
    except json.JSONDecodeError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
