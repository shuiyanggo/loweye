"""临时自检：检查方案书三章的 Markdown 表格列数是否一致。

为什么写这个脚本：方案书里的表格是手写的，列数不一致时 Markdown 会静默渲染错位，
在 PDF 里很难一眼看出来。这个脚本按表格块分组，报告每块的列数分布。

用法：
    python docs/proposal/_check_tables.py 01_overview.md 08_team.md 09_risk.md
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def split_row(line: str) -> list[str]:
    """按未转义的竖线切分，去掉首尾空单元格。"""
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in line.strip():
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "|":
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    if cells and cells[0] == "":
        cells.pop(0)
    if cells and cells[-1] == "":
        cells.pop()
    return cells


def check(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[int, list[tuple[int, int]]]] = []
    current: list[tuple[int, int]] = []
    start = 0

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
            if not current:
                start = idx
            current.append((idx, len(split_row(line))))
        else:
            if current:
                blocks.append((start, current))
                current = []
    if current:
        blocks.append((start, current))

    problems = 0
    for start, rows in blocks:
        widths = {w for _, w in rows}
        total = len(rows)
        status = "OK" if len(widths) == 1 else "列数不一致"
        if len(widths) != 1:
            problems += 1
        print(f"[{status}] line {start} 起，{total} 行，列数 {sorted(widths)}")
        if len(widths) != 1:
            for ln, w in rows:
                print(f"    line {ln}: {w} 列 | {lines[ln - 1][:70]}")
    print(f"--- {path.name}: 表格块 {len(blocks)} 个，异常 {problems} 个")
    return problems


def main() -> int:
    names = sys.argv[1:]
    if not names:
        names = ["01_overview.md", "08_team.md", "09_risk.md"]
    bad = 0
    for name in names:
        bad += check(HERE / name)
    print(f"=== 合计异常 {bad} 个")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
