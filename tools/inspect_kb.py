"""看一眼知识库切成了什么：逐条打印 chunk_id / 条款号 / 层级路径 / 正文开头。

用途：
- 改完切分逻辑后肉眼核对（比只看总数可靠）；
- 写金标问答集时确认条款号的确切写法；
- 答辩前自证「切分是按章-节-条语义做的」，不是按固定字数硬切。

用法：
    .venv\\Scripts\\python.exe tools\\inspect_kb.py
    .venv\\Scripts\\python.exe tools\\inspect_kb.py --doc demo_line_guide
    .venv\\Scripts\\python.exe tools\\inspect_kb.py --no-clause-only   # 连无编号块一起看
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loweye.config import load_config  # noqa: E402
from loweye.text import is_numbered_clause  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="知识库切分结果检视")
    ap.add_argument("--config", default=None)
    ap.add_argument("--doc", default=None, help="只看某篇文档（doc_id）")
    ap.add_argument("--clauses-only", action="store_true", default=False, help="只打印带条款号的块")
    ap.add_argument("--no-clause-only", dest="clauses_only", action="store_false")
    ap.add_argument("--width", type=int, default=58)
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = cfg.path("data_processed") / "clauses.jsonl"
    if not path.exists():
        print(f"找不到 {path}，先跑 build_kb", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.doc:
        rows = [r for r in rows if r.get("doc_id") == args.doc]

    numbered = sum(1 for r in rows if is_numbered_clause(r.get("clause_no", "")))
    print(f"来源：{path}")
    print(f"共 {len(rows)} 块，带条款号 {numbered} 块（{numbered / max(len(rows), 1) * 100:.1f}%）")
    print("-" * 110)
    for r in rows:
        ok = is_numbered_clause(r.get("clause_no", ""))
        if args.clauses_only and not ok:
            continue
        text = " ".join(str(r.get("text", "")).split())[: args.width]
        mark = "OK " if ok else "-- "
        print(f"{mark}{r['chunk_id']:<28} | {str(r.get('clause_no'))[:10]:<10} | "
              f"{str(r.get('hierarchy_path'))[:24]:<24} | {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
