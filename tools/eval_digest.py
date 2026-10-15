"""把 out/eval_report.json 里关心的几组数字打成一张小表。

用途：写报告与答辩口径时对照，避免每次翻 5000 行 JSON。
用法：.venv\\Scripts\\python.exe tools\\eval_digest.py [--json out/eval_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = [
    "bm25_only",
    "vector_only",
    "hybrid_rrf",
    "hybrid_rerank",
    "hybrid_rerank_no_lexicon",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/eval_report.json")
    args = ap.parse_args()

    path = Path(args.json)
    if not path.is_absolute():
        path = ROOT / path
    data = json.loads(path.read_text(encoding="utf-8"))
    ret = data["retrieval"]

    print(f"知识库 {data['kb']['chunks']} 块 / {data['kb']['docs']} 文档  生成于 {data['created_at']}")
    print(f"min_relevance = {data['refusal']['min_relevance']}")
    print()
    header = f"{'分组':<12}{'n':>4}" + "".join(f"{c[:16]:>18}" for c in CONFIGS)
    print(header)
    print("-" * len(header))

    def row(label: str, block: dict) -> None:
        cells = []
        for cfg in CONFIGS:
            v = block.get(cfg) or {}
            cells.append(f"{v.get('recall@1', 0):.0%}/{v.get('mrr', 0):.3f}")
        print(f"{label:<12}{block.get('n_questions', 0):>4}" + "".join(f"{c:>18}" for c in cells))

    overall = {"n_questions": ret["n_questions"], **{c: ret[c] for c in CONFIGS if c in ret}}
    row("全部", overall)
    for group, block in sorted((ret.get("by_group") or {}).items()):
        row(group, block)
    print("（每格 = Recall@1 / MRR）")
    print()

    covered = ret.get("covered_rate", 0.0)
    print(f"域内 covered_rate = {covered:.0%}")
    print(
        f"越界拒答率 = {data['refusal']['refusal_rate']:.0%} "
        f"（漏判 {data['refusal']['false_accept'] or '无'}）"
    )
    print()

    print("未命中或误判的题（以 hybrid_rerank 为准）：")
    for r in data.get("retrieval_rows", []):
        rank = r.get("hybrid_rerank")
        if rank == 1 and r.get("covered"):
            continue
        flag = "误拒答" if rank == 1 else ("未命中" if rank is None else f"rank {rank}")
        if rank == 1 and not r.get("covered"):
            flag = "top1 正确但被拒答"
        print(
            f"  {r['id']:<6}{r['group']:<11}{flag:<18}"
            f"covered={r['covered']!s:<6}top1={r.get('top1','')} score={r.get('top_score', 0):.4f}"
        )
        print(f"        {r['question']}   期望 {r['expected']}")
    print()

    lat = data.get("latency", {})
    for key in ("search_ms", "analyze_ms"):
        if lat.get(key):
            b = lat[key]
            print(f"{key:<12} n={b['n']:<4} P50 {b['p50']} / P95 {b['p95']} / max {b['max']} ms")
    stages = data.get("stage_latency") or {}
    for stage, b in stages.items():
        print(f"  {stage:<16} P50 {b['p50']} / P95 {b['p95']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
