"""检索自测：直接看某一句话能召回哪些条款、每一路打分各是多少。

用法：
    .venv\\Scripts\\python.exe -m loweye.cli.search "绝缘子自爆怎么处理"
    .venv\\Scripts\\python.exe -m loweye.cli.search "绝缘子自爆怎么处理" --top-k 3 --json
"""

from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..retrieval import KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser(description="低空之眼知识库检索自测")
    parser.add_argument("query", help="查询语句")
    parser.add_argument("--config", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kb = KnowledgeBase.load(cfg.path("index_dir"), cfg)
    resp = kb.search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(resp.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    print("=" * 68)
    print(f"查询：{resp.query}")
    print(f"是否覆盖：{'是' if resp.covered else '否（低于拒答阈值，系统应回答「知识库未覆盖」）'}"
          f"  最高分={resp.top_score:.4f}  耗时={resp.elapsed_ms:.1f}ms")
    print("=" * 68)
    for hit in resp.hits:
        c = hit.clause
        print(f"[{hit.rank}] rerank={hit.rerank:.4f} rrf={hit.rrf:.4f} bm25={hit.bm25:.4f} vec={hit.vector:.4f}")
        print(f"    {c.citation_label}  （{c.hierarchy_path}）")
        print(f"    {c.text[:110]}{'…' if len(c.text) > 110 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
