"""构建知识库：data/raw 文档 → 条款切分 → 索引落盘。

用法：
    .venv\\Scripts\\python.exe -m loweye.cli.build_kb
    .venv\\Scripts\\python.exe -m loweye.cli.build_kb --json
"""

from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..parsing.builder import build_clauses, write_clauses_jsonl
from ..retrieval import KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser(description="构建低空之眼知识库索引")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出构建报告")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()

    clauses, report = build_clauses(cfg)
    processed = cfg.path("data_processed") / "clauses.jsonl"
    write_clauses_jsonl(clauses, processed)

    kb = KnowledgeBase.from_clauses(clauses, cfg)
    index_dir = kb.save(cfg.path("index_dir"))

    payload = {
        "docs": report.docs,
        "chunks": report.chunks,
        "numbered_chunks": report.numbered,
        "numbered_ratio": round(report.numbered_ratio, 4),
        "per_doc": report.per_doc,
        "index_dir": str(index_dir),
        "clauses_jsonl": str(processed),
        "stats": kb.stats().to_dict(),
        "degradations": kb.degradations,
        "warnings": report.warnings,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 68)
        print("低空之眼 · 知识库构建完成")
        print("=" * 68)
        print(f"文档数        : {report.docs}")
        print(f"条款块数      : {report.chunks}（带条款号 {report.numbered}，占比 {report.numbered_ratio:.1%}）")
        for doc_id, n in report.per_doc.items():
            print(f"  - {doc_id}: {n} 块")
        print(f"BM25 后端     : {kb.stats().bm25_backend}")
        print(f"向量后端      : {kb.stats().embedder}（dim={kb.stats().dim}）")
        print(f"重排后端      : {kb.stats().reranker}")
        print(f"索引目录      : {index_dir}")
        print(f"条款 JSONL    : {processed}")
        if report.warnings:
            print("警告：")
            for w in report.warnings:
                print(f"  ! {w}")
        if kb.degradations:
            print("降级项：")
            for d in kb.degradations:
                print(f"  - {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
