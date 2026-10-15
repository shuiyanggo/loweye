"""对比两份评测报告：词法基线 vs BGE 后端。

评测脚本每次只按当前配置出一份报告，要说明「换后端到底带来多少提升」，
就需要把两份 JSON 摆在一起算差。指标一律从 retrieval_rows 现场重算，
不直接抄报告里的小结数字——这样分组口径（字面 / 改写 / 留出）与全量口径
必然一致，也不会出现「README 抄了旧数字」这类不一致。

用法：
    python -m tools.eval_compare --a out/eval_report.json --b out/eval_report_bge.json \\
        --label-a 词法（hashing_tfidf + lexical） --label-b BGE（bge-m3 + bge-reranker-v2-m3） \\
        --out out/eval_compare.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

CONFIGS = [
    ("bm25_only", "仅 BM25"),
    ("vector_only", "仅向量"),
    ("hybrid_rrf", "RRF 融合（无重排）"),
    ("hybrid_rerank", "融合 + 重排（本系统）"),
    ("hybrid_rerank_no_lexicon", "融合 + 重排（关闭术语表）"),
]
GROUPS = [("literal", "字面题"), ("paraphrase", "改写题（开发集）"), ("heldout", "留出题")]


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def metrics(rows: list[dict], config: str) -> dict:
    """从行级排名重算 Recall@1/3/5 与 MRR。"""
    n = len(rows)
    if not n:
        return {"n": 0}
    hit1 = hit3 = hit5 = 0
    rr = 0.0
    for row in rows:
        rank = row.get(config)
        if rank is None:
            continue
        if rank <= 1:
            hit1 += 1
        if rank <= 3:
            hit3 += 1
        if rank <= 5:
            hit5 += 1
        rr += 1.0 / rank
    return {
        "n": n,
        "recall@1": hit1 / n,
        "recall@3": hit3 / n,
        "recall@5": hit5 / n,
        "mrr": rr / n,
    }


def by_group(report: dict, group: str) -> list[dict]:
    return [r for r in report["retrieval_rows"] if r.get("group") == group]


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def delta(a: float | None, b: float | None, *, digits: int = 3) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    if abs(d) < 0.0005:
        return "持平"
    return f"{d:+.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(PROJECT / "out" / "eval_report.json"))
    ap.add_argument("--b", default=str(PROJECT / "out" / "eval_report_bge.json"))
    ap.add_argument("--label-a", default="A 词法后端")
    ap.add_argument("--label-b", default="B BGE 后端")
    ap.add_argument("--out", default=str(PROJECT / "out" / "eval_compare.md"))
    args = ap.parse_args()

    ra, rb = load(Path(args.a)), load(Path(args.b))
    lines: list[str] = []
    add = lines.append

    add(f"# 评测对照：{args.label_a} vs {args.label_b}")
    add("")
    add(f"- A 报告：`{args.a}`（生成于 {ra.get('created_at', '未知')}）")
    add(f"- B 报告：`{args.b}`（生成于 {rb.get('created_at', '未知')}）")
    add("")

    add("## 1 后端口径")
    add("")
    add(f"| 项 | {args.label_a} | {args.label_b} |")
    add("| --- | --- | --- |")
    ka, kb_ = ra.get("kb", {}), rb.get("kb", {})
    for key, name in (("chunks", "条款块"), ("docs", "文档数"), ("bm25_backend", "BM25 后端"),
                      ("embedder", "向量后端"), ("dim", "向量维度"), ("reranker", "重排后端")):
        add(f"| {name} | {ka.get(key, '—')} | {kb_.get(key, '—')} |")
    add("")

    rows_a, rows_b = ra["retrieval_rows"], rb["retrieval_rows"]
    add(f"## 2 全量域内题（A n={len(rows_a)} / B n={len(rows_b)}）")
    add("")
    add(f"| 配置 | {args.label_a} R@1 / MRR | {args.label_b} R@1 / MRR | ΔR@1 | ΔMRR |")
    add("| --- | --- | --- | --- | --- |")
    for config, label in CONFIGS:
        ma, mb = metrics(rows_a, config), metrics(rows_b, config)
        add(f"| {label} | {pct(ma.get('recall@1'))} / {ma.get('mrr', 0):.3f} | "
            f"{pct(mb.get('recall@1'))} / {mb.get('mrr', 0):.3f} | "
            f"{delta(ma.get('recall@1'), mb.get('recall@1'))} | {delta(ma.get('mrr'), mb.get('mrr'))} |")
    add("")

    add("## 3 分组表现（本系统所在配置：融合 + 重排）")
    add("")
    add(f"| 分组 | n | {args.label_a} R@1 / MRR | {args.label_b} R@1 / MRR | ΔR@1 | ΔMRR |")
    add("| --- | --- | --- | --- | --- | --- |")
    for group, label in GROUPS:
        ga, gb = by_group(ra, group), by_group(rb, group)
        ma, mb = metrics(ga, "hybrid_rerank"), metrics(gb, "hybrid_rerank")
        add(f"| {label} | {ma.get('n', 0)} | {pct(ma.get('recall@1'))} / {ma.get('mrr', 0):.3f} | "
            f"{pct(mb.get('recall@1'))} / {mb.get('mrr', 0):.3f} | "
            f"{delta(ma.get('recall@1'), mb.get('recall@1'))} | {delta(ma.get('mrr'), mb.get('mrr'))} |")
    add("")

    add("## 4 越界拒答")
    add("")
    add(f"| 项 | {args.label_a} | {args.label_b} |")
    add("| --- | --- | --- |")
    fa, fb = ra.get("refusal", {}), rb.get("refusal", {})
    add(f"| 题数 / 拒答率 | {fa.get('n_questions', '—')} / {pct(fa.get('refusal_rate'))} | "
        f"{fb.get('n_questions', '—')} / {pct(fb.get('refusal_rate'))} |")
    add(f"| 阈值 | {fa.get('min_relevance', '—')} | {fb.get('min_relevance', '—')} |")
    add(f"| 误接收 | {len(fa.get('false_accept', []))} 条 | {len(fb.get('false_accept', []))} 条 |")
    add("")

    add("## 5 时延（同一台机器，CPU，无 GPU）")
    add("")
    add(f"| 指标 | {args.label_a} | {args.label_b} |")
    add("| --- | --- | --- |")
    la, lb = ra.get("latency", {}), rb.get("latency", {})
    sa, sb = la.get("search_ms", {}), lb.get("search_ms", {})
    aa, ab = la.get("analyze_ms", {}), lb.get("analyze_ms", {})
    add(f"| 检索 P50 / P95（n）| {sa.get('p50', '—')} / {sa.get('p95', '—')}（{sa.get('n', '—')}）| "
        f"{sb.get('p50', '—')} / {sb.get('p95', '—')}（{sb.get('n', '—')}）|")
    add(f"| 端到端 P50 / P95（n）| {aa.get('p50', '—')} / {aa.get('p95', '—')}（{aa.get('n', '—')}）| "
        f"{ab.get('p50', '—')} / {ab.get('p95', '—')}（{ab.get('n', '—')}）|")
    st_a, st_b = ra.get("stage_latency", {}), rb.get("stage_latency", {})
    for stage in ("detect", "diagnose", "grade_dispatch", "archive"):
        add(f"| 分阶段 P95 {stage} | {st_a.get(stage, {}).get('p95', '—')} ms | "
            f"{st_b.get(stage, {}).get('p95', '—')} ms |")
    add("")

    notes_a = ra.get("degradation_notes") or []
    notes_b = rb.get("degradation_notes") or []
    add("## 6 降级说明（非空表示没走成配置的后端）")
    add("")
    add(f"- {args.label_a}：{notes_a if notes_a else '无'}")
    add(f"- {args.label_b}：{notes_b if notes_b else '无'}")
    add("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n已写出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
