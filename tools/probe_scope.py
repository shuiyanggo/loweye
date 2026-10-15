"""探测检索分数分布，为「覆盖率判定规则」提供依据（只读，不改任何东西）。

背景：min_relevance=0.18 是个绝对阈值，对原词提问够用，但现场口吻的改写题
（不含文档原词）拿不到这么高的词面分，结果是被拒答 —— 明明条款已经排在第一。
本脚本把每道金标题的「绝对分」和「首位相对其余候选的突出程度」都摊开打印，
用来判断「绝对阈值」是否应该与「相对突出度」组成双信号判据，以及两条信号的
取值边界应该画在哪里。

输出三张表：域内题（原词）/ 域内题（改写）/ 越界题，每行给出
top1、top2、top1/top2、池中位数、top1/池P90、当前 covered 判定。
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loweye.config import load_config  # noqa: E402
from loweye.eval.goldset import load_goldset  # noqa: E402
from loweye.retrieval.index import KnowledgeBase  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if p >= 100:
        return ordered[-1]
    k = max(0, min(len(ordered) - 1, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def main() -> int:
    cfg = load_config()
    kb = KnowledgeBase.load(cfg.path("index_dir"), cfg)
    gold = load_goldset(cfg.abs_path(cfg.require("eval.goldset")))
    print(f"知识库：{len(kb.clauses)} 块，min_relevance={kb.min_relevance}")
    print()

    header = (
        f"{'id':<6}{'cov':<6}{'top1':>7}{'top2':>7}{'t1/t2':>7}"
        f"{'中位':>7}{'top1/P90':>9}  {'命中':<6}{'top1 条款':<34}"
    )
    groups: list[tuple[str, str]] = [
        ("literal", "域内题 · 原词提问"),
        ("paraphrase", "域内题 · 改写提问（现场口吻）"),
        ("refusal", "越界题（应当拒答）"),
    ]

    for key, title in groups:
        print("=" * 110)
        print(title)
        print("-" * 110)
        print(header)
        if key == "refusal":
            items = [(r.id, r.question, "", []) for r in gold.out_of_scope]
        else:
            items = [(g.id, g.question, g.doc, g.clauses) for g in gold.by_group(key)]

        ratios: list[float] = []
        pool_ratios: list[float] = []
        for qid, question, doc, clauses in items:
            resp = kb.search(question, top_k=len(kb.clauses))
            scores = [h.rerank for h in resp.hits]
            top1 = scores[0] if scores else 0.0
            top2 = scores[1] if len(scores) > 1 else 0.0
            med = statistics.median(scores) if scores else 0.0
            p90 = percentile(scores, 90)
            ratio = (top1 / top2) if top2 > 1e-9 else float("inf")
            pratio = (top1 / p90) if p90 > 1e-9 else float("inf")
            if key != "refusal":
                ratios.append(ratio)
                pool_ratios.append(pratio)
                hit = "-"
                if resp.hits:
                    h = resp.hits[0]
                    hit = "是" if (h.clause.doc_id == doc and h.clause.clause_no in clauses) else "否"
                top1_label = f"{resp.hits[0].clause.doc_id}:{resp.hits[0].clause.clause_no}" if resp.hits else ""
            else:
                hit = "-"
                top1_label = f"{resp.hits[0].clause.doc_id}:{resp.hits[0].clause.clause_no}" if resp.hits else ""
            ratio_s = "inf" if ratio == float("inf") else f"{ratio:.2f}"
            pratio_s = "inf" if pratio == float("inf") else f"{pratio:.2f}"
            print(
                f"{qid:<6}{('是' if resp.covered else '否'):<6}{top1:>7.3f}{top2:>7.3f}{ratio_s:>7}"
                f"{med:>7.3f}{pratio_s:>9}  {hit:<6}{top1_label:<34}"
            )
        if ratios:
            finite = [r for r in ratios if r != float("inf")]
            print(
                f"  → top1/top2 中位数 {statistics.median(finite) if finite else 0:.2f}"
                f"（inf {ratios.count(float('inf'))} 题）"
                f"；top1/池P90 中位数 {statistics.median([p for p in pool_ratios if p != float('inf')]) if any(p != float('inf') for p in pool_ratios) else 0:.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
