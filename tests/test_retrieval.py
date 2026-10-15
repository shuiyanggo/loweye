"""检索链路单测：RRF 融合、重排绝对分、拒答阈值。

这里守住三条设计承诺（答辩会被追问的正是这三条）：
  1. 两路召回用 RRF 融合，不靠调权重；
  2. 重排分数是绝对分，不做「一批里的相对归一化」——否则越界问题也会被顶到 1.0；
  3. 低于覆盖阈值必须拒答，而不是硬答。
"""

from __future__ import annotations

import unittest

from loweye.config import load_config
from loweye.retrieval.fusion import reciprocal_rank_fusion, top_indices
from loweye.retrieval.rerank import LexicalReranker
from loweye.retrieval.embedder import HashingTfidfEmbedder
from loweye.schemas import Clause


def clause(no: str, text: str, doc: str = "演示导则") -> Clause:
    return Clause(
        chunk_id=f"{doc}#{no}",
        doc_id=doc,
        doc_title=doc,
        clause_no=no,
        hierarchy_path="第二章 缺陷判定与处置 › 2.1 绝缘子类缺陷",
        text=text,
    )


class FusionTest(unittest.TestCase):
    def test_rank1_beats_rank2(self) -> None:
        fused = reciprocal_rank_fusion([[7, 3], [7]])
        self.assertGreater(fused[7], fused[3])

    def test_agreement_across_paths_wins(self) -> None:
        """两路都排第一的 doc，必须赢过只在一路排第一的 doc。"""
        bm25 = [1, 2]
        vector = [1, 3]
        fused = reciprocal_rank_fusion([bm25, vector])
        self.assertEqual(max(fused, key=lambda d: fused[d]), 1)

    def test_k_60_formula(self) -> None:
        fused = reciprocal_rank_fusion([[5]], k=60)
        self.assertAlmostEqual(fused[5], 1 / 61)

    def test_weights_must_match(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[1], [2]], weights=[1.0])

    def test_top_indices_tie_breaks_by_index(self) -> None:
        self.assertEqual(top_indices([0.5, 0.5, 0.9], 2), [2, 0])


class RerankTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reranker = LexicalReranker()
        self.clauses = [
            clause("2.1.2", "发现绝缘子自爆后，应立即上报线路运维管理部门，并同步安排输电运检班组在 24 小时内到达现场复核。"),
            clause("3.1.1", "导地线断股截面损失超过 7% 的，判定为危急缺陷，应立即申请停电处理。"),
            clause("4.3.2", "通道内树竹与导线净空距离不足 3 米的，应安排砍伐或修剪。"),
        ]

    def test_relevant_clause_scores_higher(self) -> None:
        scores = self.reranker.score("绝缘子自爆怎么处理", self.clauses)
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores.index(max(scores)), 0)

    def test_scores_are_absolute_not_batch_normalized(self) -> None:
        """同一批条款里，无关问题的最高分必须显著低于相关问题——
        否则拒答阈值形同虚设。"""
        good = max(self.reranker.score("绝缘子自爆怎么处理", self.clauses))
        bad = max(self.reranker.score("今天股市行情怎么样", self.clauses))
        self.assertGreater(good, 0.3)
        self.assertLess(bad, 0.12)
        self.assertGreater(good, bad * 3)

    def test_score_within_unit_range(self) -> None:
        for s in self.reranker.score("树障 净空距离", self.clauses):
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_single_clause_scores_same_as_in_batch(self) -> None:
        """绝对分的关键性质：单独喂一条，分数不能变。"""
        batch = self.reranker.score("绝缘子自爆", self.clauses)
        alone = self.reranker.score("绝缘子自爆", [self.clauses[0]])
        self.assertAlmostEqual(batch[0], alone[0], places=6)

    def test_clause_number_query_hits(self) -> None:
        scores = self.reranker.score("2.1.2 条怎么规定的", self.clauses)
        self.assertEqual(scores.index(max(scores)), 0)

    def test_empty_input(self) -> None:
        self.assertEqual(self.reranker.score("随便", []), [])


class EmbedderTest(unittest.TestCase):
    def test_vectors_are_l2_normalized(self) -> None:
        emb = HashingTfidfEmbedder(dim=512).fit(["绝缘子自爆", "树障超限"])
        vecs = emb.encode(["绝缘子自爆"])
        norm = float((vecs[0] ** 2).sum() ** 0.5)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_state_roundtrip_keeps_scores(self) -> None:
        """索引存盘再读回来，向量分必须一模一样，否则线上与离线结果不一致。"""
        emb = HashingTfidfEmbedder(dim=512).fit(["绝缘子自爆", "导地线断股", "树障超限"])
        before = emb.encode(["绝缘子自爆 处置"])[0]
        restored = HashingTfidfEmbedder.from_state(emb.state())
        after = restored.encode(["绝缘子自爆 处置"])[0]
        for a, b in zip(before, after):
            self.assertAlmostEqual(float(a), float(b), places=6)


class RefusalThresholdTest(unittest.TestCase):
    def test_threshold_sane(self) -> None:
        cfg = load_config()
        thr = float(cfg.get("retrieval.min_relevance"))
        self.assertGreater(thr, 0.0)
        self.assertLess(thr, 0.6)


if __name__ == "__main__":
    unittest.main()
