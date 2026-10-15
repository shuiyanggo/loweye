"""检索闸门与术语表效果的系统级测试（真跑知识库，不打桩）。

三条不变量：
  1. 范围排除闸门：最相似条款是「本导则不适用于……」时，该问题必须 covered=False
     且 scope_excluded=True，不能因为词面重合度高（分数 0.30~0.53）被当成可回答。
  2. 元信息条款不上位：「适用范围 / 术语和定义 / 编制目的」不得占据 top1 被当成处置依据。
  3. 术语表有效：同一批改写题，开着术语表的命中数必须**严格多于**关掉时的命中数。
     这条是消融断言——它保证 R@1 从 42% 到 83% 那个提升真的有机制在支撑，
     而不是某次碰巧。反过来说，如果将来有人把 lexicon 接线拆了，这里会立刻红。
"""

from __future__ import annotations

import unittest

from loweye.config import load_config
from loweye.eval.goldset import load_goldset
from loweye.retrieval.index import KnowledgeBase
from loweye.retrieval.lexicon import TermLexicon
from loweye.text import is_exclusion_text, is_scope_statement

CFG = load_config()


class RetrievalScopeGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = KnowledgeBase.load(CFG.path("index_dir"), CFG)
        cls.gold = load_goldset(CFG.abs_path(CFG.require("eval.goldset")))

    def _search(self, question: str):
        # 查询扩展由 KnowledgeBase.search() 内部完成（它会把实际用的查询串
        # 回传到 search_query），调用方一律传原始提问，避免扩展两次。
        return self.kb.search(question)

    def test_out_of_scope_questions_are_refused(self) -> None:
        """四条边界越界题：知识库里恰好有「不适用于……」的声明，必须拒答。"""
        ids = ("X-04", "X-05", "X-06", "X-07")
        by_id = {item.id: item for item in self.gold.out_of_scope}
        for qid in ids:
            item = by_id[qid]
            with self.subTest(qid=qid):
                resp = self._search(item.question)
                self.assertFalse(resp.covered, f"{qid} 应拒答：{item.question}")
                self.assertTrue(resp.scope_excluded, f"{qid} 应命中范围排除闸门，而不是分数不够")
                self.assertTrue(
                    is_exclusion_text(resp.hits[0].clause.text),
                    f"{qid} 的 top1 应当是排除声明，实际是 {resp.hits[0].clause.clause_no}",
                )

    def test_noise_questions_are_refused_without_scope_flag(self) -> None:
        """纯噪声问题（股市之类）走的是分数阈值，不是范围排除，两者要能分开。"""
        resp = self._search("今天股市行情怎么样")
        self.assertFalse(resp.covered)
        self.assertFalse(resp.scope_excluded)

    def test_scope_statement_never_becomes_top1(self) -> None:
        """域内问题里，top1 不允许是「适用范围/术语和定义/编制目的」这类自我介绍条款。"""
        for item in self.gold.in_scope:
            with self.subTest(qid=item.id):
                resp = self._search(item.question)
                self.assertTrue(resp.hits, f"{item.id} 一条都没检索到")
                self.assertFalse(
                    is_scope_statement(resp.hits[0].clause.text),
                    f"{item.id} 的 top1 是元信息条款 {resp.hits[0].clause.clause_no}",
                )

    def test_corpus_exclusion_clauses_are_detectable(self) -> None:
        """三篇演示文档各有一条排除声明，且都不属于「适用范围」类条款。

        这条同时是对 search() 判定顺序的回归保护：两个判据在真实语料上是分开的，
        但排序判定必须先做范围排除（见 loweye/retrieval/index.py 的注释）。
        """
        exclusions = [c for c in self.kb.clauses if is_exclusion_text(c.text)]
        self.assertEqual(len(exclusions), 3, [c.clause_no for c in exclusions])
        for c in exclusions:
            self.assertFalse(
                is_scope_statement(c.text),
                f"{c.doc_id}:{c.clause_no} 同时被判为元信息条款，判定顺序的保护就生效了",
            )

    def test_search_query_is_reported(self) -> None:
        """响应要回传「系统把这句话听成了什么」，前端才能解释。"""
        q = "瓷瓶碎了一片"
        resp = self._search(q)
        self.assertEqual(resp.search_query, self.kb.lexicon.expand(q))
        self.assertIn("绝缘子", resp.matched_terms)

class LexiconAblationTest(unittest.TestCase):
    """术语表消融：关掉之后，改写题的命中率必须掉下来。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = KnowledgeBase.load(CFG.path("index_dir"), CFG)
        cls.gold = load_goldset(CFG.abs_path(CFG.require("eval.goldset")))
        cls.paraphrase = [i for i in cls.gold.in_scope if i.group == "paraphrase"]

    def _hit_count(self, lexicon) -> int:
        saved = self.kb.lexicon
        self.kb.set_lexicon(lexicon)
        try:
            hits = 0
            for item in self.paraphrase:
                resp = self.kb.search(item.question, top_k=1)
                if resp.hits and item.matches(resp.hits[0].clause.doc_id, resp.hits[0].clause.clause_no):
                    hits += 1
            return hits
        finally:
            self.kb.set_lexicon(saved)

    def test_paraphrase_questions_benefit_from_lexicon(self) -> None:
        self.assertGreaterEqual(len(self.paraphrase), 10, "改写题太少，消融没有说服力")
        on = self._hit_count(self.kb.lexicon)
        off = self._hit_count(TermLexicon([]))
        self.assertGreater(
            on,
            off,
            f"术语表没有产生作用：开启命中 {on}/{len(self.paraphrase)}，关闭命中 {off}",
        )

    def test_literal_questions_do_not_regress(self) -> None:
        """加了术语表不能把原词题带坏（曾经的替换式归一化把 L-30 弄丢过）。"""
        literal = [i for i in self.gold.in_scope if i.group == "literal"]
        missed_on, missed_off = [], []
        saved = self.kb.lexicon
        for lex, bucket in ((saved, missed_on), (TermLexicon([]), missed_off)):
            self.kb.set_lexicon(lex)
            try:
                for item in literal:
                    resp = self.kb.search(item.question, top_k=1)
                    if not (resp.hits and item.matches(resp.hits[0].clause.doc_id, resp.hits[0].clause.clause_no)):
                        bucket.append(item.id)
            finally:
                self.kb.set_lexicon(saved)
        self.assertLessEqual(
            len(missed_on),
            len(missed_off),
            f"术语表让原词题变差了：开启漏 {missed_on}，关闭漏 {missed_off}",
        )


if __name__ == "__main__":
    unittest.main()
