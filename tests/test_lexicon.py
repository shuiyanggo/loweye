"""术语表测试：口语词 → 规程术语的查询扩展与概念折叠。

这组测试护的是检索质量攻坚的那套机制：
  expand() 只追加、不改写（改写会丢判别性词，L-30「明火扑灭后」曾被换没）；
  collapse() 把别名并到规范词（否则同一个概念在分母里占两个名额，
  正确条款的覆盖率反而被拉低）；
  排除声明必须同时被 is_exclusion_text 与 is_scope_statement 命中
  —— search() 里「先判范围排除、再清理元信息条款」的顺序就靠这条成立。
"""

from __future__ import annotations

import unittest

from loweye.config import load_config
from loweye.retrieval.lexicon import TermLexicon, build_lexicon
from loweye.text import is_exclusion_text, is_scope_statement

CFG = load_config()


class TermLexiconTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lex = build_lexicon(CFG)
        cls.kb = None

    def test_real_lexicon_is_loaded(self) -> None:
        """配置里的术语表要真的被读进来，不是空表。"""
        self.assertGreaterEqual(len(self.lex), 30, "术语表条目太少，检查 config/lexicon.yaml")
        self.assertGreater(self.lex.alias_count, 100)

    def test_expand_appends_and_keeps_original(self) -> None:
        """扩展只追加规范词，原始用词必须原样保留。"""
        q = "明火扑灭后还要监视多久"
        out = self.lex.expand(q)
        self.assertIn("明火", out, "expand 把原词改掉了，判别性词会丢")
        self.assertIn(q, out, "原句必须完整保留在扩展后的查询里")
        self.assertIn("火情", out, "别名未映射到规范词")

    def test_normalize_is_replacement_style(self) -> None:
        """normalize 是替换式写法，仅用于展示「系统听成了什么」。"""
        self.assertEqual(self.lex.normalize("瓷瓶"), "绝缘子")

    def test_collapse_merges_aliases(self) -> None:
        tokens = self.lex.collapse(["瓷瓶", "绝缘子", "绝缘串"])
        self.assertEqual(tokens, {"绝缘子"}, f"别名未折叠：{tokens}")

    def test_collapse_keeps_unknown_tokens(self) -> None:
        tokens = self.lex.collapse(["火星车", "绝缘子"])
        self.assertIn("火星车", tokens, "未收录的词必须原样保留，不能吞掉")
        self.assertIn("绝缘子", tokens)

    def test_describe_reports_matched_terms(self) -> None:
        info = self.lex.describe("瓷瓶碎了一片")
        self.assertIn("绝缘子", info["matched_terms"])
        self.assertIn("自爆", info["matched_terms"])

    def test_empty_lexicon_disables_expansion(self) -> None:
        """空术语表 = 关闭开关，扩展与折叠都必须退化成恒等。"""
        empty = TermLexicon([])
        q = "瓷瓶碎了一片"
        self.assertEqual(empty.expand(q), q)
        self.assertEqual(empty.collapse(["绝缘子", "自爆"]), {"绝缘子", "自爆"})

    def test_load_missing_file_returns_empty(self) -> None:
        lex = TermLexicon.load(CFG.abs_path("config/__not_exists__.yaml"))
        self.assertEqual(len(lex), 0)


class ScopeGateHelpersTest(unittest.TestCase):
    """两个判据本身的行为——search() 的判定顺序依赖它们同时命中排除条款。"""

    EXCLUSION = "本导则不适用于带电作业、直升机巡检以及红外测温的定量判读。"
    SCOPE = "本导则适用于 110kV 及以上架空线路的可见光巡检。"

    def test_exclusion_text_detected(self) -> None:
        self.assertTrue(is_exclusion_text(self.EXCLUSION))
        self.assertFalse(is_exclusion_text(self.SCOPE))

    def test_scope_statement_detected(self) -> None:
        self.assertTrue(is_scope_statement(self.SCOPE))

    def test_exclusion_and_scope_predicates_are_separate(self) -> None:
        """演示文档里两类声明是分开写的：排除声明不是「适用范围」条款。"""
        self.assertFalse(is_scope_statement(self.EXCLUSION))
        self.assertFalse(is_exclusion_text(self.SCOPE))

    def test_one_clause_can_match_both_predicates(self) -> None:
        """但只要一句话里既有「本导则适用于」又有「不适用于」，两个判据就都会命中。

        这正是 search() 里顺序不能反的原因：范围排除必须在**未过滤**的排序上判定，
        若先做元信息条款清理、再判排除，这条拒答信号就被清掉了。
        """
        both = "本导则适用于 110kV 及以上架空线路，不适用于带电作业与红外测温的定量判读。"
        self.assertTrue(is_scope_statement(both))
        self.assertTrue(is_exclusion_text(both))

    def test_normal_clause_is_neither(self) -> None:
        text = "发现绝缘子自爆后，应立即上报线路运维管理部门，并同步安排输电运检班组在 24 小时内到达现场复核。"
        self.assertFalse(is_exclusion_text(text))
        self.assertFalse(is_scope_statement(text))


if __name__ == "__main__":
    unittest.main()
