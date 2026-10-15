"""切分与条款号识别的单测。

对应评赛点「文档切分是否合理」：切分错，后面检索与溯源全是错的，
所以这里测得最细。
"""

from __future__ import annotations

import unittest

from loweye.text import is_numbered_clause, split_clauses

SAMPLE = """# 架空线路无人机巡检缺陷判定与处置导则（演示占位文本）

> 本文档为项目自编演示文本，不得作为现场作业依据。

## 第一章 总则

### 1.1 适用范围

1.1.1 本导则适用于 10kV 及以上架空线路的无人机精细化巡检作业。
1.1.2 巡检影像应保留原始文件，不得裁剪或美化。

## 第二章 缺陷判定与处置

### 2.1 绝缘子类缺陷

2.1.1 绝缘子伞裙破损面积超过 5% 的，判定为危急缺陷。
2.1.2 发现绝缘子自爆后，应立即上报线路运维管理部门，
并同步安排输电运检班组在 24 小时内到达现场复核。
2.1.3 复核确认自爆片数超过该串绝缘子总片数 5% 的，应立即申请停电更换整串绝缘子。

（1）现场作业前应核对线路双重名称。
（2）作业人员应穿绝缘鞋。

第五条 本导则由运检部负责解释。
"""


class SplitClausesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clauses = split_clauses(SAMPLE)

    def test_numbered_clause_detected(self) -> None:
        numbers = [c.clause_no for c in self.clauses]
        for want in ("1.1.1", "1.1.2", "2.1.1", "2.1.2", "2.1.3", "第五条"):
            self.assertIn(want, numbers, f"没切出条款号 {want}（实际：{numbers}）")

    def test_multiline_body_not_split(self) -> None:
        """条款正文跨行时必须留在同一条里，否则引用会出现半句话。"""
        c = next(c for c in self.clauses if c.clause_no == "2.1.2")
        self.assertIn("24 小时内到达现场复核", c.text)
        self.assertIn("应立即上报线路运维管理部门", c.text)

    def test_hierarchy_path_carries_section(self) -> None:
        c = next(c for c in self.clauses if c.clause_no == "2.1.3")
        self.assertIn("第二章", c.hierarchy_path)
        self.assertIn("2.1", c.hierarchy_path)

    def test_markdown_prefix_stripped(self) -> None:
        for c in self.clauses:
            self.assertFalse(c.text.startswith("#"), f"正文残留 markdown：{c.text[:20]}")
            self.assertFalse(c.clause_no.startswith("#"), f"条款号残留 markdown：{c.clause_no}")

    def test_disclaimer_kept_but_unnumbered(self) -> None:
        """文档开头的免责声明要能检索到（否则检索不到"不得作为依据"），
        但它不是条款号——后续由 require_clause_no 过滤，不当作可执行条款。"""
        head = self.clauses[0]
        self.assertFalse(is_numbered_clause(head.clause_no))
        self.assertIn("演示文本", head.text + self.clauses[1].text)

    def test_parenthesized_items_are_clauses(self) -> None:
        numbers = [c.clause_no for c in self.clauses]
        self.assertTrue(any(n.startswith("（1）") for n in numbers), numbers)

    def test_no_empty_clause(self) -> None:
        for c in self.clauses:
            self.assertTrue(c.text.strip(), f"出现空条款：{c.clause_no}")


class IsNumberedClauseTest(unittest.TestCase):
    def test_accepts_real_clause_numbers(self) -> None:
        for good in ("1.1.1", "2.1", "4.3.2.1", "第五条", "第十二条", "3.1.1"):
            self.assertTrue(is_numbered_clause(good), good)

    def test_rejects_placeholders(self) -> None:
        for bad in ("", "段落12", "（无编号正文）", "附录A", "—", "2.1 绝缘子类缺陷"):
            self.assertFalse(is_numbered_clause(bad), bad)

    def test_parenthesized_not_counted_as_numbered(self) -> None:
        # （1）是条内项，不是可独立引用的条款号
        self.assertFalse(is_numbered_clause("（1）"))


class SplitStatsTest(unittest.TestCase):
    def test_numbered_ratio_between_0_and_1(self) -> None:
        clauses = split_clauses(SAMPLE)
        numbered = sum(1 for c in clauses if is_numbered_clause(c.clause_no))
        self.assertGreater(numbered / len(clauses), 0.6)


if __name__ == "__main__":
    unittest.main()
