"""配置读取单测。

`Config.path` 只服务 paths.* 下的键，其它位置的路径要用 `Config.abs_path`——
这个区别踩过一次坑（severity.rules_file 报「缺少必需配置项」），所以固化在这里。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from loweye.agent.grading import GradingTable
from loweye.config import Config, ConfigError, load_config, load_routes
from loweye.detect.onnx_detector import DEFAULT_CLASS_NAMES


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()

    def test_required_keys_present(self) -> None:
        for key in (
            "app.name",
            "retrieval.min_relevance",
            "severity.rules_file",
            "llm.provider",
            "detect.detector",
            "tools.report.template",
        ):
            self.assertIsNotNone(self.cfg.get(key), f"缺配置项 {key}")

    def test_get_dotted(self) -> None:
        self.assertEqual(self.cfg.get("server.port"), 8848)

    def test_default_value(self) -> None:
        self.assertEqual(self.cfg.get("不存在的键", "兜底"), "兜底")

    def test_require_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            self.cfg.require("不存在的键")

    def test_path_resolves_under_root(self) -> None:
        p = self.cfg.path("assets")
        self.assertTrue(p.is_absolute())
        self.assertEqual(p.name, "assets")
        self.assertTrue(p.exists())

    def test_abs_path_handles_non_paths_section(self) -> None:
        """severity.rules_file 不在 paths 段里，必须也能解析成绝对路径。"""
        p = self.cfg.abs_path(self.cfg.require("severity.rules_file"))
        self.assertTrue(p.is_absolute())
        self.assertTrue(p.exists(), f"{p} 不存在")

    def test_abs_path_relative_goes_to_project_root(self) -> None:
        """相对路径一律相对项目根解析（root 由 paths.out 反推）。"""
        root = self.cfg.path("out").parent  # <项目根>/out → 项目根
        self.assertEqual(self.cfg.abs_path("out/x.json"), root / "out" / "x.json")

    def test_report_template_exists(self) -> None:
        p = self.cfg.abs_path(self.cfg.get("tools.report.template"))
        self.assertTrue(p.exists(), f"报告模板缺失：{p}")


class RoutesTest(unittest.TestCase):
    """routes.yaml 是**一张平表**：defaults / rules（按类别名索引）/ human_review。

    场景不在这张表里分层，而是由类别名推导（pipeline 的 `_scene_for`）。
    这里固化该结构，并盯住一条关键不变量：规则表必须覆盖全部检测类别——
    漏一个类别，该缺陷就会以「未知类别」身份被强制转人工。
    """

    def setUp(self) -> None:
        self.routes = load_routes()
        self.table = GradingTable(self.routes)

    def test_top_level_sections(self) -> None:
        for key in ("defaults", "rules", "human_review"):
            self.assertIn(key, self.routes, f"routes.yaml 缺 {key} 段")

    def test_rules_cover_detector_classes(self) -> None:
        """规则表要覆盖：演示图的类别（DEFAULT_CLASS_NAMES）+ 真实权重导出的类别。

        真实权重的类别（CPLID 微调的 `defect` / `insulator`）漏在表外不会报错，
        只会静默走默认等级并全部转人工——现场表现就是「检出 0.82 也判 P3」。
        """
        expected = set(DEFAULT_CLASS_NAMES) | {"defect", "insulator"}
        self.assertEqual(set(self.table.known_classes), expected)

    def test_real_weight_classes_are_routed(self) -> None:
        g = self.table.grade("defect", confidence=0.82, grounded=True)
        self.assertTrue(g.matched, "真实权重的 defect 必须在规则表里")
        self.assertEqual(g.severity, "P1")
        self.assertFalse(g.normal)

    def test_normal_class_creates_no_work_order(self) -> None:
        """标了 normal: true 的类别（未见缺陷的绝缘子）只出结论，不出工单。"""
        g = self.table.grade("insulator", confidence=0.9, grounded=True)
        self.assertTrue(g.matched)
        self.assertTrue(g.normal)
        self.assertTrue(self.table.is_normal("insulator"))
        self.assertFalse(self.table.is_normal("defect"))

    def test_insulator_broken_is_p1(self) -> None:
        rule = self.routes["rules"]["insulator_broken"]
        self.assertEqual(rule["severity"], "P1")
        self.assertEqual(rule["sla_hours"], 24)
        self.assertEqual(rule["assignee_role"], "输电运检班")

    def test_p1_forces_human_review(self) -> None:
        g = self.table.grade("insulator_broken", confidence=0.93, grounded=True)
        self.assertTrue(g.matched)
        self.assertTrue(g.needs_human_review, "P1 必须转人工确认")
        self.assertTrue(any("强制人工确认" in r for r in g.reasons))

    def test_unknown_class_falls_back_and_reviews(self) -> None:
        g = self.table.grade("not_a_real_class", confidence=0.95, grounded=True)
        self.assertFalse(g.matched)
        self.assertEqual(g.severity, self.routes["defaults"]["severity"])
        self.assertTrue(any("不在分级规则表" in r for r in g.reasons))

    def test_every_rule_has_keywords_and_valid_severity(self) -> None:
        for cls, spec in self.routes["rules"].items():
            self.assertTrue(spec.get("keywords"), f"{cls} 缺检索关键词")
            self.assertIn(spec.get("severity"), {"P1", "P2", "P3", "P4"}, f"{cls} 等级异常")
            self.assertGreater(int(spec.get("sla_hours", 0)), 0, f"{cls} 的 SLA 不合理")

    def test_build_query_carries_label_and_keywords(self) -> None:
        q = self.table.build_query("fire_smoke")
        self.assertIn("火情烟雾", q)
        self.assertIn("山火", q)
        self.assertTrue(q.endswith("处置"))


class ConfigFromDictTest(unittest.TestCase):
    def test_root_override(self) -> None:
        cfg = Config({"paths": {"out": "out"}}, root=Path("D:/tmp/xyz"))
        self.assertEqual(str(cfg.path("out")).replace("\\", "/"), "D:/tmp/xyz/out")


if __name__ == "__main__":
    unittest.main()
