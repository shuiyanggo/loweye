"""README 路径核对工具的判定规则与整体行为。

这些用例锁住「哪些反引号 token 算路径候选」这条规则——它决定了检查是漏报还是误报：
漏掉真路径会让文档里的坏引用继续存在，把代码片段当路径则会天天误报。
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools import audit_readme_paths as audit


class TokenClassificationTests(unittest.TestCase):
    def test_line_suffix_is_stripped(self):
        self.assertEqual(audit.strip_line_suffix("loweye/detect/base.py:31-60"), "loweye/detect/base.py")
        self.assertEqual(audit.strip_line_suffix("tools/x.py:7"), "tools/x.py")
        self.assertEqual(audit.strip_line_suffix("tools/x.py"), "tools/x.py")

    def test_directory_candidates(self):
        self.assertTrue(audit.is_directory_candidate("loweye/agent/"))
        self.assertFalse(audit.is_directory_candidate("loweye/agent"))
        self.assertFalse(audit.is_directory_candidate("/root/loweye/"))
        self.assertFalse(audit.is_directory_candidate("out/{task}_manifest.json/"))

    def test_file_candidates(self):
        self.assertTrue(audit.is_file_candidate("tools/build_proposal.py"))
        self.assertTrue(audit.is_file_candidate("out/低空之眼-参赛方案书.pdf"))
        self.assertFalse(audit.is_file_candidate("detect.onnx_path"))  # 配置键，不是路径
        self.assertFalse(audit.is_file_candidate("README.md"))  # 没有斜杠，指代的是本文件
        self.assertFalse(audit.is_file_candidate("loweye/detect/"))  # 目录写法
        self.assertFalse(audit.is_file_candidate("out/{task}_manifest.json"))  # 占位符
        self.assertFalse(audit.is_file_candidate("/root/autodl-tmp/loweye/models/"))  # 主机侧
        self.assertFalse(audit.is_file_candidate("https://github.com/xpwu95/IP102"))


class AuditTests(unittest.TestCase):
    def test_reports_only_the_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "present").mkdir()
            (root / "present" / "here.py").write_text("x", encoding="utf-8")
            text = (
                "存在：`present/here.py`，缺：`present/gone.json`；\n"
                "代码片段：`detect.onnx_path`；主机侧：`/root/loweye/logs/`；\n"
                "占位：`out/{task}_manifest.json`；目录：`present/`；行号写法：`present/here.py:3-9`。\n"
            )
            report = audit.audit(text, root)
        self.assertEqual(report["missing"], ["present/gone.json"])
        self.assertEqual(report["checked_count"], 3)  # here.py（行号写法归一到同一条）、gone.json、present/ 目录
        self.assertTrue(report["checked"]["present/here.py"]["exists"])
        self.assertFalse(report["checked"]["present/gone.json"]["exists"])

    def test_report_is_json_serialisable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit.audit("`a/b.py`", Path(tmp))
        self.assertEqual(json.loads(json.dumps(report, ensure_ascii=False))["missing"], ["a/b.py"])

    def test_alternate_root_absorbs_tree_relative_paths(self):
        """README 目录结构一节以 loweye/ 为基准，别把这些路径报成缺失。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "loweye" / "agent").mkdir(parents=True)
            (root / "loweye" / "agent" / "pipeline.py").write_text("x", encoding="utf-8")
            report = audit.audit(
                "结构：`agent/pipeline.py`；真缺：`agent/nope.py`", root, (root / "loweye",)
            )
        self.assertEqual(report["missing"], ["agent/nope.py"])
        self.assertTrue(report["checked"]["agent/pipeline.py"]["exists"])
        self.assertEqual(
            report["checked"]["agent/pipeline.py"]["path"],
            str((root / "loweye" / "agent" / "pipeline.py").resolve()),
        )
        self.assertFalse(report["checked"]["agent/nope.py"]["exists"])


if __name__ == "__main__":
    unittest.main()
