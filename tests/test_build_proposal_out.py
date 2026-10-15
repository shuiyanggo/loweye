"""方案书构建的 --out 路径约束。

背景（实测）：Edge 的无头 `--print-to-pdf` 按 Edge 自己的进程工作目录解析相对路径。
构建脚本原先把 `--out` 原样传给 Edge，于是 `--out out\\_rehearsal_proposal.pdf` 会得到
`Failed to write file out\\_rehearsal_proposal.pdf: 系统找不到指定的路径。 (0x3)`，
脚本以「打印 PDF 失败」退出 1。默认值是绝对路径，所以这条只在显式传相对路径时暴露——
本次是在收尾前做 PDF 路径排练时才发现的。

这两组测试锁住修复：归一化函数本身，以及「传给 Edge 的永远是绝对路径」这条约束。
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

from tools import build_proposal


class ResolvePdfOutTests(unittest.TestCase):
    def test_relative_path_becomes_absolute(self):
        got = build_proposal.resolve_pdf_out("out/_edge_relative_check.pdf")
        self.assertTrue(got.is_absolute())
        self.assertEqual(got, (Path.cwd() / "out" / "_edge_relative_check.pdf").resolve())

    def test_absolute_path_is_preserved(self):
        src = Path.cwd() / "out" / "低空之眼-参赛方案书.pdf"
        self.assertEqual(build_proposal.resolve_pdf_out(str(src)), src.resolve())

    def test_chinese_output_name_survives(self):
        got = build_proposal.resolve_pdf_out("out/低空之眼-参赛方案书.pdf")
        self.assertEqual(got.name, "低空之眼-参赛方案书.pdf")


class PrintToPdfArgumentTests(unittest.TestCase):
    def test_edge_receives_absolute_path_even_for_relative_out(self):
        captured = {}

        def fake_run(argv, **_kwargs):
            captured["argv"] = list(argv)
            target = next(a.split("=", 1)[1] for a in argv if a.startswith("--print-to-pdf="))
            captured["target"] = target
            Path(target).write_bytes(b"%PDF-1.4\n")  # 让构建脚本的 out.exists() 成立
            return mock.Mock(returncode=0, stderr="")

        argv = ["build_proposal", "--out", "out/_edge_arg_check.pdf"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            build_proposal, "find_edge", return_value="edge"
        ), mock.patch.object(build_proposal.subprocess, "run", side_effect=fake_run):
            code = build_proposal.main()

        self.assertEqual(code, 0)
        self.assertTrue(
            Path(captured["target"]).is_absolute(),
            f"传给 Edge 的路径必须是绝对路径，实际是 {captured['target']!r}",
        )
        self.assertEqual(Path(captured["target"]).name, "_edge_arg_check.pdf")
        Path(captured["target"]).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
