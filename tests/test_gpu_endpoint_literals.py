"""GPU 端点字面量的仓库级守卫。

背景：本会话里实例换过三次，工具里写死的旧地址有两种坏法——后台轮询进程会静默轮询一个不存在的
地址（已在 tests/test_monitor_env_guard.py 单独守住），人手动跑的工具则会照着过期的用法示例连到
已退役的机器。这里把规则收敛成两条，扫 tools/ 下全部 Python 文件：

  R1 退役地址不得以任何形式留在 tools/ 里（含文档字符串与注释）；
  R2 代码里的主机字面量必须出现在 os.environ.get("LOWEYE_GPU_HOST", ...) 的默认值位置——
     否则换实例时的唯一改动点会散落在代码里。

静态扫描与运行行为都测：R1/R2 是静态的，endpoint_from_env() 是实跑的。
"""
import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RETIRED = "connect.bjb1.seetacloud.com"
HOST_SUFFIX = ".seetacloud.com"
CREDENTIAL_KEYS = ("LOWEYE_GPU_HOST", "LOWEYE_GPU_PORT", "LOWEYE_GPU_PASS")
IMPORT_LINE = "import tools.gpu.await_and_fetch as a; print(a.endpoint_from_env())"


def tool_files():
    return sorted(TOOLS.rglob("*.py"))


def docstring_constants(tree):
    marks = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                marks.add(id(first.value))
    return marks


def env_default_constants(tree):
    marks = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "attr", None) != "get":
            continue
        if getattr(getattr(func, "value", None), "attr", None) != "environ":
            continue
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                marks.add(id(arg))
    return marks


def run_import(overrides):
    env = dict(os.environ)
    for key in CREDENTIAL_KEYS:
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(
        # -X utf8 与 errors="replace" 是必需的：子进程的中文 SystemExit 消息默认按 Windows 本地
        # 代码页编码，父进程按 UTF-8 解码会在 stdout 上抛 UnicodeDecodeError。
        [sys.executable, "-X", "utf8", "-c", IMPORT_LINE],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )


class GpuEndpointLiteralTests(unittest.TestCase):
    def test_scan_actually_covers_many_tools(self):
        # 防止扫描范围被悄悄改空：tools/ 下确实有一批 Python 文件。
        self.assertGreaterEqual(len(tool_files()), 10)

    def test_no_retired_endpoint_anywhere_in_tools(self):
        hits = []
        for path in tool_files():
            if RETIRED in path.read_text(encoding="utf-8"):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(hits, [], "退役地址仍留在 tools/ 里：%s" % hits)

    def test_host_literals_are_env_overridable(self):
        hits = []
        for path in tool_files():
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            allowed = docstring_constants(tree) | env_default_constants(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if HOST_SUFFIX not in node.value or id(node) in allowed:
                    continue
                hits.append("%s:%d" % (path.relative_to(ROOT).as_posix(), node.lineno))
        self.assertEqual(hits, [], "主机字面量没有走环境变量：%s" % hits)

    def test_await_and_fetch_requires_endpoint_env(self):
        done = run_import({})
        self.assertNotEqual(done.returncode, 0)
        output = done.stdout + done.stderr
        self.assertIn("LOWEYE_GPU_HOST", output)
        self.assertIn("LOWEYE_GPU_PORT", output)

    def test_await_and_fetch_requires_password_env(self):
        done = run_import({"LOWEYE_GPU_HOST": "example.invalid", "LOWEYE_GPU_PORT": "4321"})
        self.assertNotEqual(done.returncode, 0)
        output = done.stdout + done.stderr
        self.assertIn("LOWEYE_GPU_PASS", output)
        self.assertNotIn("LOWEYE_GPU_HOST / LOWEYE_GPU_PORT", output)

    def test_await_and_fetch_uses_explicit_endpoint_verbatim(self):
        done = run_import({
            "LOWEYE_GPU_HOST": "example.invalid",
            "LOWEYE_GPU_PORT": "4321",
            "LOWEYE_GPU_PASS": "not-a-real-password",
        })
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip().splitlines()[-1], "('example.invalid', '4321')")


if __name__ == "__main__":
    unittest.main()
