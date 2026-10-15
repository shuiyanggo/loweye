"""monitor_inspection.py 的端点环境变量守卫。

背景：回收收据只由这个监控进程写。端点在本会话里换过三次，写死的旧地址会让它在主机换址后
静默轮询一个不存在的地址——进程活着、日志在打印、收据永远不写。所以它必须缺环境变量就立刻
退出，而不是退回某个默认值。
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREDENTIAL_KEYS = ("LOWEYE_GPU_HOST", "LOWEYE_GPU_PORT", "LOWEYE_GPU_PASS")
IMPORT_LINE = "import json, tools.gpu.monitor_inspection as m; print(json.dumps(m.endpoint_from_env()))"


def run_import(overrides):
    env = dict(os.environ)
    for key in CREDENTIAL_KEYS:
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(
        # -X utf8 必须显式传给子进程：否则子进程的中文 SystemExit 消息按 Windows 本地代码页（cp936）
        # 编码，父进程按 UTF-8 解码会 UnicodeDecodeError，stdout 变成 None。
        [sys.executable, "-X", "utf8", "-c", IMPORT_LINE],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )


class MonitorEndpointGuardTests(unittest.TestCase):
    def test_missing_endpoint_exits_and_names_the_variables(self):
        done = run_import({})
        self.assertNotEqual(done.returncode, 0)
        output = done.stdout + done.stderr
        self.assertIn("LOWEYE_GPU_HOST", output)
        self.assertIn("LOWEYE_GPU_PORT", output)

    def test_missing_password_exits_and_names_the_variable(self):
        done = run_import({"LOWEYE_GPU_HOST": "example.invalid", "LOWEYE_GPU_PORT": "1234"})
        self.assertNotEqual(done.returncode, 0)
        output = done.stdout + done.stderr
        self.assertIn("LOWEYE_GPU_PASS", output)
        self.assertNotIn("LOWEYE_GPU_HOST / LOWEYE_GPU_PORT", output)

    def test_explicit_endpoint_is_used_verbatim(self):
        done = run_import({
            "LOWEYE_GPU_HOST": "example.invalid",
            "LOWEYE_GPU_PORT": "4321",
            "LOWEYE_GPU_PASS": "not-a-real-password",
        })
        self.assertEqual(done.returncode, 0, done.stderr)
        payload = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(payload, ["example.invalid", 4321])

    def test_no_stale_default_survives_in_the_source(self):
        source = (ROOT / "tools" / "gpu" / "monitor_inspection.py").read_text(encoding="utf-8")
        self.assertNotIn("connect.bjb1.seetacloud.com", source)
        self.assertNotIn("connect.bjb2.seetacloud.com", source)
        self.assertNotIn("'23837'", source)
        self.assertIn("def endpoint_from_env():", source)
        # main() 必须先取环境里的端点再连接，不能反过来用模块常量直连
        self.assertIn("host,port=endpoint_from_env()", source)


if __name__ == "__main__":
    unittest.main()
