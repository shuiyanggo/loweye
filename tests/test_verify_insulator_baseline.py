#!/usr/bin/env python3
"""tests for tools/verify_insulator_baseline.py

覆盖：全部一致（允许新增）、字节或摘要改变、文件缺失、缺基线文件本身的结构错误，
以及真实项目基线的形状。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.verify_insulator_baseline import (  # noqa: E402
    BaselineError,
    compare,
    load_baseline,
    main,
    sha256_file,
)

RECORDED = ("cplid_fcos_1024.onnx", "cplid_fcos_best.pth")


class VerifyInsulatorBaselineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.directory = self.tmp / "detector"
        self.directory.mkdir()
        (self.directory / RECORDED[0]).write_bytes(b"insulator-onnx-payload")
        (self.directory / RECORDED[1]).write_bytes(b"insulator-pth-payload")
        self.baseline = self.tmp / "insulator_baseline.json"
        self.write_baseline()

    def tearDown(self):
        self._tmp.cleanup()

    def write_baseline(self, files=None):
        entries = files
        if entries is None:
            entries = [
                {
                    "name": name,
                    "bytes": (self.directory / name).stat().st_size,
                    "sha256": sha256_file(self.directory / name),
                }
                for name in RECORDED
            ]
        self.baseline.write_text(
            json.dumps({"schema_version": 1, "files": entries}, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---------------------------------------------------------------- 一致

    def test_unchanged_files_pass_and_write_json(self):
        payload = load_baseline(self.baseline)
        result = compare(payload, self.directory)
        self.assertTrue(result["ok"])
        self.assertEqual([row["status"] for row in result["rows"]], ["unchanged", "unchanged"])
        self.assertEqual(result["unchanged"], 2)
        self.assertEqual(result["changed"], [])
        json_out = self.tmp / "nested" / "check.json"
        code = main(
            [
                "--baseline",
                str(self.baseline),
                "--dir",
                str(self.directory),
                "--json-out",
                str(json_out),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(json_out.exists())
        self.assertTrue(json.loads(json_out.read_text(encoding="utf-8"))["ok"])

    def test_new_file_is_allowed(self):
        (self.directory / "crop_pest_fcos_800.onnx").write_bytes(b"new-model")
        result = compare(load_baseline(self.baseline), self.directory)
        self.assertTrue(result["ok"])
        self.assertEqual([item["name"] for item in result["new_files"]], ["crop_pest_fcos_800.onnx"])
        self.assertEqual(main(["--baseline", str(self.baseline), "--dir", str(self.directory)]), 0)

    # ---------------------------------------------------------------- 改动

    def test_modified_bytes_are_detected_and_exit_three(self):
        (self.directory / RECORDED[0]).write_bytes(b"insulator-onnx-payload-ALTERED")
        result = compare(load_baseline(self.baseline), self.directory)
        self.assertFalse(result["ok"])
        self.assertEqual([row["name"] for row in result["changed"]], [RECORDED[0]])
        row = result["changed"][0]
        self.assertEqual(row["status"], "modified")
        self.assertEqual(row["expected_bytes"], len(b"insulator-onnx-payload"))
        self.assertEqual(row["actual_bytes"], len(b"insulator-onnx-payload-ALTERED"))
        self.assertNotEqual(row["actual_sha256"], row["expected_sha256"])
        self.assertEqual(main(["--baseline", str(self.baseline), "--dir", str(self.directory)]), 3)

    def test_same_size_different_digest_is_modified(self):
        original = (self.directory / RECORDED[1]).read_bytes()
        (self.directory / RECORDED[1]).write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
        result = compare(load_baseline(self.baseline), self.directory)
        self.assertFalse(result["ok"])
        self.assertEqual(result["changed"][0]["status"], "modified")
        self.assertEqual(result["changed"][0]["actual_bytes"], result["changed"][0]["expected_bytes"])

    def test_missing_file_is_detected_and_exit_three(self):
        (self.directory / RECORDED[1]).unlink()
        result = compare(load_baseline(self.baseline), self.directory)
        self.assertFalse(result["ok"])
        self.assertEqual(result["changed"][0]["status"], "missing")
        self.assertIsNone(result["changed"][0]["actual_sha256"])
        self.assertEqual(main(["--baseline", str(self.baseline), "--dir", str(self.directory)]), 3)

    # ---------------------------------------------------------------- 结构

    def test_malformed_baseline_exits_two(self):
        self.write_baseline([{"name": "x.onnx", "sha256": "not-a-digest"}])
        with self.assertRaises(BaselineError):
            load_baseline(self.baseline)
        self.assertEqual(main(["--baseline", str(self.baseline), "--dir", str(self.directory)]), 2)

    def test_baseline_missing_or_not_json_exits_two(self):
        self.assertEqual(main(["--baseline", str(self.tmp / "absent.json"), "--dir", str(self.directory)]), 2)
        self.baseline.write_text("{not json", encoding="utf-8")
        self.assertEqual(main(["--baseline", str(self.baseline), "--dir", str(self.directory)]), 2)

    def test_baseline_rejects_paths_and_empty_list(self):
        self.write_baseline([{"name": "sub/dir.onnx", "sha256": "a" * 64}])
        with self.assertRaises(BaselineError):
            load_baseline(self.baseline)
        self.write_baseline([])
        with self.assertRaises(BaselineError):
            load_baseline(self.baseline)

    # ---------------------------------------------------------------- 真件

    def test_real_project_baseline_shape(self):
        real = PROJECT / "out" / "insulator_baseline.json"
        if not real.exists():
            self.skipTest("project baseline is absent")
        payload = load_baseline(real)
        self.assertGreaterEqual(len(payload["files"]), 5)
        for entry in payload["files"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertIsInstance(entry.get("bytes"), int)
        names = {entry["name"] for entry in payload["files"]}
        self.assertIn("cplid_fcos_1024.onnx", names)
        self.assertIn("rtmdet-tiny-coco.onnx", names)


if __name__ == "__main__":
    unittest.main()
