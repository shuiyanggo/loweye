#!/usr/bin/env python3
"""tests for tools/verify_delivered_detectors.py 的回收收据解析。

回归背景：tools/gpu/monitor_inspection.py 写出的 {task}_recovery.json 是一个
JSON 列表（收据条目数组），而本工具早期只认 {"recovered": [...]}，导致真实回收
之后 verify_delivered_detectors 仍然报 not_recovered。这里锁死三种形状都接受。
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

from tools.verify_delivered_detectors import (  # noqa: E402
    DEFAULT_OUT,
    entry_for,
    local_for,
    receipt_entries,
)

ONNX_ENTRY = {
    "remote": "/root/autodl-tmp/loweye/models/crop_pest_fcos_800.onnx",
    "local": "models/detector/crop_pest_fcos_800.onnx",
    "sha256": "425cfd8a7920a420864bfff3e16779c0464adb640997136666a21af921907512",
}
BEST_ENTRY = {
    "remote": "/root/autodl-tmp/loweye/crop_pest/crop_pest_fcos_best.pth",
    "local": "models/detector/crop_pest_fcos_best.pth",
    "sha256": "8a599154" + "0" * 56,
}


class ReceiptEntriesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, payload, task="crop_pest"):
        path = self.out_dir / f"{task}_recovery.json"
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------ 真实形态（回归点）

    def test_monitor_bare_list_receipt_is_accepted(self):
        self.write([ONNX_ENTRY, BEST_ENTRY])
        entries = receipt_entries(self.out_dir, "crop_pest")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entry_for(entries, ".onnx")["sha256"], ONNX_ENTRY["sha256"])
        self.assertEqual(entry_for(entries, "fcos_best.pth")["local"], BEST_ENTRY["local"])

    def test_bare_list_ignores_non_dict_items(self):
        self.write([ONNX_ENTRY, "garbage", 12, None])
        self.assertEqual(len(receipt_entries(self.out_dir, "crop_pest")), 1)

    # ------------------------------------------------ 兼容形态

    def test_legacy_recovered_list_and_dict(self):
        self.write({"recovered": [ONNX_ENTRY, BEST_ENTRY]})
        self.assertEqual(len(receipt_entries(self.out_dir, "crop_pest")), 2)
        self.write({"recovered": ONNX_ENTRY})
        self.assertEqual(len(receipt_entries(self.out_dir, "crop_pest")), 1)

    def test_single_entry_object_is_accepted(self):
        self.write(ONNX_ENTRY)
        entries = receipt_entries(self.out_dir, "crop_pest")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["remote"], ONNX_ENTRY["remote"])

    def test_unrelated_object_is_not_a_receipt(self):
        self.write({"note": "no entries here"})
        self.assertEqual(receipt_entries(self.out_dir, "crop_pest"), [])

    # ------------------------------------------------ 错误输入

    def test_missing_malformed_and_non_json_receipts_are_empty(self):
        self.assertEqual(receipt_entries(self.out_dir, "crop_pest"), [])
        self.write("{not json")
        self.assertEqual(receipt_entries(self.out_dir, "crop_pest"), [])
        self.write('"just a string"')
        self.assertEqual(receipt_entries(self.out_dir, "crop_pest"), [])

    def test_entry_for_returns_empty_when_absent(self):
        self.assertEqual(entry_for([ONNX_ENTRY], ".pth"), {})

    # ------------------------------------------------ 落点约定

    def test_local_for_keeps_model_files_under_models_detector(self):
        onnx_local = local_for(ONNX_ENTRY["remote"])
        self.assertEqual(onnx_local.parent, PROJECT / "models" / "detector")
        self.assertEqual(onnx_local.name, "crop_pest_fcos_800.onnx")
        json_local = local_for("/root/loweye/logs/crop_pest_test.json")
        self.assertEqual(json_local.parent, DEFAULT_OUT)
        self.assertEqual(json_local.name, "crop_pest_test.json")

    # ------------------------------------------------ 磁盘上的真件

    def test_real_project_receipt_is_readable(self):
        receipt = DEFAULT_OUT / "crop_pest_recovery.json"
        if not receipt.exists():
            self.skipTest("crop_pest recovery receipt is absent")
        entries = receipt_entries(DEFAULT_OUT, "crop_pest")
        self.assertTrue(entries, f"{receipt} 应能解析出条目（monitor 写的是 JSON 列表）")
        onnx = entry_for(entries, ".onnx")
        self.assertTrue(str(onnx.get("remote", "")).endswith(".onnx"))
        self.assertEqual(len(str(onnx.get("sha256", ""))), 64)


if __name__ == "__main__":
    unittest.main()
