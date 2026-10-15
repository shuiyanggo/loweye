"""Tests for tools/record_pua_delivery.py.

台账是要长期留存、以后会被逐行读的交付记录，所以属性只有三条：一行一条合法 JSON、UTF-8 无 BOM、
追加不破坏已有行。另外用户明确说「不记录评分」时，脚本必须拒绝写入评分，而不是静默丢掉这个字段——
静默丢弃会让记录看起来完整、其实缺项。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.record_pua_delivery import append_records, load_records  # noqa: E402

BOM = b"\xef\xbb\xbf"


def record(**overrides):
    payload = {
        "ts": "2026-10-14T21:58:00+08:00",
        "rating": None,
        "ratings_recorded": False,
        "user_instruction": "记录一次有可见 PUA 输出的交付，不记录评分",
        "task_summary": "低空之眼第 12 轮交付：crop_pest 11/24 轮、本机 102 类推理通过",
    }
    payload.update(overrides)
    return payload


class AppendTests(unittest.TestCase):
    def test_writes_utf8_lines_without_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "out" / "pua_feedback.jsonl"
            summary = append_records(ledger, [record(), record(ts="2026-10-14T22:30:00+08:00")])

            self.assertEqual(summary["lines_before"], 0)
            self.assertEqual(summary["lines_after"], 2)
            self.assertEqual(summary["bad_lines"], [])
            self.assertFalse(summary["bom"])
            self.assertFalse(ledger.read_bytes().startswith(BOM))

            lines = ledger.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["ts"] for line in lines],
                             ["2026-10-14T21:58:00+08:00", "2026-10-14T22:30:00+08:00"])
            self.assertIn("crop_pest", json.loads(lines[0])["task_summary"])

    def test_missing_trailing_newline_does_not_glue_records(self):
        """上一次写入没留换行时先补换行；否则两条记录会粘成一行、整行不再是合法 JSON。"""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "pua_feedback.jsonl"
            ledger.write_text(json.dumps(record(ts="earlier"), ensure_ascii=False), encoding="utf-8")

            summary = append_records(ledger, [record(ts="later")])

            self.assertEqual(summary["lines_before"], 1)
            self.assertEqual(summary["lines_after"], 2)
            self.assertEqual(summary["bad_lines"], [])
            self.assertEqual([json.loads(line)["ts"] for line in
                              ledger.read_text(encoding="utf-8").splitlines()], ["earlier", "later"])

    def test_reports_existing_corruption_instead_of_hiding_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "pua_feedback.jsonl"
            ledger.write_text('{"ts": "broken"\n' + json.dumps(record()) + "\n", encoding="utf-8")

            summary = append_records(ledger, [record(ts="later")])

            self.assertEqual(summary["bad_lines"], [1])
            self.assertEqual(summary["lines_after"], 3)


class LoadTests(unittest.TestCase):
    def _write(self, tmp: str, payload) -> Path:
        path = Path(tmp) / "record.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_accepts_single_object_and_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(load_records(self._write(tmp, record()))), 1)
            self.assertEqual(len(load_records(self._write(tmp, [record(), record()]))), 2)

    def test_rejects_a_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                load_records(self._write(tmp, record(rating="很有用")))
            self.assertIn("rating", str(caught.exception))

    def test_rejects_missing_core_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                load_records(self._write(tmp, {"ts": "2026-10-14T21:58:00+08:00"}))
            self.assertIn("task_summary", str(caught.exception))

    def test_rejects_empty_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                load_records(self._write(tmp, []))


if __name__ == "__main__":
    unittest.main()
