"""tools/write_detector_variant_config.py 的渲染与元数据解析测试。

这些断言针对的正是「只在推理期才会暴露」的那四项：onnx_path / input_size /
class_names（顺序）/ model_version，以及生成后模板其余部分是否原样保留。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.write_detector_variant_config import load_metadata, render_variant

PROJECT = Path(__file__).resolve().parents[1]
BASE = (PROJECT / "config" / "app.cplid_gpu.yaml").read_text(encoding="utf-8")


class RenderVariantTests(unittest.TestCase):
    def render(self, count: int = 2, **overrides):
        classes = [f"class-{index}" for index in range(count)]
        kwargs = dict(
            onnx_path="models/detector/variant.onnx",
            input_size=800,
            class_names=classes,
            model_version="variant-fcos-r50fpn-800",
            port=8899,
            title="变体测试",
        )
        kwargs.update(overrides)
        return render_variant(BASE, **kwargs), kwargs

    def test_cplid_specifics_are_gone(self):
        text, _ = self.render()
        for leftover in ("cplid_fcos_1024", "insulator", "cplid-fcos-r50fpn-1024", "port: 8851"):
            self.assertNotIn(leftover, text)

    def test_detect_block_parses_with_expected_values(self):
        text, kwargs = self.render(count=102)
        data = yaml.safe_load(text)
        detect = data["detect"]
        self.assertEqual(detect["detector"], "onnx")
        self.assertEqual(detect["onnx_path"], kwargs["onnx_path"])
        self.assertEqual(detect["input_size"], 800)
        self.assertEqual(detect["model_version"], kwargs["model_version"])
        self.assertEqual(detect["class_names"], kwargs["class_names"])
        self.assertEqual(detect["slice"]["size"], 800)
        self.assertFalse(detect["slice"]["enabled"])

    def test_class_order_is_preserved_for_long_lists(self):
        text, kwargs = self.render(count=102)
        data = yaml.safe_load(text)
        self.assertEqual(len(data["detect"]["class_names"]), 102)
        self.assertEqual(data["detect"]["class_names"][0], "class-0")
        self.assertEqual(data["detect"]["class_names"][-1], "class-101")
        self.assertIn("    - class-7", text, "102 类必须用块状列表，不能挤成一行")

    def test_short_lists_stay_inline(self):
        text, _ = self.render(count=2)
        self.assertIn("  class_names: [class-0, class-1]", text)
        self.assertNotIn("    - class-0", text)

    def test_unrelated_blocks_survive(self):
        text, _ = self.render()
        data = yaml.safe_load(text)
        for key in ("app", "paths", "retrieval", "llm", "tools", "severity", "eval", "server"):
            self.assertIn(key, data)
        self.assertEqual(data["server"]["host"], "127.0.0.1")
        self.assertEqual(data["server"]["port"], 8899)
        self.assertEqual(data["retrieval"]["rrf_k"], 60)
        self.assertEqual(data["app"]["name"], "低空之眼")

    def test_header_names_the_model_version(self):
        text, kwargs = self.render()
        head = text.split("app:", 1)[0]
        self.assertIn(kwargs["model_version"], head)
        self.assertIn(kwargs["onnx_path"], head)
        self.assertIn("class_names   = 2 类", head)

    def test_empty_classes_rejected(self):
        with self.assertRaises(ValueError):
            self.render(class_names=[])

    def test_non_positive_size_rejected(self):
        with self.assertRaises(ValueError):
            self.render(input_size=0)


class LoadMetadataTests(unittest.TestCase):
    def write(self, payload) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="variant-config-test-"))
        path = directory / "metadata.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_reads_classes_and_declared_size(self):
        path = self.write({"classes": ["smoke", "fire"], "preprocessing": {"input": [1, 3, 640, 640]}})
        classes, size = load_metadata(path)
        self.assertEqual(classes, ["smoke", "fire"])
        self.assertEqual(size, 640)

    def test_missing_size_is_none(self):
        path = self.write({"classes": ["smoke"]})
        classes, size = load_metadata(path)
        self.assertEqual(classes, ["smoke"])
        self.assertIsNone(size)

    def test_non_square_input_is_not_a_size(self):
        path = self.write({"classes": ["a"], "preprocessing": {"input": [1, 3, 640, 512]}})
        _, size = load_metadata(path)
        self.assertIsNone(size)

    def test_classes_required(self):
        path = self.write({"preprocessing": {"input": [1, 3, 640, 640]}})
        with self.assertRaises(ValueError):
            load_metadata(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_metadata(Path(tempfile.mkdtemp()) / "absent.json")


if __name__ == "__main__":
    unittest.main()
