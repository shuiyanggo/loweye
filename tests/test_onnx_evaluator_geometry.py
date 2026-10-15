"""Lock the ONNX evaluator's coordinate contract.

The evaluator scores the exported graph, whose boxes live on the letterbox
canvas, against manifest ground truth that lives in original-image pixels. If
that transform disappears, IoUs collapse and the independent mAP falls by two
orders of magnitude while preprocessing parity still looks perfect - which is
exactly the failure this file guards against.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVALUATOR = PROJECT / "tools" / "gpu" / "evaluate_onnx_detector.py"
TRAINER = PROJECT / "tools" / "gpu" / "train_inspection_fcos.py"

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class GeometryContractTest(unittest.TestCase):
    def test_evaluator_borrows_the_training_transform(self):
        text = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("def geometry_transform(", text)
        self.assertIn("trainer.transform_boxes(boxes, width, height, size, False)", text)
        self.assertIn('geometry_transform(sample["boxes"], width, height, size)', text)
        self.assertNotIn('zip(sample["boxes"], sample["labels"])', text)

    def test_training_dataset_still_transforms_its_targets(self):
        text = TRAINER.read_text(encoding="utf-8")
        self.assertIn('boxes = transform_boxes(sample["boxes"], width, height, self.size, flip)', text)

    @unittest.skipUnless(TORCH_AVAILABLE, "geometry_transform imports the training module, which needs torch")
    def test_geometry_transform_matches_letterbox_arithmetic(self):
        import sys

        sys.path.insert(0, str(PROJECT / "tools" / "gpu"))
        import evaluate_onnx_detector as evaluator

        boxes = [[10.0, 20.0, 110.0, 120.0], [0.0, 0.0, 4.0, 8.0]]
        width, height, size = 400, 200, 800
        scale = size / max(width, height)
        nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
        px, py = (size - nw) // 2, (size - nh) // 2
        sx, sy = nw / width, nh / height
        expected = [[x1 * sx + px, y1 * sy + py, x2 * sx + px, y2 * sy + py]
                    for x1, y1, x2, y2 in boxes]
        got = evaluator.geometry_transform(boxes, width, height, size)
        self.assertEqual(len(got), len(expected))
        for produced, wanted in zip(got, expected):
            for left, right in zip(produced, wanted):
                self.assertAlmostEqual(left, right, places=9)


if __name__ == "__main__":
    unittest.main()
