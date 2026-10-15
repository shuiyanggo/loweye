"""ONNX 解码分支的单测（torchvision 导出的三输出 boxes / labels / scores）。

为什么单测这条路径：`loweye/detect/onnx_detector.py` 要同时认三种导出格式，
其中 torchvision 系的标签**从 1 开始**（0 是背景），映射类别名时忘记减一不会报错，
只会安静地把 `defect` 判成 `insulator` —— 这种错在演示里看不出来，必须测死。

不加载真实 ONNX：只用 `object.__new__` 造一个带字段的实例，直接调解码函数。
"""

from __future__ import annotations

import unittest

import numpy as np

from loweye.detect.onnx_detector import OnnxDetector


def make_detector() -> OnnxDetector:
    det = object.__new__(OnnxDetector)
    det.class_names = ["insulator", "defect"]
    det.conf_threshold = 0.35
    det.iou_threshold = 0.5
    det._layout = "tv_dets"
    det._out_names = ["boxes", "labels", "scores"]
    return det


class TvDetsDecodeTest(unittest.TestCase):
    def test_label_is_one_based(self) -> None:
        """标签 1 必须是 insulator，标签 2 必须是 defect（不能直接按索引取）。"""
        det = make_detector()
        boxes = np.array([[10.0, 20.0, 110.0, 70.0], [200.0, 30.0, 260.0, 80.0]], dtype="float32")
        labels = np.array([1, 2], dtype="int64")
        scores = np.array([0.9, 0.8], dtype="float32")
        out = det._decode_tv_dets([boxes, labels, scores], scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual([d.cls for d in out], ["insulator", "defect"])

    def test_letterbox_is_undone(self) -> None:
        """坐标要减填充、除缩放，回到原图尺度。"""
        det = make_detector()
        boxes = np.array([[20.0, 40.0, 220.0, 140.0]], dtype="float32")
        labels = np.array([1], dtype="int64")
        scores = np.array([0.95], dtype="float32")
        out = det._decode_tv_dets([boxes, labels, scores], scale=0.5, pad_x=10, pad_y=20)
        box = out[0].bbox
        self.assertAlmostEqual(box.x1, 20.0)   # (20 - 10) / 0.5
        self.assertAlmostEqual(box.y1, 40.0)   # (40 - 20) / 0.5
        self.assertAlmostEqual(box.x2, 420.0)  # (220 - 10) / 0.5
        self.assertAlmostEqual(box.y2, 240.0)  # (140 - 20) / 0.5

    def test_low_score_is_dropped(self) -> None:
        det = make_detector()
        boxes = np.array([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]], dtype="float32")
        labels = np.array([1, 2], dtype="int64")
        scores = np.array([0.20, 0.80], dtype="float32")
        out = det._decode_tv_dets([boxes, labels, scores], scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].cls, "defect")

    def test_unknown_label_falls_back(self) -> None:
        """标签超出 class_names 时用 class_N 占位，不抛异常也不静默错标。"""
        det = make_detector()
        boxes = np.array([[1.0, 1.0, 9.0, 9.0]], dtype="float32")
        labels = np.array([7], dtype="int64")
        scores = np.array([0.7], dtype="float32")
        out = det._decode_tv_dets([boxes, labels, scores], scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual(out[0].cls, "class_7")

    def test_output_names_are_not_trusted(self) -> None:
        """输出名会串位：名为 labels 的其实是 float32 分数、名为 scores 的其实是 int64 标签。

        实测 torchvision 2.x 导出的 FCOS 就是这样 —— 按名字取会把标签 2 当成分数 2.0，
        pydantic 直接以「score 必须 ≤ 1」拒绝。必须按 dtype 认路。
        """
        det = make_detector()
        boxes = np.array([[10.0, 20.0, 110.0, 70.0]], dtype="float32")
        lying_labels = np.array([0.9], dtype="float32")   # 名字叫 labels，装的是分数
        lying_scores = np.array([2], dtype="int64")       # 名字叫 scores，装的是标签
        out = det._decode_tv_dets([boxes, lying_labels, lying_scores], scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual([d.cls for d in out], ["defect"])
        self.assertLessEqual(out[0].score, 1.0)
        self.assertAlmostEqual(out[0].score, 0.9)

    def test_score_then_label_order(self) -> None:
        """run() 返回顺序是 boxes → scores → labels 时也要解对（不能按位置认）。"""
        det = make_detector()
        det._out_names = ["boxes", "scores", "labels"]
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype="float32")
        scores = np.array([0.9, 0.8], dtype="float32")
        labels = np.array([1, 2], dtype="int64")
        out = det._decode_tv_dets([boxes, scores, labels], scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual([d.cls for d in out], ["insulator", "defect"])
        self.assertEqual([round(d.score, 2) for d in out], [0.9, 0.8])

    def test_shape_mismatch_raises(self) -> None:
        """形状不认识就直接报错，绝不猜。"""
        det = make_detector()
        with self.assertRaises(RuntimeError):
            det._decode_tv_dets(
                [np.zeros((3, 3), dtype="float32"), np.zeros((2,), dtype="int64"), np.zeros((2,), dtype="float32")],
                scale=1.0, pad_x=0, pad_y=0,
            )

    def test_batch_dim_is_squeezed(self) -> None:
        """带 batch 维的导出（(1,N,4) / (1,N) / (1,N)）也要能解。"""
        det = make_detector()
        boxes = np.zeros((1, 2, 4), dtype="float32")
        boxes[0, 0] = [0.0, 0.0, 10.0, 10.0]
        boxes[0, 1] = [20.0, 20.0, 30.0, 30.0]
        labels = np.array([[1, 2]], dtype="int64")
        scores = np.array([[0.9, 0.9]], dtype="float32")
        out = det._decode_tv_dets([boxes, labels, scores], scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual([d.cls for d in out], ["insulator", "defect"])


if __name__ == "__main__":
    unittest.main()
