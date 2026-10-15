"""切片推理与 NMS 的单测。

小目标漏检是无人机影像最要命的坑（绝缘子自爆在 4K 图里只有几十像素），
SAHI 切片是对策，所以「切片覆盖全图」「坐标映射回原图」「重叠区去重」必须测死。
"""

from __future__ import annotations

import unittest

from loweye.detect.slicing import compute_starts, iou, nms, offset_boxes, slice_image
from loweye.schemas import BBox, Detection


def det(cls: str, x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> Detection:
    return Detection(cls=cls, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), score=score)


class ComputeStartsTest(unittest.TestCase):
    def test_covers_tail(self) -> None:
        """最后一块必须贴到边上，否则图片右下角永远不被扫到。"""
        starts = compute_starts(total=1000, size=640, stride=512)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 360)  # 1000 - 640
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))

    def test_smaller_than_tile(self) -> None:
        self.assertEqual(compute_starts(total=300, size=640, stride=512), [0])


class SliceImageTest(unittest.TestCase):
    def test_tiles_cover_whole_image(self) -> None:
        """四个角落必须各自落在某块切片的矩形内。

        注意判定方式是**矩形包含**，不是网格采样命中：切片起点由 stride 决定，
        网格点（如 (1,1)）常常不等于任何采样点，用采样会把好实现判成失败。
        """
        tiles = slice_image(1600, 1000, 640, overlap=0.2)
        self.assertGreaterEqual(len(tiles), 6)

        def covered(x: float, y: float) -> bool:
            return any(t.x0 <= x <= t.x1 and t.y0 <= y <= t.y1 for t in tiles)

        for corner in ((0, 0), (1599, 0), (0, 999), (1599, 999)):
            self.assertTrue(covered(*corner), f"角落 {corner} 没被任何切片覆盖")

    def test_tile_size_never_exceeds_request(self) -> None:
        for t in slice_image(1600, 1000, 640, overlap=0.2):
            self.assertLessEqual(t.x1 - t.x0, 640)
            self.assertLessEqual(t.y1 - t.y0, 640)


class OffsetBoxesTest(unittest.TestCase):
    def test_maps_back_to_original_coordinates(self) -> None:
        tile = slice_image(1600, 1000, 640, overlap=0.2)[-1]
        moved = offset_boxes([det("insulator_broken", 10, 20, 110, 120)], tile)
        self.assertEqual(moved[0].bbox.x1, 10 + tile.x0)
        self.assertEqual(moved[0].bbox.y1, 20 + tile.y0)
        self.assertEqual(moved[0].slice_id, f"tile{tile.index}")

    def test_clamped_inside_tile(self) -> None:
        tile = slice_image(1600, 1000, 640, overlap=0.2)[-1]
        moved = offset_boxes([det("x", 0, 0, 9999, 9999)], tile)
        self.assertLessEqual(moved[0].bbox.x2, tile.x1)
        self.assertLessEqual(moved[0].bbox.y2, tile.y1)

    def test_does_not_mutate_input(self) -> None:
        tile = slice_image(1600, 1000, 640)[0]
        src = det("x", 1, 2, 3, 4)
        offset_boxes([src], tile)
        self.assertEqual((src.bbox.x1, src.bbox.y1), (1, 2))


class IouAndNmsTest(unittest.TestCase):
    def test_iou_basics(self) -> None:
        a = BBox(x1=0, y1=0, x2=10, y2=10)
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, BBox(x1=20, y1=20, x2=30, y2=30)), 0.0)

    def test_duplicates_merged_keep_highest(self) -> None:
        """重叠区同一目标被两块切片各检一次 → 只留分数高的那个。"""
        dets = [
            det("insulator_broken", 100, 100, 200, 200, score=0.72),
            det("insulator_broken", 104, 102, 203, 201, score=0.93),
        ]
        kept = nms(dets, iou_threshold=0.5)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].score, 0.93)

    def test_different_classes_not_suppressed(self) -> None:
        dets = [det("a", 0, 0, 100, 100), det("b", 0, 0, 100, 100)]
        self.assertEqual(len(nms(dets, iou_threshold=0.5)), 2)

    def test_far_apart_kept(self) -> None:
        dets = [det("a", 0, 0, 50, 50), det("a", 400, 400, 450, 450)]
        self.assertEqual(len(nms(dets, iou_threshold=0.5)), 2)

    def test_empty(self) -> None:
        self.assertEqual(nms([]), [])


if __name__ == "__main__":
    unittest.main()
