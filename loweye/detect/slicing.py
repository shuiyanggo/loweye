"""小目标切片推理与跨切片去重。

航拍绝缘子缺陷在原图里往往只有几十像素，整幅缩放到 640 必然漏检。
标准做法是切片（SAHI 思路）：把原图切成带重叠的瓦片，逐片推理，
再把各片的框映射回原图坐标去重。

本模块是纯函数实现，与具体模型无关，因此可以用假检测器做单测 ——
切片边界、坐标映射、跨片去重这三件事必须是被验证过的，而不是"应该没问题"。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import BBox, Detection


@dataclass(frozen=True)
class Tile:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int


def compute_starts(total: int, size: int, stride: int) -> list[int]:
    """给定图像边长、切片边长、步长，返回切片起点列表（保证覆盖到边界）。"""
    if size <= 0:
        raise ValueError("切片边长必须为正")
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] != total - size:
        starts.append(total - size)  # 补上右/下边界，避免边缘被裁掉
    return sorted(set(starts))


def slice_image(width: int, height: int, size: int, overlap: float = 0.2) -> list[Tile]:
    """把 width×height 的图像切成覆盖全图的瓦片，overlap 取 0~0.9。"""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap 必须落在 [0, 1) 区间")
    stride = max(1, int(round(size * (1.0 - overlap))))
    xs = compute_starts(width, size, stride)
    ys = compute_starts(height, size, stride)
    tiles: list[Tile] = []
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            tiles.append(
                Tile(index=yi * len(xs) + xi, x0=x0, y0=y0, x1=min(x0 + size, width), y1=min(y0 + size, height))
            )
    return tiles


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """按类别做贪心 NMS，用于合并切片重叠区里重复检出的同一目标。"""
    if not detections:
        return []
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.cls, []).append(det)

    kept: list[Detection] = []
    for cls, dets in by_class.items():
        ordered = sorted(dets, key=lambda d: (-d.score, d.bbox.x1, d.bbox.y1))
        survivors: list[Detection] = []
        for cand in ordered:
            if all(iou(cand.bbox, s.bbox) <= iou_threshold for s in survivors):
                survivors.append(cand)
        kept.extend(survivors)
    kept.sort(key=lambda d: (-d.score, d.cls, d.bbox.x1, d.bbox.y1))
    return kept


def offset_boxes(detections: list[Detection], tile: Tile) -> list[Detection]:
    """把切片内坐标映射回原图坐标。"""
    out: list[Detection] = []
    for det in detections:
        b = det.bbox
        out.append(
            det.model_copy(
                update={
                    "bbox": BBox(
                        x1=min(max(b.x1 + tile.x0, 0.0), float(tile.x1)),
                        y1=min(max(b.y1 + tile.y0, 0.0), float(tile.y1)),
                        x2=min(max(b.x2 + tile.x0, 0.0), float(tile.x1)),
                        y2=min(max(b.y2 + tile.y0, 0.0), float(tile.y1)),
                    ),
                    "slice_id": f"tile{tile.index}",
                }
            )
        )
    return out
