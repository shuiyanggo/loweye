"""PyTorch 检测后端：跑在 CPLID 上微调出来的权重。

为什么要有这条后端（而不是只用 ONNX）：torchvision 的检测模型（Faster R-CNN /
RetinaNet / FCOS）BSD-3 许可干净、自带 COCO 预训练权重，能在只有 CPU 的机器上做
小规模微调；但它们的后处理（ROI Align + NMS）导出 ONNX 比较麻烦。所以策略是
「训练与验证走 torch，需要上线再单独导出 ONNX」——先在真实数据上拿到每类漏检率，
不让导出这一步卡住评测。

与 OnnxDetector 一致的约定：坐标回到原图、坐标越界裁剪、conf 阈值过滤、
可选 SAHI 切片（航拍缺陷目标小，整幅缩放容易漏）。
"""

from __future__ import annotations

import time
from pathlib import Path

from ..schemas import BBox, Detection, DetectResult
from .slicing import nms, offset_boxes, slice_image


class TorchDetector:
    name = "torch"

    def __init__(
        self,
        *,
        weights_path: str | None,
        class_names: list[str] | None = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        input_size: int = 640,
        slice_enabled: bool = True,
        slice_size: int = 640,
        slice_overlap: float = 0.2,
        model_version: str = "torch-cplid",
        threads: int | None = None,
        infer_min_size: int | None = None,
        infer_max_size: int | None = None,
    ) -> None:
        if not weights_path:
            raise ValueError("未配置 detect.torch_path（没有权重就没有推理）")
        path = Path(weights_path)
        if not path.is_absolute():
            from ..config import PROJECT_ROOT

            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"PyTorch 权重不存在：{path}")

        import torch  # 延迟导入：没装 torch 的机器照常跑 sidecar
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        if threads:
            torch.set_num_threads(int(threads))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        classes = list(class_names or payload.get("classes") or ["insulator", "defect"])
        arch = payload.get("arch", "fasterrcnn_mobilenet_v3_large_fpn")
        if arch != "fasterrcnn_mobilenet_v3_large_fpn":
            raise ValueError(f"暂不支持的架构：{arch}")

        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None, num_classes=len(classes) + 1
        )
        # 推理尺度可以单独设：torchvision 默认 min_size=800/max_size=1333，与训练时的
        # 416/640 不一致；小目标缺陷对推理尺度很敏感，所以留出显式开关便于做对照。
        if infer_min_size:
            model.transform.min_size = (int(infer_min_size),)
        if infer_max_size:
            model.transform.max_size = int(infer_max_size)
        model.load_state_dict(payload["model"])
        model.eval()

        self._torch = torch
        self._model = model
        self.weights_path = path
        self.class_names = classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.slice_enabled = slice_enabled
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.model_version = payload.get("model_version", model_version)
        self.trained = payload.get("trained", {})
        _ = FasterRCNN_MobileNet_V3_Large_FPN_Weights  # 仅用于说明权重来源，避免未使用告警

    # ------------------------------------------------------------------ 推理

    def _infer_tile(self, tile_arr) -> list[Detection]:
        torch = self._torch
        # tile_arr 来自 PIL 的 RGB，torchvision 检测模型要的就是 RGB，不要反转通道
        tensor = torch.from_numpy(tile_arr).permute(2, 0, 1).float().div(255.0)
        with torch.no_grad():
            out = self._model([tensor])[0]
        dets: list[Detection] = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            s = float(score)
            if s < self.conf_threshold:
                continue
            idx = int(label) - 1  # 0 是背景
            if idx < 0 or idx >= len(self.class_names):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append(
                Detection(
                    cls=self.class_names[idx],
                    bbox=BBox(x1=max(0.0, x1), y1=max(0.0, y1), x2=max(0.0, x2), y2=max(0.0, y2)),
                    score=s,
                )
            )
        return dets

    def detect(self, image_path: Path, image_id: str) -> DetectResult:
        import numpy as np
        from PIL import Image

        image_path = Path(image_path)
        t0 = time.perf_counter()
        with Image.open(image_path) as im:
            full = np.asarray(im.convert("RGB"))
        height, width = full.shape[:2]

        if self.slice_enabled:
            tiles = slice_image(width, height, self.slice_size, self.slice_overlap)
            collected: list[Detection] = []
            for tile in tiles:
                patch = full[tile.y0:tile.y1, tile.x0:tile.x1]
                if patch.size == 0:
                    continue
                collected.extend(offset_boxes(self._infer_tile(patch), tile))
            detections = nms(collected, self.iou_threshold)
        else:
            tiles = []
            detections = self._infer_tile(full)

        return DetectResult(
            image_id=image_id,
            detections=detections,
            detector=self.name,
            model_version=self.model_version,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
            slice_count=len(tiles),
            warnings=[],
        )
