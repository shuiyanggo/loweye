"""混合检测器：sidecar 优先，无离线结果文件时回落到真实 ONNX 推理。

设计动机：
- 演示图（data/assets/*.png）旁边有手工编排的 .detections.json，走 sidecar 保证
  演示剧情稳定、零时延、结果可复现（答辩现场逐条解释得清）；
- 用户上传的真实照片没有 .detections.json，此时用真实权重（FCOS 微调 ONNX）
  跑推理，让「上传现实照片」也能得到真实检测结论。

这样对外服务只有一份配置，却同时支持「演示图」与「真实照片」两种输入。
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import DetectResult
from .onnx_detector import OnnxDetector
from .sidecar import SidecarDetector


class HybridDetector:
    name = "hybrid"

    def __init__(self, sidecar: SidecarDetector, onnx: OnnxDetector) -> None:
        self.sidecar = sidecar
        self.onnx = onnx
        self.model_version = f"sidecar|{onnx.model_version}"

    def detect(self, image_path: Path, image_id: str) -> DetectResult:
        path = Path(image_path)
        if self.sidecar.sidecar_path_for(path).exists():
            # 有编排好的离线结果：走 sidecar，保持演示剧情稳定
            return self.sidecar.detect(path, image_id)
        # 无离线结果（上传的真实照片）：走真实 ONNX 推理
        return self.onnx.detect(path, image_id)
