"""检测器接口与工厂。

三个后端：
  sidecar —— 读取离线批量推理产出的检测结果（真实部署里推理本就在边缘端或
             批处理任务里跑完，把结果落库是最常见做法；也是无权重环境下
             保证全链路可跑、可评测的方式）
  onnx    —— 真实模型推理（需提供 .onnx 权重与 onnxruntime / Pillow）
  null    —— 空检测器，用于接口自测与消融实验

换后端时上层代码零改动，因为统一走 DetectResult 这个结构化契约。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Config
from ..schemas import DetectResult


@runtime_checkable
class Detector(Protocol):
    name: str
    model_version: str

    def detect(self, image_path: Path, image_id: str) -> DetectResult:
        ...


def build_detector(cfg: Config) -> tuple[Detector, str | None]:
    """返回 (检测器, 降级原因)。降级原因非空表示没走成配置指定的后端。"""
    kind = str(cfg.get("detect.detector", "sidecar")).lower()
    assets = cfg.path("assets")
    slice_cfg = cfg.get("detect.slice", {}) or {}
    conf = float(cfg.get("detect.conf_threshold", 0.35))

    if kind == "onnx":
        from .onnx_detector import OnnxDetector

        try:
            detector = OnnxDetector(
                onnx_path=cfg.get("detect.onnx_path"),
                conf_threshold=conf,
                iou_threshold=float(cfg.get("detect.nms_iou", 0.5)),
                input_size=int(cfg.get("detect.input_size", 640)),
                slice_enabled=bool(slice_cfg.get("enabled", True)),
                slice_size=int(slice_cfg.get("size", 640)),
                slice_overlap=float(slice_cfg.get("overlap", 0.2)),
                model_version=str(cfg.get("detect.model_version", "onnx")),
                class_names=cfg.get("detect.class_names"),
            )
            return detector, None
        except Exception as exc:
            from .sidecar import SidecarDetector

            return (
                SidecarDetector(assets, conf_threshold=conf),
                f"ONNX 检测器不可用（{type(exc).__name__}: {exc}），已降级为 sidecar 离线结果",
            )

    if kind == "hybrid":
        from .hybrid import HybridDetector
        from .onnx_detector import OnnxDetector
        from .sidecar import SidecarDetector

        sidecar = SidecarDetector(assets, conf_threshold=conf)
        try:
            onnx = OnnxDetector(
                onnx_path=cfg.get("detect.onnx_path"),
                conf_threshold=conf,
                iou_threshold=float(cfg.get("detect.nms_iou", 0.5)),
                input_size=int(cfg.get("detect.input_size", 640)),
                slice_enabled=bool(slice_cfg.get("enabled", True)),
                slice_size=int(slice_cfg.get("size", 640)),
                slice_overlap=float(slice_cfg.get("overlap", 0.2)),
                model_version=str(cfg.get("detect.model_version", "onnx")),
                class_names=cfg.get("detect.class_names"),
            )
            return HybridDetector(sidecar, onnx), None
        except Exception as exc:
            return (
                sidecar,
                f"ONNX 检测器不可用（{type(exc).__name__}: {exc}），上传照片将退回 sidecar 空结果",
            )

    if kind == "torch":
        from .torch_detector import TorchDetector

        try:
            detector = TorchDetector(
                weights_path=cfg.get("detect.torch_path"),
                class_names=cfg.get("detect.class_names"),
                conf_threshold=conf,
                iou_threshold=float(cfg.get("detect.nms_iou", 0.5)),
                input_size=int(cfg.get("detect.input_size", 640)),
                slice_enabled=bool(slice_cfg.get("enabled", True)),
                slice_size=int(slice_cfg.get("size", 640)),
                slice_overlap=float(slice_cfg.get("overlap", 0.2)),
                model_version=str(cfg.get("detect.model_version", "torch")),
                threads=cfg.get("detect.threads"),
                infer_min_size=cfg.get("detect.infer_min_size"),
                infer_max_size=cfg.get("detect.infer_max_size"),
            )
            return detector, None
        except Exception as exc:
            from .sidecar import SidecarDetector

            return (
                SidecarDetector(assets, conf_threshold=conf),
                f"PyTorch 检测器不可用（{type(exc).__name__}: {exc}），已降级为 sidecar 离线结果",
            )

    if kind == "null":
        from .null_detector import NullDetector

        return NullDetector(), None

    from .sidecar import SidecarDetector

    return SidecarDetector(assets, conf_threshold=conf), None
