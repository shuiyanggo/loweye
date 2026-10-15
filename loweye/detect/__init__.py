"""M1 识别层：可插拔目标检测器 + 小目标切片推理。"""

from .base import Detector, build_detector
from .null_detector import NullDetector
from .onnx_detector import OnnxDetector
from .sidecar import SidecarDetector
from .slicing import Tile, nms, slice_image

__all__ = [
    "Detector",
    "build_detector",
    "NullDetector",
    "OnnxDetector",
    "SidecarDetector",
    "Tile",
    "slice_image",
    "nms",
]
