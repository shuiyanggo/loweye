"""sidecar 检测器：读取与图片同名的 .detections.json。

用途有两个，都是真实的工程用途：
1. 批量离线推理的结果落库后再读入（生产环境常见形态，推理与业务解耦）
2. 在没有训练权重 / 没有 GPU 的机器上，保证识别→诊断→派单→归档
   全链路可运行、可回归测试、可做消融实验

文件格式：
    {
      "image_id": "demo_tower_01",
      "model_version": "rtdetr-l@demo",
      "detections": [
        {"cls": "insulator_broken", "bbox": [x1, y1, x2, y2], "score": 0.93}
      ]
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..schemas import BBox, Detection, DetectResult


class SidecarDetector:
    name = "sidecar"

    def __init__(self, assets_dir: Path, *, conf_threshold: float = 0.35) -> None:
        self.assets_dir = Path(assets_dir)
        self.conf_threshold = conf_threshold
        self.model_version = "sidecar"

    def sidecar_path_for(self, image_path: Path) -> Path:
        return image_path.with_suffix(".detections.json")

    def detect(self, image_path: Path, image_id: str) -> DetectResult:
        started = time.perf_counter()
        path = self.sidecar_path_for(Path(image_path))
        warnings: list[str] = []
        detections: list[Detection] = []
        model_version = self.model_version

        if not path.exists():
            warnings.append(
                f"未找到检测结果文件 {path.name}；当前检测后端为 sidecar，"
                "因此本次结果为空（真机部署应改为 onnx 后端或先跑批量推理）"
            )
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            model_version = str(data.get("model_version", model_version))
            image_id = str(data.get("image_id", image_id))
            for i, item in enumerate(data.get("detections", [])):
                score = float(item.get("score", 0.0))
                if score < self.conf_threshold:
                    continue
                bbox = item["bbox"]
                detections.append(
                    Detection(
                        cls=str(item["cls"]),
                        bbox=BBox(x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3])),
                        score=score,
                        slice_id=item.get("slice_id"),
                    )
                )

        detections.sort(key=lambda d: (-d.score, d.cls))
        return DetectResult(
            image_id=image_id,
            detections=detections,
            detector=self.name,
            model_version=model_version,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            warnings=warnings,
        )
