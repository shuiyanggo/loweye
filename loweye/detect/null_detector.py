"""空检测器：不产出任何检测框。

用途：接口自测、消融实验里的对照基线（方案 v2 M6 的基线 A 需要一个
"只跑检测"的极简配置，null 后端用来验证下游在零检测时的行为是否健壮）。
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import DetectResult


class NullDetector:
    name = "null"
    model_version = "null"

    def detect(self, image_path: Path, image_id: str) -> DetectResult:
        return DetectResult(
            image_id=image_id,
            detections=[],
            detector=self.name,
            model_version=self.model_version,
            elapsed_ms=0.0,
            warnings=["检测后端为 null，本次不产出任何检测框"],
        )
