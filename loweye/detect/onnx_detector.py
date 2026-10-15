"""ONNX 推理后端（可选增强，需要 onnxruntime + Pillow + numpy）。

设计取舍说清楚：
- 权重不随仓库分发。没有 onnx_path 或没装 onnxruntime 时，工厂会明确降级到
  sidecar 并把原因写进 PipelineResult.degradations，不会假装推理成功。
- 切片推理在这里落地：航拍缺陷目标小，整幅缩放必然漏检，所以按瓦片逐块推理，
  再把框映射回原图坐标做跨片 NMS。
- 只实现了三类导出格式的解析：
  yolo_head   单输出 (1, 4+nc, N) 或 (1, N, 4+nc)，类别分在一张张量里；
  dets_labels 双输出 dets(1, N, 5)=[x1,y1,x2,y2,score] + labels(1, N)，NMS 已在图内
              （MMDeploy / OpenMMLab 系，如 RTMDet）；
  tv_dets     三输出 boxes(N,4) + labels(N) + scores(N)，NMS 已在图内
              （torchvision 系，如 FCOS / RetinaNet / Faster R-CNN；标签从 1 开始，0 是背景）。
  其他导出格式（如 RT-DETR 的 (1, N, 6)）需要按需在 `_decode` 里补分支 ——
  遇到不认识的输出形状时直接报错并打印真实形状，不猜。
"""

from __future__ import annotations

import time
from pathlib import Path

from ..schemas import BBox, Detection, DetectResult
from .slicing import nms, offset_boxes, slice_image

# 类别顺序必须与训练时的 data.yaml 一致；可由 config/app.yaml 的 detect.class_names 覆盖
DEFAULT_CLASS_NAMES = [
    "insulator_broken",
    "insulator_polluted",
    "fitting_corrosion",
    "conductor_strand_break",
    "tower_tilt",
    "foreign_object",
    "vegetation_overreach",
    "crop_disease",
    "crop_pest",
    "fire_smoke",
]


class OnnxDetector:
    name = "onnx"

    def __init__(
        self,
        *,
        onnx_path: str | None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        input_size: int = 640,
        slice_enabled: bool = True,
        slice_size: int = 640,
        slice_overlap: float = 0.2,
        model_version: str = "onnx",
        class_names: list[str] | None = None,
    ) -> None:
        if not onnx_path:
            raise ValueError("未配置 detect.onnx_path（没有权重就没有推理）")
        path = Path(onnx_path)
        if not path.is_absolute():
            from ..config import PROJECT_ROOT

            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"ONNX 权重不存在：{path}")

        try:
            import numpy as np  # noqa: F401
            import onnxruntime  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception as exc:  # pragma: no cover - 取决于是否装了可选依赖
            raise ImportError(
                f"缺少可选依赖（{type(exc).__name__}: {exc}）；"
                "请先 pip install onnxruntime Pillow numpy"
            ) from exc

        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # 两种导出格式要分开认：
        #   yolo_head   ：单输出 (1, 4+nc, N) 或 (1, N, 4+nc)，原始头，类别分在一张张量里；
        #   dets_labels ：双输出 dets(1, N, 5)=[x1,y1,x2,y2,score] + labels(1, N)，NMS 已在图内，
        #                 这是 MMDeploy / OpenMMLab 系（RTMDet 等）的标准导出。
        self._out_names = [o.name for o in self.session.get_outputs()]
        out_names = set(self._out_names)
        if {"dets", "labels"} <= out_names:
            self._layout = "dets_labels"
        elif {"boxes", "labels", "scores"} <= out_names:
            self._layout = "tv_dets"
        else:
            self._layout = "yolo_head"
        self.onnx_path = path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.slice_enabled = slice_enabled
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.model_version = model_version
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)

    # ------------------------------------------------------------ 预处理

    def _letterbox(self, arr):
        """等比例缩放 + 灰边填充到 input_size，返回 (张量, 缩放比, 左右填充, 上下填充)。"""
        import numpy as np
        from PIL import Image

        h, w = arr.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        tensor = canvas.astype("float32").transpose(2, 0, 1)[None] / 255.0
        return tensor, scale, pad_x, pad_y

    # ------------------------------------------------------------ 解码

    def _decode(self, raw, scale: float, pad_x: int, pad_y: int) -> list[Detection]:
        import numpy as np

        out = np.asarray(raw)
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2:
            raise RuntimeError(f"无法解析的模型输出形状 {np.asarray(raw).shape}，请在 _decode 中补充分支")
        # YOLO 导出：(4+nc, N) 或 (N, 4+nc)
        if out.shape[0] < out.shape[1]:
            out = out.T
        if out.shape[1] < 5:
            raise RuntimeError(f"无法解析的模型输出形状 {np.asarray(raw).shape}，请在 _decode 中补充分支")

        boxes, scores_all = out[:, :4], out[:, 4:]
        cls_idx = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(cls_idx)), cls_idx]
        keep = scores >= self.conf_threshold
        if not keep.any():
            return []

        dets: list[Detection] = []
        for (cx, cy, bw, bh), score, ci in zip(boxes[keep], scores[keep], cls_idx[keep]):
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            cls_name = self.class_names[int(ci)] if int(ci) < len(self.class_names) else f"class_{int(ci)}"
            dets.append(
                Detection(
                    cls=cls_name,
                    bbox=BBox(x1=max(0.0, x1), y1=max(0.0, y1), x2=max(0.0, x2), y2=max(0.0, y2)),
                    score=float(score),
                )
            )
        return dets

    def _decode_dets_labels(self, outputs, scale: float, pad_x: int, pad_y: int) -> list[Detection]:
        """解析 dets(1,N,5) + labels(1,N) 这种双输出（图内已做 NMS）。

        坐标同样在 letterbox 后的输入尺度上，要减填充、除缩放才能回到原图。
        """
        import numpy as np

        dets = np.asarray(outputs[0])
        labels = np.asarray(outputs[1])
        if dets.ndim == 3:
            dets = dets[0]
        if labels.ndim == 2:
            labels = labels[0]
        if dets.ndim != 2 or dets.shape[1] < 5:
            raise RuntimeError(
                f"无法解析的模型输出形状 dets={np.asarray(outputs[0]).shape} "
                f"labels={np.asarray(outputs[1]).shape}，请在 _decode 系列里补分支"
            )

        dets_out: list[Detection] = []
        for row, ci in zip(dets, labels):
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            score = float(row[4])
            if score < self.conf_threshold:
                continue
            idx = int(ci)
            cls_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            dets_out.append(
                Detection(
                    cls=cls_name,
                    bbox=BBox(
                        x1=max(0.0, (x1 - pad_x) / scale),
                        y1=max(0.0, (y1 - pad_y) / scale),
                        x2=max(0.0, (x2 - pad_x) / scale),
                        y2=max(0.0, (y2 - pad_y) / scale),
                    ),
                    score=score,
                )
            )
        return dets_out

    def _split_tv_outputs(self, outputs):
        """分辨 boxes / labels / scores —— 只信 dtype 与形状，不信输出名。

        实测（torchvision 2.x 导出的 FCOS）：图里名字会串位 —— 名为 `labels` 的输出是
        float32（其实装的是分数），名为 `scores` 的输出是 int64（其实装的是标签）。
        按名字取会把标签当分数（症状：score=2.0 越界，pydantic 直接拒绝）。
        所以主路径按 dtype 认路，名字只在 dtype 不典型时作为退化路径。
        """
        import numpy as np

        arrays = [np.asarray(a) for a in outputs]
        boxes = next((a for a in arrays if a.ndim >= 2 and a.shape[-1] == 4), None)
        rest = [a for a in arrays if a is not boxes]
        labels = next((a for a in rest if np.issubdtype(a.dtype, np.integer)), None)
        scores = next((a for a in rest if a is not labels and np.issubdtype(a.dtype, np.floating)), None)
        if boxes is not None and labels is not None and scores is not None:
            return boxes, labels, scores

        by_name = {name: arr for name, arr in zip(getattr(self, "_out_names", ()), arrays)}
        if {"boxes", "labels", "scores"} <= set(by_name):
            return by_name["boxes"], by_name["labels"], by_name["scores"]
        raise RuntimeError(
            f"无法分辨 boxes/labels/scores：输出名={getattr(self, '_out_names', ())}，"
            f"形状={[a.shape for a in arrays]}，dtype={[str(a.dtype) for a in arrays]}，"
            f"请在 _decode 系列里补分支"
        )

    def _decode_tv_dets(self, outputs, scale: float, pad_x: int, pad_y: int) -> list[Detection]:
        """解析 torchvision 系导出的三输出 boxes / labels / scores（图内已做 NMS）。

        两点与 YOLO 系不同，必须记住：
        1. 标签从 1 开始（0 是背景），映射类别名要减一；
        2. 坐标同样在 letterbox 后的输入尺度上，要减填充、除缩放才能回到原图。
        """
        import numpy as np

        boxes, labels, scores = self._split_tv_outputs(outputs)
        if boxes.ndim == 3:      # 导出时带了 batch 维
            boxes = boxes[0]
        if labels.ndim >= 2:
            labels = labels.reshape(-1)
        if scores.ndim >= 2:
            scores = scores.reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] < 4 or len(boxes) != len(labels):
            raise RuntimeError(
                f"无法解析的模型输出形状 boxes={boxes.shape} labels={labels.shape} "
                f"scores={scores.shape}，请在 _decode 系列里补分支"
            )

        dets_out: list[Detection] = []
        for (x1, y1, x2, y2), score, ci in zip(boxes[:, :4], scores, labels):
            if float(score) < self.conf_threshold:
                continue
            idx = int(ci) - 1
            cls_name = self.class_names[idx] if 0 <= idx < len(self.class_names) else f"class_{int(ci)}"
            dets_out.append(
                Detection(
                    cls=cls_name,
                    bbox=BBox(
                        x1=max(0.0, (float(x1) - pad_x) / scale),
                        y1=max(0.0, (float(y1) - pad_y) / scale),
                        x2=max(0.0, (float(x2) - pad_x) / scale),
                        y2=max(0.0, (float(y2) - pad_y) / scale),
                    ),
                    score=float(score),
                )
            )
        return dets_out

    def _infer_tile(self, tile_arr) -> list[Detection]:
        tensor, scale, pad_x, pad_y = self._letterbox(tile_arr)
        outputs = self.session.run(None, {self.input_name: tensor})
        if self._layout == "dets_labels":
            return self._decode_dets_labels(outputs, scale, pad_x, pad_y)
        if self._layout == "tv_dets":
            return self._decode_tv_dets(outputs, scale, pad_x, pad_y)
        return self._decode(outputs[0], scale, pad_x, pad_y)

    # ------------------------------------------------------------ 对外

    def detect(self, image_path: Path, image_id: str) -> DetectResult:
        import numpy as np
        from PIL import Image

        image_path = Path(image_path)
        t0 = time.perf_counter()
        warnings: list[str] = []

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

        if not self.class_names:
            warnings.append("类别名称为空，输出将使用 class_N 占位")

        return DetectResult(
            image_id=image_id,
            detections=detections,
            detector=self.name,
            model_version=self.model_version,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
            slice_count=len(tiles),
            warnings=warnings,
        )
