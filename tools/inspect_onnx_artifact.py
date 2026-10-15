"""用项目自己的后端探一个已交付的 ONNX + metadata 对（本机可用性轻量探针）。

为什么单独有它：本机常常只有 `{task}_fcos_{size}.onnx` 和 `{task}_fcos_metadata.json`
两个文件——manifest 有十几 MB、留在 GPU 主机上——但仍然需要对「这个文件在本机到底
跑不跑得起来」给出有证据的回答。`tools/verify_delivered_detectors.py` 是完整交付闸门
（SHA256 收据、manifest 类别交叉核对、best/last 两个权重）；本脚本是暂存/排练阶段用的
单件探针：读合同（输入输出名、形状、dtype）→ 用 `loweye.detect.onnx_detector.OnnxDetector`
真正推理一次 → 报告布局判定、检测框数量、类别下标是否越界。

输入边长不靠猜：先读 ONNX 自己声明的输入形状，再和 metadata 的
`preprocessing.input`（例如 crop_pest 是 [1,3,800,800]）对照，两者不一致就直接报缺口——
踩过的坑：metadata 里没有 `input_size` 字段，按 640 兜底会让 800 的模型抛
`Got invalid dimensions for input: images ... index: 2 Got: 640 Expected: 800`。

    python tools/inspect_onnx_artifact.py \
        --onnx out/inspection_gpu/_localcheck/crop_pest_fcos_800.onnx \
        --metadata out/inspection_gpu/_localcheck/crop_pest_fcos_metadata.json

    # 换真实图片、按生产阈值（0.35）看结果：
    python tools/inspect_onnx_artifact.py --onnx a.onnx --metadata a.json \
        --image path/to.jpg --conf 0.35
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loweye.detect.onnx_detector import OnnxDetector  # noqa: E402

FALLBACK_SIZE = 640


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract(session) -> dict:
    def describe(items):
        out = []
        for item in items:
            shape = [d if isinstance(d, int) else str(d) for d in item.shape]
            out.append({"name": item.name, "type": item.type, "shape": shape})
        return out

    return {"inputs": describe(session.get_inputs()), "outputs": describe(session.get_outputs())}


def int_shape_at(shape, index: int) -> int:
    """从 ONNX/metadata 的 [N,C,H,W] 形状里取出指定维；动态维（字符串）返回 0。"""
    if shape is None or len(shape) <= index:
        return 0
    value = shape[index]
    return int(value) if isinstance(value, int) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata", default="", help="训练侧 metadata JSON（提供 classes / preprocessing）")
    parser.add_argument("--input-size", type=int, default=0, help="显式覆盖输入边长（默认取 ONNX 声明值）")
    parser.add_argument("--image", default="", help="真实图片；不给就用合成噪声图")
    parser.add_argument("--conf", type=float, default=0.05, help="探针阈值（生产默认 0.35）")
    parser.add_argument("--json", dest="json_out", default="", help="把结果同时写到这个文件")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = (ROOT / onnx_path).resolve()
    if not onnx_path.is_file():
        print(json.dumps({"ok": False, "gaps": [f"ONNX 不存在: {onnx_path}"]}, ensure_ascii=False))
        return 2

    meta: dict = {}
    if args.metadata:
        meta_path = Path(args.metadata)
        if not meta_path.is_absolute():
            meta_path = (ROOT / meta_path).resolve()
        if not meta_path.is_file():
            print(json.dumps({"ok": False, "gaps": [f"metadata 不存在: {meta_path}"]}, ensure_ascii=False))
            return 2
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    classes = list(meta.get("classes") or [])
    if not classes:
        print(json.dumps({"ok": False, "gaps": ["metadata 里没有 classes，无法判断类别数"]}, ensure_ascii=False))
        return 2

    raw_ids = meta.get("label_ids") or []
    ids = sorted(int(v) for v in (raw_ids.values() if isinstance(raw_ids, dict) else raw_ids))
    dense_one_based = ids == list(range(1, len(classes) + 1))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    declared = contract(session)
    gaps: list[str] = []

    declared_size = int_shape_at(session.get_inputs()[0].shape, 2)
    meta_size = int_shape_at((meta.get("preprocessing") or {}).get("input"), 2)
    if declared_size and meta_size and declared_size != meta_size:
        gaps.append(f"ONNX 声明输入边长 {declared_size} 与 metadata.preprocessing.input {meta_size} 不一致")
    size = args.input_size or declared_size or meta_size or FALLBACK_SIZE
    if not declared_size and not meta_size:
        gaps.append(f"ONNX 与 metadata 都没给出输入边长，按 {FALLBACK_SIZE} 兜底")

    tmp_dir = Path(tempfile.mkdtemp(prefix="onnx_probe_"))
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = (ROOT / image_path).resolve()
        if not image_path.is_file():
            print(json.dumps({"ok": False, "gaps": [f"图片不存在: {image_path}"]}, ensure_ascii=False))
            return 2
        source = "real"
    else:
        rng = np.random.default_rng(0)
        image_path = tmp_dir / "synthetic.png"
        Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(image_path)
        source = "synthetic"

    detector = OnnxDetector(
        onnx_path=onnx_path,
        conf_threshold=args.conf,
        input_size=size,
        class_names=classes,
        slice_enabled=False,
    )
    result = detector.detect(image_path, "inspect")

    known = set(classes)
    unknown = sorted({d.cls for d in result.detections if d.cls not in known})
    if unknown:
        gaps.append(f"有检测框的类别名不在 metadata.classes 里: {unknown[:5]}")
    if not dense_one_based:
        gaps.append(f"label_ids 不是 1..{len(classes)} 的稠密集合：{len(ids)} 个，最小 {ids[0] if ids else None}、最大 {ids[-1] if ids else None}")
    if not result.detections:
        gaps.append(f"阈值 {args.conf} 下一个框都没有：可能是坏模型，也可能是训练不足的正常表现，需人工判断")

    payload = {
        "onnx": str(onnx_path),
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256_file(onnx_path),
        "metadata": str(args.metadata) if args.metadata else None,
        "smoke": bool(meta.get("smoke")) if meta else None,
        "test_status": meta.get("test_status") if meta else None,
        "classes": len(classes),
        "label_ids": {"count": len(ids), "min": ids[0] if ids else None, "max": ids[-1] if ids else None,
                      "dense_one_based": dense_one_based},
        "input_size": size,
        "input_size_evidence": {"onnx_declared": declared_size, "metadata_preprocessing": meta_size,
                                "override": args.input_size or None},
        "contract": declared,
        "layout": detector._layout,
        "image": {"path": str(image_path), "source": source},
        "conf_threshold": args.conf,
        "detections": len(result.detections),
        "detector": result.detector,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "top": [
            {"cls": d.cls, "score": round(float(d.score), 5),
             "bbox": [round(float(d.bbox.x1), 2), round(float(d.bbox.y1), 2),
                      round(float(d.bbox.x2), 2), round(float(d.bbox.y2), 2)]}
            for d in result.detections[:3]
        ],
        "warnings": list(result.warnings),
    }
    payload["gaps"] = gaps
    payload["ok"] = not gaps
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = (ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
