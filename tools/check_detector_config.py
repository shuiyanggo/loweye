"""把配置里写的那套检测参数真正解析成检测器，并报告它实际用了什么。

为什么需要单独一支：loweye/detect/base.py:54-60 在 ONNX 后端构造失败时**不抛错**，
而是静默降级为 sidecar（读图片同目录的 .json 离线标注），只把原因放进返回值。
对着一张没有同名 .json 的真实照片，降级的后果是「零检出」——看起来像模型什么都没
找到，而不是配置错了。换权重（绝缘子 → 烟火 / 农作物害虫）时必须能一眼看出
「到底加载的是哪张图、哪个输入边长、哪张类别表」。

用法：
    # 现役配置（绝缘子，1024 px，2 类）
    python -m tools.check_detector_config --config config/app.cplid_gpu.yaml

    # 排练新权重：用元数据里的类别表覆盖配置，并指定 ONNX 与输入边长
    python -m tools.check_detector_config --config config/app.yaml \
        --onnx out/inspection_gpu/_localcheck/crop_pest_fcos_800.onnx \
        --input-size 800 \
        --classes-from out/inspection_gpu/_localcheck/crop_pest_fcos_metadata.json

退出码：0=按配置加载成功；3=发生静默降级（配置没有生效）；2=参数或配置有问题。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthesize(path: Path) -> Path:
    """没有给真实图片时用一张合成图，只为验证「能加载、能跑通」。"""
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--onnx", help="覆盖 detect.onnx_path，并强制 detect.detector=onnx")
    parser.add_argument("--input-size", type=int, help="覆盖 detect.input_size")
    parser.add_argument("--classes-from", type=Path,
                        help="从这份元数据 JSON 的 classes 字段覆盖 detect.class_names")
    parser.add_argument("--model-version",
                        help="覆盖 detect.model_version（报告与界面里显示的模型名，换权重时别留旧名）")
    parser.add_argument("--image", type=Path, help="真实图片；缺省用合成图")
    parser.add_argument("--json-out", type=Path, help="把报告同时写到这个文件")
    args = parser.parse_args()

    from loweye.config import load_config
    from loweye.detect.base import build_detector

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # 配置不存在 / YAML 不合法
        print(f"配置加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    detect = cfg.as_dict().setdefault("detect", {})
    if args.onnx:
        detect["detector"] = "onnx"
        detect["onnx_path"] = args.onnx
    if args.input_size:
        detect["input_size"] = args.input_size
    if args.classes_from:
        payload = json.loads(args.classes_from.read_text(encoding="utf-8"))
        classes = payload.get("classes") or payload.get("class_names")
        if not classes:
            print(f"{args.classes_from} 里没有 classes / class_names 字段", file=sys.stderr)
            return 2
        detect["class_names"] = list(classes)
    if args.model_version:
        detect["model_version"] = args.model_version

    detector, degrade = build_detector(cfg)
    image = args.image if args.image else synthesize(PROJECT_ROOT / "out" / "_detector_config_input.png")

    started = time.perf_counter()
    error = None
    try:
        result = detector.detect(image, "config-check")
        detections = len(result.detections)
        warnings = list(result.warnings or [])
    except Exception as exc:  # 输入边长不符这类问题只在推理时才暴露，构造期看不出来
        detections, warnings, error = None, [], f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    report = {
        "config": str(args.config),
        "requested_backend": str(detect.get("detector")),
        "resolved_backend": str(getattr(detector, "name", "?")),
        "degrade_reason": degrade,
        "error": error,
        "onnx_path": str(detect.get("onnx_path")),
        "input_size": getattr(detector, "input_size", None),
        "classes": len(getattr(detector, "class_names", []) or []),
        "layout": getattr(detector, "_layout", None),  # 私有属性仅用于诊断打点
        "model_version": getattr(detector, "model_version", None),
        "image": str(image),
        "detections": detections,
        "elapsed_ms": round(elapsed_ms, 1),
        "warnings": warnings,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if degrade:
        print("配置没有生效：ONNX 后端构造失败并已静默降级", file=sys.stderr)
        return 3
    if error:
        print(f"配置没有生效：推理阶段报错 {error}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
