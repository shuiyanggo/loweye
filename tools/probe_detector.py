"""检测器冒烟探针：给定配置与图片，直接把检测结果打出来。

为什么单独写一个：`tools/api_smoke.py` 走的是接口、`loweye.cli.analyze` 走的是整条链路
（还会写工单和报告），要单独验证「换了权重文件之后检测这一环还对不对」时都不方便。
这个探针只做一件事——建检测器、跑图、打印框与耗时，不写任何业务数据。

用法：
    python -m tools.probe_detector --config config/app.onnx.yaml --image path/to.jpg
    python -m tools.probe_detector --config config/app.onnx.yaml --image a.jpg --image b.jpg
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "config" / "app.yaml"))
    ap.add_argument("--image", action="append", required=True, help="可以重复传多张")
    ap.add_argument("--top", type=int, default=10, help="每张图最多打印几个框")
    ap.add_argument("--no-slice", action="store_true", help="关掉 SAHI 切片再看一次")
    args = ap.parse_args()

    from loweye.config import load_config
    from loweye.detect.base import build_detector

    cfg = load_config(args.config)
    detector, why = build_detector(cfg)
    print(f"配置：{args.config}")
    print(f"检测后端：{detector.name}  模型版本：{getattr(detector, 'model_version', '—')}")
    if why:
        print(f"降级原因：{why}")
    if getattr(detector, "onnx_path", None):
        size_mb = Path(detector.onnx_path).stat().st_size / 1024 / 1024
        print(f"权重：{detector.onnx_path}（{size_mb:.1f} MB，布局 {detector._layout}）")
    if args.no_slice and hasattr(detector, "slice_enabled"):
        detector.slice_enabled = False
        print("已关闭 SAHI 切片")

    for raw in args.image:
        path = Path(raw)
        if not path.is_absolute():
            path = (PROJECT / path) if not path.exists() else path
        if not path.exists():
            print(f"\n[跳过] 找不到图片：{path}")
            continue
        result = detector.detect(path, image_id=path.stem)
        print(f"\n=== {path.name} ===")
        print(f"检出 {len(result.detections)} 个目标，耗时 {result.elapsed_ms} ms，"
              f"切片数 {result.slice_count}")
        if result.warnings:
            print(f"警告：{result.warnings}")
        counter = Counter(d.cls for d in result.detections)
        if counter:
            print("类别分布：" + "、".join(f"{k}×{v}" for k, v in counter.most_common()))
        for det in sorted(result.detections, key=lambda d: -d.score)[: args.top]:
            b = det.bbox
            print(f"  {det.cls:<16} score={det.score:.3f} "
                  f"bbox=({b.x1:.0f},{b.y1:.0f})-({b.x2:.0f},{b.y2:.0f}) "
                  f"{b.x2 - b.x1:.0f}×{b.y2 - b.y1:.0f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
