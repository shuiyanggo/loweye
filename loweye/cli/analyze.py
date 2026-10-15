"""跑一次完整分析链路（识别 → 诊断 → 分级 → 派单 → 归档）。

用法：
    .venv\\Scripts\\python.exe -m loweye.cli.analyze --image demo_tower_01
    .venv\\Scripts\\python.exe -m loweye.cli.analyze --image demo_tower_01 --json
    .venv\\Scripts\\python.exe -m loweye.cli.analyze --list
"""

from __future__ import annotations

import argparse
import json

from ..agent import Pipeline
from ..config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="低空之眼 · 单图分析")
    parser.add_argument("--image", default=None, help="data/assets 下的图片名（不含扩展名）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="列出可用演示图片")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    assets = cfg.path("assets")

    if args.list:
        for p in sorted(assets.glob("*.png")):
            sidecar = p.with_suffix(".detections.json")
            print(f"{p.stem:<24} {'有标注' if sidecar.exists() else '无标注'}")
        return 0

    if not args.image:
        parser.error("需要 --image 指定图片，或使用 --list 查看可用图片")

    image_path = assets / f"{args.image}.png"
    if not image_path.exists():
        parser.error(f"未找到 {image_path}")

    pipe = Pipeline.from_config(cfg)
    result = pipe.run(image_path, image_id=args.image)

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    print("=" * 68)
    print(f"分析完成  trace_id={result.trace_id}  总耗时 {result.total_ms:.1f}ms")
    print("=" * 68)
    print(f"检测后端：{result.detect.detector} / {result.detect.model_version}"
          f"（切片 {result.detect.slice_count} 块，耗时 {result.detect.elapsed_ms:.1f}ms）")
    for d in result.diagnoses:
        print(f"\n[检测 {d.detection_index}] {d.label}（{d.cls}）置信度 {d.confidence:.2f} → {d.severity}")
        print(f"  结论：{d.finding}")
        for a in d.actions:
            print(f"  动作：{a}")
        basis = [c for c in d.citations if c.role == "basis"]
        refs = [c for c in d.citations if c.role != "basis"]
        for c in basis:
            print(f"  依据：{c.clause_no}  《{c.doc_title}》  分数 {c.score:.3f}")
        for c in refs:
            print(f"  参考：{c.clause_no}  《{c.doc_title}》  分数 {c.score:.3f}（检索命中但未作为处置动作）")
    for o in result.work_orders:
        print(f"\n工单 {o.order_id}｜{o.severity}｜{o.assignee_role}｜SLA {o.sla_hours}h｜路由 {o.route}｜状态 {o.status}")
        if o.review_reasons:
            for r in o.review_reasons:
                print(f"    - 转人工原因：{r}")
    if result.degradations:
        print("\n降级项：")
        for d in result.degradations:
            print(f"  [{d.stage}] {d.reason} → {d.fallback}")
    print("\n分阶段耗时：")
    for t in result.timings:
        print(f"  {t.stage:<14} {t.ms:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
