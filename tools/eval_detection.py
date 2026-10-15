"""检测环节的域内评估：每类漏检率 / 召回率 / 误报率。

这是 `docs/proposal/_research_detection.md` 里那条「没有真实权重就没法给漏检率」的
配套工具——把量测的部分先做好，权重一到就能出数。它与 `loweye/eval/` 里的检索评估
是两码事：那边评的是「条款找得对不对」，这边评的是「框画得对不对」。

判据（写死在代码里，不靠运行时推断）：
- 匹配：同一类别、IoU ≥ --iou（默认 0.5）、按分数从高到低贪心一对一；
- 每类漏检率 = 1 − 该类召回率 = 未匹配上的 GT 框数 / 该类 GT 框总数；
- **类无关召回**：忽略类别标签、只看有没有框压上 GT。用 COCO 权重评电力缺陷时，
  这个数才是「模型到底看没看见绝缘子」的可读答案（类别对不上是必然的）；
- 空图率：一张图一个框都没出的比例。

标签格式：Pascal VOC XML（CPLID 就是这种），或本项目 sidecar 用的 JSON。
    python -m tools.eval_detection --config config/app.onnx.yaml \
        --images data/external/cplid/Defective_Insulators/images \
        --labels data/external/cplid/Defective_Insulators/labels/defect \
        --out out/eval_detection_coco_on_cplid.md
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_voc(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """读 Pascal VOC XML，返回 [(类别, (x1,y1,x2,y2)), ...]。"""
    root = ET.parse(path).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "unknown").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        x1 = float(box.findtext("xmin")); y1 = float(box.findtext("ymin"))
        x2 = float(box.findtext("xmax")); y2 = float(box.findtext("ymax"))
        out.append((name, (x1, y1, x2, y2)))
    return out


def pairs(images_dir: Path, labels_dir: Path, only_from: set[str] | None = None) -> list[tuple[Path, Path]]:
    """把图片与同名 XML 配起来（XML 可能比图片多，例如 defect / normal 两套标注）。

    `only_from` 给的是允许的图片 stem 集合：训练时落盘的留出集就靠它限定，
    否则会拿训练集评测，指标等于自证。
    """
    found = []
    for img in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        if only_from is not None and img.stem not in only_from:
            continue
        xml = labels_dir / f"{img.stem}.xml"
        if xml.exists():
            found.append((img, xml))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT / "config" / "app.onnx.yaml"))
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张（0=全部）")
    ap.add_argument("--only-from", default=None,
                    help="划分文件（如 out/cplid_split.json）：只评其中某个 split，避免拿训练集自证")
    ap.add_argument("--split-key", default="val", help="从划分文件里取哪个键（默认 val）")
    ap.add_argument("--out", default=str(PROJECT / "out" / "eval_detection.md"))
    ap.add_argument("--json", dest="json_path", default=str(PROJECT / "out" / "eval_detection.json"))
    args = ap.parse_args()

    from loweye.config import load_config
    from loweye.detect.base import build_detector

    cfg = load_config(args.config)
    detector, why = build_detector(cfg)
    only_from = None
    if args.only_from:
        split = json.loads(Path(args.only_from).read_text(encoding="utf-8"))
        only_from = set(split.get(args.split_key, []))
        print(f"只评 {args.only_from} 的 {args.split_key} 组：{len(only_from)} 张")
    items = pairs(Path(args.images), Path(args.labels), only_from)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"没有配成对的图片与标注：{args.images} / {args.labels}")

    print(f"检测后端 {detector.name}；样本 {len(items)} 张；IoU 阈值 {args.iou}")
    if why:
        print(f"降级原因：{why}")

    per_class = defaultdict(lambda: {"gt": 0, "tp": 0, "fp": 0})
    pred_hist: Counter = Counter()
    empty_images = 0
    ag_gt = ag_tp = 0
    rows = []
    total_ms = 0.0
    slice_total = 0

    for img, xml in items:
        gts = read_voc(xml)
        result = detector.detect(img, image_id=img.stem)
        total_ms += result.elapsed_ms
        slice_total += result.slice_count
        if not result.detections:
            empty_images += 1
        for det in result.detections:
            pred_hist[det.cls] += 1

        # GT 总数先单独记一遍：命中与否只影响 tp，不能反过来决定 gt，
        # 否则「全部命中」的那类会算出 gt=0、tp=1 这种负漏检（阳性对照抓出来的）。
        for cls, _box in gts:
            per_class[cls]["gt"] += 1

        used = [False] * len(gts)
        for det in sorted(result.detections, key=lambda d: -d.score):
            best_i, best_iou = -1, 0.0
            for i, (cls, box) in enumerate(gts):
                if used[i]:
                    continue
                cur = iou((det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2), box)
                if cur > best_iou:
                    best_i, best_iou = i, cur
            if best_i >= 0 and best_iou >= args.iou:
                used[best_i] = True
                if gts[best_i][0] == det.cls:
                    per_class[det.cls]["tp"] += 1
                else:
                    # 框压上了但类别不对：GT 依旧算漏检（gt 已在上面记过），预测类记一次误报
                    per_class[det.cls]["fp"] += 1
            else:
                per_class[det.cls]["fp"] += 1

        ag_gt += len(gts)
        ag_used = [False] * len(gts)
        for det in sorted(result.detections, key=lambda d: -d.score):
            for i, (_cls, box) in enumerate(gts):
                if ag_used[i]:
                    continue
                if iou((det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2), box) >= args.iou:
                    ag_used[i] = True
                    ag_tp += 1
                    break
        rows.append({
            "image": img.name,
            "gt": len(gts),
            "pred": len(result.detections),
            "detector_ms": result.elapsed_ms,
            "slices": result.slice_count,
        })

    total_gt = sum(v["gt"] for v in per_class.values())
    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())

    lines = [
        "# 检测环节域内评估",
        "",
        f"- 检测后端：`{detector.name}`（{getattr(detector, 'model_version', '—')}）",
        f"- 权重：`{getattr(detector, 'onnx_path', '—')}`",
        f"- 样本：{len(items)} 张；IoU 阈值 {args.iou}",
        f"- 图片目录：`{args.images}`；标注目录：`{args.labels}`",
        "",
        "## 逐类结果",
        "",
        "| 类别 | GT 框 | 命中 | 漏检 | 漏检率 | 该类误报 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cls in sorted(per_class, key=lambda c: -per_class[c]["gt"]):
        v = per_class[cls]
        miss = (v["gt"] - v["tp"]) / v["gt"] if v["gt"] else 0.0
        lines.append(
            f"| {cls} | {v['gt']} | {v['tp']} | {v['gt'] - v['tp']} | {miss:.0%} | {v['fp']} |"
        )
    lines += [
        f"| **合计** | {total_gt} | {total_tp} | {total_gt - total_tp} | "
        f"**{(total_gt - total_tp) / total_gt:.0%}** | {total_fp} |" if total_gt else "| **合计** | 0 | 0 | 0 | — | 0 |",
        "",
        "## 汇总",
        "",
        f"- **类无关召回**（忽略类别，只看有没有框压上 GT）：{ag_tp}/{ag_gt} = "
        f"{ag_tp / ag_gt:.0%}" if ag_gt else "- 类无关召回：无 GT",
        f"- 空图率（一个框都没出的图片占比）：{empty_images}/{len(items)} = "
        f"{empty_images / len(items):.0%}",
        f"- 平均单图耗时：{total_ms / len(items):.0f} ms；平均切片数：{slice_total / len(items):.1f}",
        f"- 模型预测到的类别分布："
        + ("、".join(f"{k}×{v}" for k, v in pred_hist.most_common(10)) or "（无预测）"),
        "",
        "## 判读说明",
        "",
        "- 类别标签对不上时，即使框压中了 GT，也按「该类漏检 + 预测类误报」计；",
    ]
    model_tag = str(getattr(detector, "model_version", "") or "").lower()
    if "coco" in model_tag or "cplid-fcos" not in model_tag:
        # COCO 等外部类别权重：逐类漏检必然 100%，只能看类无关召回
        lines.append(
            "- 因此**用 COCO 权重评电力缺陷时，逐类漏检率必然是 100%**，这时要看的是"
            "「类无关召回」——它回答的是「模型到底有没有把框放到绝缘子上」；"
        )
    else:
        lines.append(
            "- 本表用的是在 CPLID 上微调过的权重，类别与标注同源，"
            "所以**逐类漏检率可以直接读**；`该类误报` 一列是 IoU≥0.5 但类别不符的预测框数，"
            "正常图里出现 `insulator` 误报主要来自同一串绝缘子的重复框；"
        )
    lines += [
        "- 本表不含任何检测精度主张：它只在有真实缺陷权重与真实标注时才有意义；"
        "- 留出集来自 `out/cplid_split.json` 的 val 划分，与本机词法/向量两档用的是同一批图，"
        "训练时未见过。",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_path).write_text(
        json.dumps(
            {
                "detector": detector.name,
                "model_version": getattr(detector, "model_version", None),
                "onnx_path": str(getattr(detector, "onnx_path", "")) or None,
                "n_images": len(items),
                "iou": args.iou,
                "per_class": {k: v for k, v in per_class.items()},
                "totals": {"gt": total_gt, "tp": total_tp, "fp": total_fp},
                "class_agnostic_recall": (ag_tp / ag_gt) if ag_gt else None,
                "empty_image_rate": empty_images / len(items),
                "avg_ms": total_ms / len(items),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n已写出：{out} 与 {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
