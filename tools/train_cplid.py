"""在 CPLID 上微调一个检测器，让「每类漏检率」有真实权重可测。

为什么用 torchvision 的 Faster R-CNN + MobileNetV3 而不是 YOLOX：
- 官方 YOLOX 包在本机装不上（PyPI 包需要 cmake 编译 onnx 扩展，github.com 也 clone 不动，
  见 docs/proposal/_research_detection.md 附录 B）；torchvision 是 BSD-3、pip 一条命令装好、
  自带 COCO 预训练权重，正是「许可干净 + 能在这台只有 CPU 的机器上跑」的组合。
- 代价：Faster R-CNN 的后处理（ROI Align + NMS）导出 ONNX 麻烦，所以本次先用 torch 权重
  拿评测数字，导出另做（框架里留了 `detect.detector: torch` 这条路）。

类别约定（CPLID 官方协议）：正常图里的绝缘子 → `insulator`；缺陷图里的缺陷区域 → `defect`。

用法：
    python -m tools.train_cplid --epochs 6 --limit 0        # limit=0 表示用全部
    python -m tools.train_cplid --epochs 1 --limit 60       # 先跑一轮试时间
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CPLID = PROJECT / "data" / "external" / "cplid"
OUT_WEIGHTS = PROJECT / "models" / "detector" / "cplid_insulator_fasterrcnn.pth"
OUT_LOG = PROJECT / "out" / "train_cplid_log.json"

CLASSES = ["insulator", "defect"]


def read_voc(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None or name not in CLASSES:
            continue
        out.append((
            name,
            (float(box.findtext("xmin")), float(box.findtext("ymin")),
             float(box.findtext("xmax")), float(box.findtext("ymax"))),
        ))
    return out


def collect(limit_per_split: int) -> list[tuple[Path, Path]]:
    """返回 [(图片, 标注 XML)]。正常图 → insulator，缺陷图 → defect。"""
    items: list[tuple[Path, Path]] = []
    for images_dir, labels_dir in (
        (CPLID / "Normal_Insulators" / "images", CPLID / "Normal_Insulators" / "labels"),
        (CPLID / "Defective_Insulators" / "images", CPLID / "Defective_Insulators" / "labels" / "defect"),
    ):
        if not images_dir.exists():
            continue
        pairs = []
        for img in sorted(images_dir.glob("*.jpg")):
            xml = labels_dir / f"{img.stem}.xml"
            if xml.exists():
                pairs.append((img, xml))
        if limit_per_split:
            pairs = pairs[:limit_per_split]
        items.extend(pairs)
    return items


class CplidDataset:
    """极简检测数据集：返回 (image_tensor, target{boxes, labels})，不做增广。"""

    def __init__(self, items: list[tuple[Path, Path]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        from PIL import Image

        img_path, xml_path = self.items[idx]
        with Image.open(img_path) as im:
            arr = np.asarray(im.convert("RGB"))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)

        boxes, labels = [], []
        for name, (x1, y1, x2, y2) in read_voc(xml_path):
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASSES.index(name) + 1)  # 0 留给背景
        if not boxes:
            boxes, labels = [[0.0, 0.0, 1.0, 1.0]], [1]
        return tensor, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def evaluate_loss(model, loader) -> float:
    """跑一遍验证集取平均损失（只看 loss，不下结论——指标用 tools/eval_detection.py）。"""
    torch = __import__("torch")
    model.train()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            losses = model(images, targets)
            total += float(sum(losses.values()))
            n += 1
    return total / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="每个子集最多用多少张（0=全部）")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--threads", type=int, default=8, help="限制线程数，别把演示服务拖垮")
    ap.add_argument("--min-size", type=int, default=416)
    ap.add_argument("--max-size", type=int, default=640)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    items = collect(args.limit)
    if len(items) < 10:
        raise SystemExit(f"可用样本太少（{len(items)}）；先跑 tools/fetch_cplid.py 拉数据")
    random.shuffle(items)
    n_val = max(4, int(len(items) * args.val_ratio))
    val_items, train_items = items[:n_val], items[n_val:]
    # 把划分落盘：评测必须只在留出的那些图上做，否则「每类漏检率」是在训练集上自证
    split_path = PROJECT / "out" / "cplid_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "train": [p[0].stem for p in train_items],
                "val": [p[0].stem for p in val_items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"样本合计 {len(items)}：训练 {len(train_items)} / 验证 {len(val_items)}（CPU，线程 {args.threads}）")
    print(f"划分已落盘：{split_path}（评测用 --only-from 限定在 val 上）")

    train_loader = DataLoader(CplidDataset(train_items), batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(CplidDataset(val_items), batch_size=args.batch, shuffle=False,
                            collate_fn=collate, num_workers=0)

    # COCO 预训练的检测头是 91 类，直接传 num_classes=3 会被 torchvision 拒绝
    # （_ovewrite_value_param 校验），标准做法是先按 COCO 建好、再换掉 box predictor。
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT",  # COCO 预训练主干与检测头，许可与权重来源都干净
        weights_backbone="DEFAULT",
        min_size=args.min_size,
        max_size=args.max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        running, steps = 0.0, 0
        for images, targets in train_loader:
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)
            steps += 1
            if steps % 10 == 0:
                print(f"  epoch {epoch} step {steps}/{len(train_loader)} loss={running / steps:.4f}", flush=True)
        val_loss = evaluate_loss(model, val_loader)
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(1, steps), 4),
            "val_loss": round(val_loss, 4),
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.append(row)
        print(f"epoch {epoch}/{args.epochs} 完成：{row}", flush=True)

    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "classes": CLASSES,
        "arch": "fasterrcnn_mobilenet_v3_large_fpn",
        "model_version": f"cplid-frcnn-mnv3-{args.epochs}ep",
        "trained": {
            "dataset": "CPLID (Tao et al.; local evaluation subset)",
            "n_train": len(train_items),
            "n_val": len(val_items),
            "epochs": args.epochs,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "history": history,
            "total_seconds": round(time.perf_counter() - started, 1),
            "license_note": "模型代码 torchvision = BSD-3；数据 CPLID 许可未明示，仅本地评测",
        },
    }
    torch.save(payload, OUT_WEIGHTS)
    OUT_LOG.write_text(json.dumps(payload["trained"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"权重已保存：{OUT_WEIGHTS}（{OUT_WEIGHTS.stat().st_size / 1024 / 1024:.1f} MB）")
    print(f"训练日志：{OUT_LOG}")
    print("下一步：python -m tools.eval_detection --config config/app.cplid.yaml "
          "--images data/external/cplid/Defective_Insulators/images "
          "--labels data/external/cplid/Defective_Insulators/labels/defect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
