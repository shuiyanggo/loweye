"""在 CPLID 上训一个「绝缘子裁剪图 → 健康 / 缺陷」分类器。

为什么不用「直接检测 defect 类」：实测下来那条路在这台机器上不成立——CPLID 的缺陷框
中位只有 66×47 像素、94% 的框面积占比不足 0.5%，12 epoch、444 张图的微调里
defect 类的最佳 IoU 中位数只有 0.051（没有一个框到 IoU 0.3）。而同一份权重对
insulator 类的召回是 96%（insulator 框中位 640×109 像素）。

所以改成 CPLID 论文用的两段式：**先检出绝缘子，再对裁剪图判健康 / 缺陷**。
第二段是图像分类，目标大小不再是问题，样本量也从 198 个训练框变成上千张裁剪图。

用法：
    python -m tools.train_defect_clf --epochs 4
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
OUT_WEIGHTS = PROJECT / "models" / "detector" / "cplid_defect_clf.pth"
OUT_LOG = PROJECT / "out" / "train_defect_clf_log.json"


def crops_from(labels_dir: Path, images_dir: Path, cls_name: str, label: int, limit: int = 0):
    """按 GT 框裁图，返回 [(图片路径, 框, 标签)]。"""
    out = []
    for xml in sorted(labels_dir.glob("*.xml")):
        img = images_dir / f"{xml.stem}.jpg"
        if not img.exists():
            continue
        root = ET.parse(xml).getroot()
        for obj in root.findall("object"):
            if (obj.findtext("name") or "").strip() != cls_name:
                continue
            b = obj.find("bndbox")
            box = (float(b.findtext("xmin")), float(b.findtext("ymin")),
                   float(b.findtext("xmax")), float(b.findtext("ymax")))
            out.append((img, box, label, xml.stem))
    if limit:
        out = out[:limit]
    return out


class CropDataset:
    def __init__(self, items, size: int = 224, train: bool = True) -> None:
        self.items = items
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        from PIL import Image

        img_path, box, label, _stem = self.items[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            x1, y1, x2, y2 = box
            # 向外扩 15%：缺陷常常贴在框边界上，裁太紧会切掉上下文
            pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            crop = im.crop((
                max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y)),
            )).resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(crop).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
        if self.train and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        return tensor, label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    healthy = crops_from(CPLID / "Normal_Insulators" / "labels",
                         CPLID / "Normal_Insulators" / "images", "insulator", 0, args.limit)
    defective = crops_from(CPLID / "Defective_Insulators" / "labels" / "defect",
                           CPLID / "Defective_Insulators" / "images", "defect", 1, args.limit)
    print(f"裁剪样本：健康 {len(healthy)} / 缺陷 {len(defective)}")
    if not healthy or not defective:
        raise SystemExit("样本不足，先跑 tools/fetch_cplid.py")

    random.shuffle(healthy)
    random.shuffle(defective)

    def split(items):
        n_val = max(2, int(len(items) * args.val_ratio))
        return items[n_val:], items[:n_val]

    h_tr, h_va = split(healthy)
    d_tr, d_va = split(defective)
    train_items = h_tr + d_tr
    val_items = h_va + d_va
    random.shuffle(train_items)
    random.shuffle(val_items)
    print(f"训练 {len(train_items)}（健康 {len(h_tr)} / 缺陷 {len(d_tr)}）"
          f" / 验证 {len(val_items)}（健康 {len(h_va)} / 缺陷 {len(d_va)}）")

    # 类别不平衡：健康样本多得多，用权重把缺陷类抬上来
    w_defect = len(h_tr) / max(1, len(d_tr))
    weights = torch.tensor([1.0, w_defect], dtype=torch.float32)
    print(f"缺陷类损失权重：{w_defect:.2f}")

    train_loader = DataLoader(CropDataset(train_items, args.size, True),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_items, args.size, False),
                            batch_size=args.batch, shuffle=False, num_workers=0)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
        model.eval()
        cm = [[0, 0], [0, 0]]  # cm[真实][预测]
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).argmax(1)
                for t, p in zip(y.tolist(), pred.tolist()):
                    cm[t][p] += 1
        healthy_miss = cm[0][1] / max(1, cm[0][0] + cm[0][1])
        defect_miss = cm[1][0] / max(1, cm[1][0] + cm[1][1])
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(1, len(train_loader)), 4),
            "train_acc": round(correct / max(1, seen), 4),
            "val_confusion": cm,
            "healthy_miss_rate": round(healthy_miss, 4),
            "defect_miss_rate": round(defect_miss, 4),
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(row)
        print(f"epoch {epoch}/{args.epochs}：{row}", flush=True)

    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "classes": ["healthy", "defect"],
        "arch": "mobilenet_v3_small",
        "model_version": f"cplid-defect-clf-{args.epochs}ep",
        "trained": {
            "dataset": "CPLID crops (normal insulators = healthy, defective regions = defect)",
            "n_train": len(train_items),
            "n_val": len(val_items),
            "epochs": args.epochs,
            "size": args.size,
            "history": history,
            "total_seconds": round(time.perf_counter() - started, 1),
            "license_note": "torchvision MobileNetV3 = BSD-3；数据 CPLID 许可未明示，仅本地评测",
        },
    }
    torch.save(payload, OUT_WEIGHTS)
    OUT_LOG.write_text(json.dumps(payload["trained"], ensure_ascii=False, indent=2), encoding="utf-8")
    last = history[-1]
    print(f"\n权重：{OUT_WEIGHTS}（{OUT_WEIGHTS.stat().st_size / 1024 / 1024:.1f} MB）")
    print(f"最后一轮：健康漏检 {last['healthy_miss_rate']:.1%} / 缺陷漏检 {last['defect_miss_rate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
