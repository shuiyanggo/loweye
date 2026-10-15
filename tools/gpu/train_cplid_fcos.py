#!/usr/bin/env python3
"""在 GPU 机器上微调 torchvision FCOS-ResNet50-FPN（CPLID：insulator / defect 两类），并导出 ONNX。

为什么不是 YOLOX-tiny / RTMDet-tiny（用户最初的目标）：
  远端 `pip index` 能装 mmdet 3.3.0 / mmcv 2.2.0，但 openmmlab 的预编译 mmcv 轮子只到
  cu121/torch2.2（cu128、cu126 都是 404），机器上是 torch 2.8.0+cu128，且没有 nvcc 没法源码编译。
  要装 mmdet 就得把 torch 降级重装，风险大于收益。FCOS 与 YOLOX/RTMDet 同属 anchor-free 单阶段
  检测器，是 torchvision 里最贴近的等价物，且能直接 torch.onnx.export 导出（不依赖 mmdeploy）。

用法（远端）：
    python3 train_cplid_fcos.py --data /root/loweye/data/external/cplid \
        --split /root/loweye/data/cplid_split.json \
        --out /root/autodl-tmp/loweye/models --epochs 24 --batch 4 --size 1024
"""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
from torchvision.ops import box_iou

CLASSES = {"insulator": 1, "defect": 2}
NAMES = {0: "background", 1: "insulator", 2: "defect"}
NUM_CLASSES = 3


# --------------------------------------------------------------------- 数据

def parse_voc(xml_path: Path) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CLASSES[name])
    return boxes, labels


def collect(data_root: Path) -> list[tuple[Path, Path]]:
    """返回 [(图, 标注)]；正常图标注在 labels/，缺陷图标注在 labels/defect/。"""
    items: list[tuple[Path, Path]] = []
    for subset, lab_rel in (("Normal_Insulators", "labels"), ("Defective_Insulators", "labels/defect")):
        img_dir = data_root / subset / "images"
        lab_dir = data_root / subset / lab_rel
        for img in sorted(img_dir.glob("*.jpg")):
            xml = lab_dir / f"{img.stem}.xml"
            if xml.exists():
                items.append((img, xml))
    return items


class CplidDataset(Dataset):
    def __init__(self, items, size: int, train: bool) -> None:
        self.items = items
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.items[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = self.size / max(w, h)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = im.resize((nw, nh), Image.BILINEAR)
            canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
            pad_x, pad_y = (self.size - nw) // 2, (self.size - nh) // 2
            canvas.paste(resized, (pad_x, pad_y))
            arr = np.asarray(canvas, dtype="float32") / 255.0

        boxes, labels = parse_voc(xml_path)
        boxes = [
            [b[0] * scale + pad_x, b[1] * scale + pad_y, b[2] * scale + pad_x, b[3] * scale + pad_y]
            for b in boxes
        ]
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.train and torch.rand(1).item() < 0.5:
            arr = arr[:, ::-1, :].copy()
            if boxes_t.numel():
                x1 = boxes_t[:, 0].clone()
                x2 = boxes_t[:, 2].clone()
                boxes_t[:, 0] = self.size - x2
                boxes_t[:, 2] = self.size - x1

        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}
        return image, target


def collate(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------- 指标

@torch.no_grad()
def evaluate(model, loader, device, conf: float = 0.05) -> dict:
    model.eval()
    gt_total = {c: 0 for c in (1, 2)}
    hit = {c: 0 for c in (1, 2)}
    pred_total = {c: 0 for c in (1, 2)}
    gt_all, hit_all, empty = 0, 0, 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            gb, gl = tgt["boxes"].to(device), tgt["labels"].to(device)
            pb, pl, ps = out["boxes"], out["labels"], out["scores"]
            keep = ps >= conf
            pb, pl, ps = pb[keep], pl[keep], ps[keep]
            if len(pb) == 0:
                empty += 1
            gt_all += len(gb)
            for c in (1, 2):
                gt_total[c] += int((gl == c).sum())
                pred_total[c] += int((pl == c).sum())
            if len(gb) and len(pb):
                ious = box_iou(gb, pb)
                used = set()
                order = torch.argsort(ps, descending=True)
                for pi in order:
                    best_iou, best_gi = 0.0, -1
                    for gi in range(len(gb)):
                        if gi in used:
                            continue
                        v = float(ious[gi, pi])
                        if v > best_iou:
                            best_iou, best_gi = v, gi
                    if best_iou >= 0.5:
                        used.add(best_gi)
                        hit_all += 1
                        c = int(gl[best_gi])
                        if c in hit:
                            hit[c] += 1
    metrics = {
        "gt_total": gt_total,
        "hit": hit,
        "pred_total": pred_total,
        "miss_rate": {NAMES[c]: (0.0 if gt_total[c] == 0 else 1 - hit[c] / gt_total[c]) for c in (1, 2)},
        "recall_all": 0.0 if gt_all == 0 else hit_all / gt_all,
        "gt_all": gt_all,
        "hit_all": hit_all,
        "empty_images": empty,
        "images": len(loader.dataset),
    }
    model.train()
    return metrics


# --------------------------------------------------------------------- 主流程

def build_model(size: int, device):
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
    head = model.head.classification_head
    old = head.cls_logits
    num_anchors = getattr(head, "num_anchors", old.out_channels // head.num_classes)
    new = nn.Conv2d(old.in_channels, num_anchors * NUM_CLASSES, kernel_size=3, stride=1, padding=1)
    nn.init.normal_(new.weight, std=0.01)
    nn.init.constant_(new.bias, -math.log((1 - 0.01) / 0.01))
    head.num_classes = NUM_CLASSES
    head.cls_logits = new
    # 让内置 transform 不再改变尺寸：数据集已经 letterbox 成 size×size
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default=None, help="cplid_split.json，其中的 val 作为留出集")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=0, help=">0 时只取这么多训练图（冒烟用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261013)
    np.random.seed(20261013)

    data_root = Path(args.data)
    items = collect(data_root)
    val_stems: set[str] = set()
    if args.split and Path(args.split).exists():
        val_stems = set(json.loads(Path(args.split).read_text(encoding="utf-8"))["val"])
    train_items = [it for it in items if it[0].stem not in val_stems]
    val_items = [it for it in items if it[0].stem in val_stems]
    if args.limit_train:
        train_items = train_items[: args.limit_train]
    print(f"数据：总 {len(items)}，训练 {len(train_items)}，留出 {len(val_items)}", flush=True)

    train_ds = CplidDataset(train_items, args.size, train=True)
    val_ds = CplidDataset(val_items, args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              collate_fn=collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch // 2), shuffle=False,
                            num_workers=max(2, args.workers // 2), collate_fn=collate, pin_memory=True)

    model = build_model(args.size, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    best = {"score": -1.0, "epoch": 0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, n = 0.0, 0
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            running += float(loss.detach())
            n += 1
        train_loss = running / max(1, n)

        with torch.no_grad():
            vloss, vn = 0.0, 0
            model.train()
            for images, targets in val_loader:
                images = [im.to(device) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                ld = model(images, targets)
                vloss += float(sum(v for v in ld.values()))
                vn += 1
            val_loss = vloss / max(1, vn)

        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "miss_insulator": round(metrics["miss_rate"]["insulator"], 4),
            "miss_defect": round(metrics["miss_rate"]["defect"], 4),
            "recall_all": round(metrics["recall_all"], 4),
            "seconds": round(dt, 1),
        }
        log.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = (1 - metrics["miss_rate"]["insulator"]) + (1 - metrics["miss_rate"]["defect"])
        if score > best["score"]:
            best = {"score": score, "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "cplid_fcos_best.pth")
        torch.save(model.state_dict(), out_dir / "cplid_fcos_last.pth")
        (out_dir / "train_fcos_log.json").write_text(
            json.dumps({"args": vars(args), "log": log, "best": best}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------------------------------------------------------------- 导出 ONNX
    model.load_state_dict(torch.load(out_dir / "cplid_fcos_best.pth", map_location=device))
    model.eval()
    onnx_path = out_dir / f"cplid_fcos_{args.size}.onnx"
    dummy = torch.rand(1, 3, args.size, args.size, device=device)
    try:
        torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=16,
                          input_names=["images"], output_names=["boxes", "labels", "scores"],
                          dynamo=False, do_constant_folding=True)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=16,
                          input_names=["images"], output_names=["boxes", "labels", "scores"])
    print("ONNX 导出完成:", onnx_path, f"{onnx_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        probe = np.random.rand(1, 3, args.size, args.size).astype("float32")
        outs = sess.run(None, {sess.get_inputs()[0].name: probe})
        print("ONNX 输出:", [(o.shape, str(o.dtype)) for o in outs], flush=True)
    except Exception as exc:  # noqa: BLE001
        print("onnxruntime 校验跳过:", type(exc).__name__, exc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
