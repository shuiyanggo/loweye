"""生成演示用合成图与对应的离线检测标注（sidecar）。

必须说清楚的事：
1. 这些图是**程序画出来的合成图**，不是真实航拍影像；
2. 标注框是画图时一并写出的，因此它验证的是「识别→诊断→派单→归档」整条链路的
   连通性与可复现性，**不能用来证明模型精度**；
3. 真实精度必须在真实数据集（CPLID / InsPLAD / 自采）上训练与评测，
   评测脚本会明确区分这两种性质不同的结论。

用法：
    .venv\\Scripts\\python.exe tools\\make_demo_assets.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "data" / "assets"

W, H = 1600, 1000
random.seed(20261012)


def sky(draw: ImageDraw.ImageDraw, top=(176, 208, 235), bottom=(232, 240, 246)) -> None:
    for y in range(H):
        t = y / H
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=c)


def hill(draw: ImageDraw.ImageDraw, base_y: int, color) -> None:
    pts = [(0, H)] + [(x, base_y + int(40 * math.sin(x / 260.0))) for x in range(0, W + 40, 40)] + [(W, H)]
    draw.polygon(pts, fill=color)


def tower(draw: ImageDraw.ImageDraw, cx: int, top_y: int, base_y: int, color=(72, 80, 88)) -> None:
    """画一座简化输电塔（主材 + 横隔 + 交叉斜材 + 顶部横担）。"""
    half_top, half_bottom, n = 34, 96, 9

    def x_at(t: float, side: int) -> float:
        return cx + side * (half_top + (half_bottom - half_top) * t)

    def y_at(t: float) -> int:
        return int(top_y + (base_y - top_y) * t)

    draw.line([(x_at(0, -1), top_y), (x_at(1, -1), base_y)], fill=color, width=5)
    draw.line([(x_at(0, 1), top_y), (x_at(1, 1), base_y)], fill=color, width=5)
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        y0, y1 = y_at(t0), y_at(t1)
        draw.line([(x_at(t0, -1), y0), (x_at(t0, 1), y0)], fill=color, width=3)
        draw.line([(x_at(t0, -1), y0), (x_at(t1, 1), y1)], fill=color, width=2)
        draw.line([(x_at(t0, 1), y0), (x_at(t1, -1), y1)], fill=color, width=2)
    draw.line([(x_at(1, -1), base_y), (x_at(1, 1), base_y)], fill=color, width=4)
    draw.line([(cx - half_top, top_y), (cx + half_top, top_y)], fill=color, width=4)


def insulator_string(draw: ImageDraw.ImageDraw, x: int, y: int, n: int, *, broken_at: int | None = None,
                     color=(210, 214, 218), edge=(120, 128, 136)) -> None:
    for i in range(n):
        cy = y + i * 13
        if broken_at is not None and i == broken_at:
            draw.line([(x - 12, cy - 6), (x + 12, cy + 6)], fill=(176, 32, 32), width=4)
            draw.line([(x - 12, cy + 6), (x + 12, cy - 6)], fill=(176, 32, 32), width=4)
            continue
        draw.ellipse([x - 11, cy - 6, x + 11, cy + 6], fill=color, outline=edge)


def make_tower_scene() -> tuple[Image.Image, list[dict]]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    sky(d)
    hill(d, 760, (196, 206, 200))
    hill(d, 830, (168, 182, 172))
    ground_y = 900

    # 远处塔
    tower(d, 300, 300, ground_y - 40, color=(150, 158, 166))
    # 主塔
    tower(d, 820, 210, ground_y, color=(74, 82, 90))
    ins_x = 820 - 34
    insulator_string(d, ins_x, 250, 11, broken_at=4)
    insulator_string(d, 820 + 34, 250, 11)
    # 金具（横担与绝缘子串连接处）锈蚀
    d.rectangle([ins_x - 16, 236, ins_x + 16, 252], fill=(140, 96, 52), outline=(104, 70, 36), width=2)

    # 导线
    d.line([(300, 330), (820, 262)], fill=(60, 66, 72), width=3)
    d.line([(820 + 34, 262), (1560, 320)], fill=(60, 66, 72), width=3)

    dets = [
        {"cls": "insulator_broken", "bbox": [ins_x - 26, 250 + 4 * 13 - 16, ins_x + 26, 250 + 4 * 13 + 16], "score": 0.93},
        {"cls": "fitting_corrosion", "bbox": [ins_x - 20, 232, ins_x + 20, 256], "score": 0.81},
    ]
    return img, dets


def make_channel_scene() -> tuple[Image.Image, list[dict]]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    sky(d, top=(166, 202, 232), bottom=(226, 238, 244))
    hill(d, 700, (150, 168, 148))
    d.rectangle([0, 860, W, H], fill=(126, 146, 118))

    # 导线横穿画面
    d.line([(0, 330), (900, 300)], fill=(58, 64, 70), width=4)
    d.line([(900, 300), (1600, 336)], fill=(58, 64, 70), width=4)
    # 异物（风筝/塑料薄膜）挂在导线上
    d.polygon([(1000, 300), (1044, 340), (1000, 380), (956, 340)], fill=(206, 74, 60), outline=(150, 46, 36))
    d.line([(1000, 380), (1012, 436)], fill=(80, 84, 90), width=2)

    # 树障：树冠逼近导线
    for i, x in enumerate(range(360, 860, 66)):
        h = 250 + (i % 3) * 24
        d.rectangle([x - 6, 860 - h, x + 6, 860], fill=(96, 74, 52))
        d.ellipse([x - 52, 860 - h - 78, x + 52, 860 - h + 34], fill=(58, 108, 62), outline=(44, 86, 48))

    dets = [
        {"cls": "foreign_object", "bbox": [944, 288, 1056, 448], "score": 0.88},
        {"cls": "vegetation_overreach", "bbox": [330, 470, 880, 800], "score": 0.79},
    ]
    return img, dets


def make_field_scene() -> tuple[Image.Image, list[dict]]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 260], fill=(198, 216, 232))  # 天空
    d.rectangle([0, 260, W, H], fill=(112, 140, 84))   # 田块

    # 作物行
    for y in range(300, H, 46):
        for x in range(30, W, 54):
            r = 15 + random.randint(-3, 3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(74 + random.randint(-12, 12), 128, 66))

    # 病害斑块（黄褐枯死区）
    disease_box = [900, 420, 1360, 800]
    d.ellipse(disease_box, fill=(186, 158, 74))
    for _ in range(260):
        x = random.randint(disease_box[0], disease_box[2])
        y = random.randint(disease_box[1], disease_box[3])
        r = random.randint(6, 16)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(160, 122, 56))

    # 虫害斑块（缺刻/孔洞）
    pest_box = [180, 560, 520, 860]
    for _ in range(220):
        x = random.randint(pest_box[0], pest_box[2])
        y = random.randint(pest_box[1], pest_box[3])
        r = random.randint(4, 11)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(58, 96, 58))

    img = img.filter(ImageFilter.GaussianBlur(0.6))

    dets = [
        {"cls": "crop_disease", "bbox": disease_box, "score": 0.86},
        {"cls": "crop_pest", "bbox": pest_box, "score": 0.72},
    ]
    return img, dets


def make_fire_scene() -> tuple[Image.Image, list[dict]]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    sky(d, top=(150, 168, 190), bottom=(214, 208, 196))
    hill(d, 760, (98, 106, 92))
    hill(d, 840, (74, 82, 70))

    # 烟雾羽流
    for i in range(46):
        t = i / 46
        cx = 1080 + int(90 * math.sin(i * 0.8))
        cy = 740 - int(560 * t)
        r = int(36 + 120 * t)
        shade = int(196 - 40 * t)
        d.ellipse([cx - r, cy - r * 0.72, cx + r, cy + r * 0.72], fill=(shade, shade - 6, shade - 12))
    img = img.filter(ImageFilter.GaussianBlur(3.0))

    d2 = ImageDraw.Draw(img)
    # 明火
    d2.polygon([(1030, 790), (1080, 690), (1114, 790)], fill=(232, 128, 44))
    d2.polygon([(1052, 790), (1080, 726), (1100, 790)], fill=(246, 196, 76))

    dets = [{"cls": "fire_smoke", "bbox": [880, 150, 1320, 860], "score": 0.91}]
    return img, dets


SCENES = {
    "demo_tower_01": make_tower_scene,
    "demo_channel_01": make_channel_scene,
    "demo_field_01": make_field_scene,
    "demo_fire_01": make_fire_scene,
}


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, fn in SCENES.items():
        img, dets = fn()
        img_path = ASSETS / f"{name}.png"
        img.save(img_path)
        payload = {
            "image_id": name,
            "model_version": "synthetic-demo-v1",
            "note": "合成演示图的离线检测结果，用于验证链路连通性，不代表真实模型精度",
            "image_size": [img.width, img.height],
            "detections": dets,
        }
        (ASSETS / f"{name}.detections.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{name}: {len(dets)} 个标注，{img.width}x{img.height} -> {img_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
