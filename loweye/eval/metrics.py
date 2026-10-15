"""评估指标：全部自己实现，不引第三方评估库。

理由：这些数字会作为「可复现结论」写进参赛材料，口径必须能被逐行读懂——
Recall@k 的分母是「有多少问题」，MRR 用的是第一条命中，AP 用全点插值（VOC2010+ 口径）。
把它们藏在某个库的默认参数里，答辩时说不清是怎么算的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


# ----------------------------------------------------------------- 检索指标


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """前 k 条里是否命中期望条款。ranks[i] 为第 i 个问题的命中排名（1 起，未命中为 None）。"""
    if not ranks:
        return 0.0
    hit = sum(1 for r in ranks if r is not None and r <= k)
    return hit / len(ranks)


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean Reciprocal Rank：第一条命中的排名倒数均值。未命中的问题贡献 0。"""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def percentile(values: Iterable[float], p: float) -> float:
    """最近秩法（nearest-rank）求分位数，不插值。

    时延这种右偏分布，P95 用最近秩法比线性插值保守，也更好解释：
    「95% 的请求快于这个数」。
    """
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if not 0 < p <= 100:
        raise ValueError(f"p 必须在 (0, 100]，收到 {p}")
    idx = max(1, min(len(xs), int(round(p / 100 * len(xs) + 0.5)) - 1))
    return xs[idx]


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


# ----------------------------------------------------------------- 检测指标


@dataclass
class DetBox:
    """评估用的检测框（与 loweye.schemas.BBox 解耦，便于直接吃标注文件）。"""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def iou(a: DetBox, b: DetBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(preds: Sequence[DetBox], gts: Sequence[DetBox], iou_threshold: float = 0.5) -> float:
    """单个类别的 AP（全点插值，VOC2010+ 之后的通行口径）。

    贪心匹配：按分数降序，每个预测框匹配一个 IoU 最高且未被占用的同类别真值框。
    没有真值框时返回 0（该类在指标里不参与 mAP 平均，由调用方过滤）。
    """
    if not gts:
        return 0.0
    if not preds:
        return 0.0
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    matched = [False] * len(gts)
    tp = [0] * len(preds)
    fp = [0] * len(preds)
    for rank, pi in enumerate(order):
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if matched[gi]:
                continue
            v = iou(preds[pi], g)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched[best_gi] = True
            tp[rank] = 1
        else:
            fp[rank] = 1

    cum_tp = cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / len(gts))

    # 全点插值：对每个召回台阶取其后（含自身）最大精度
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(precisions) - 1, -1, -1):
        pass
    # 从右向左做精度包络
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    for i, r in enumerate(recalls):
        ap += precisions[i] * (r - prev_recall)
        prev_recall = r
    return ap


def detection_report(
    preds: Sequence[DetBox],
    gts: Sequence[DetBox],
    *,
    iou_threshold: float = 0.5,
) -> dict:
    """按类别汇总：每类 AP、漏检率（1 - 召回），以及 mAP。

    注意：这套指标的结论强弱取决于标注质量。演示图是合成图 + 自编标注，
    在它上面算出来的 mAP 只能证明「评测脚本能跑通、口径自洽」，
    不能用来声称模型精度——真实精度必须在实拍标注集上测（见 README）。
    """
    classes = sorted({g.cls for g in gts} | {p.cls for p in preds})
    per_class: dict[str, dict] = {}
    aps: list[float] = []
    for cls in classes:
        c_preds = [p for p in preds if p.cls == cls]
        c_gts = [g for g in gts if g.cls == cls]
        ap = average_precision(c_preds, c_gts, iou_threshold)
        # 贪心匹配一次拿召回（与 AP 同一套匹配规则）
        matched = 0
        used = [False] * len(c_gts)
        for pi in sorted(range(len(c_preds)), key=lambda i: c_preds[i].score, reverse=True):
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(c_gts):
                if used[gi]:
                    continue
                v = iou(c_preds[pi], g)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= iou_threshold and best_gi >= 0:
                used[best_gi] = True
                matched += 1
        recall = rate(matched, len(c_gts))
        per_class[cls] = {
            "n_gt": len(c_gts),
            "n_pred": len(c_preds),
            "ap": round(ap, 4),
            "recall": round(recall, 4),
            "miss_rate": round(1 - recall, 4) if c_gts else None,
        }
        if c_gts:
            aps.append(ap)
    return {
        "iou_threshold": iou_threshold,
        "mAP": round(sum(aps) / len(aps), 4) if aps else 0.0,
        "classes_evaluated": len(aps),
        "per_class": per_class,
    }
