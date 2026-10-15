"""Deterministic delivery report for the two trained detectors.

Every number written into the document is read from an artifact recovered from the
training host (metadata / test receipt / independent ONNX evaluation / per-epoch
JSONL / manifest / recovery receipt). Nothing is estimated, rounded to a friendlier
value, or transcribed by hand.

A missing input is printed as an explicit "文件缺失" marker and recorded in the
``gaps`` list of the JSON sidecar; it is never silently replaced by a plausible
value.

Usage:
    python tools/write_model_report.py --out-dir out/inspection_gpu \
        --dest docs/proposal/06d_trained_detectors.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = (
    {"task": "fire_smoke", "dataset": "dfire", "size": 640,
     "title": "fire_smoke — D-Fire 烟火/明火检测",
     "role": "低空巡检中的烟火与明火告警"},
    {"task": "crop_pest", "dataset": "ip102", "size": 800,
     "title": "crop_pest — IP102 农业害虫检测",
     "role": "农田虫情普查与靶向施药辅助"},
)

MISSING = "——（文件缺失）"
# Absolute tolerance for the two-independent-implementations mAP cross-check. It is a
# disclosure threshold, not a gate: both numbers are always printed, and exceeding it
# raises a warning in the document and in the JSON sidecar instead of hiding the gap.
PARITY_TOLERANCE = 0.005


def load_json(path: Path):
    if not path.exists():
        return None, f"missing:{path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"unreadable:{path.name}:{type(exc).__name__}"


def load_rows(path: Path):
    if not path.exists():
        return [], f"missing:{path.name}"
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except (OSError, ValueError) as exc:
        return [], f"unreadable:{path.name}:{type(exc).__name__}"
    return rows, None


def num(value, digits=6):
    if value is None or isinstance(value, bool):
        return MISSING if value is None else str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def human_bytes(value):
    if not isinstance(value, int):
        return MISSING
    return f"{value} ({value / 1048576:.1f} MiB)"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def collect(out_dir: Path, spec, gaps):
    """Read every artifact this task is expected to have; record what is missing."""
    task, dataset = spec["task"], spec["dataset"]
    files = {
        "metadata": out_dir / f"{task}_fcos_metadata.json",
        "receipt": out_dir / f"{task}_test.json",
        "onnx_eval": out_dir / f"{task}_onnx_eval.json",
        "run": out_dir / f"{task}_run.json",
        "manifest": out_dir / f"{task}_manifest.json",
        "license": out_dir / f"{dataset}_source_license.json",
        "transport": out_dir / f"{dataset}_transport.json",
        "recovery": out_dir / f"{task}_recovery.json",
        "epochs": out_dir / f"{task}_epochs.jsonl",
    }
    data = {}
    for key, path in files.items():
        if key == "epochs":
            data[key], note = load_rows(path)
        else:
            data[key], note = load_json(path)
        if note:
            gaps.append(f"{task}:{note}")
    return data


def epoch_view(rows):
    """Summarise the per-epoch curve without inventing missing epochs."""
    scored = [row for row in rows if isinstance(row.get("map"), (int, float))]
    seconds = [row["seconds"] for row in rows if isinstance(row.get("seconds"), (int, float))]
    view = {
        "epochs_logged": len(rows),
        "first_epoch": rows[0]["epoch"] if rows else None,
        "last_epoch": rows[-1]["epoch"] if rows else None,
        "first_map": rows[0].get("map") if rows else None,
        "last_map": rows[-1].get("map") if rows else None,
        "last_map50": rows[-1].get("map50") if rows else None,
        "last_loss": rows[-1].get("loss") if rows else None,
        "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
        "total_seconds": sum(seconds) if seconds else None,
    }
    if scored:
        best = max(scored, key=lambda row: row["map"])
        view["best_epoch"] = best["epoch"]
        view["best_map"] = best["map"]
        view["best_map50"] = best.get("map50")
    return view


def per_class_rows(onnx_eval, limit=5):
    """Top/bottom classes by the independent evaluator, evaluable classes only."""
    per_class = (onnx_eval or {}).get("per_class") or {}
    scored = [(name, value.get("map"), value.get("gt"))
              for name, value in per_class.items()
              if value.get("evaluable") and isinstance(value.get("map"), (int, float))]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit], scored[-limit:] if len(scored) > limit else [], len(scored), len(per_class)


def task_block(out_dir: Path, spec, data, gaps):
    task = spec["task"]
    metadata = data["metadata"] or {}
    onnx_eval = data["onnx_eval"] or {}
    manifest = data["manifest"] or {}
    receipt = data["receipt"] or {}
    run = data["run"] or {}
    license_info = data["license"] or {}
    transport = data["transport"] or {}
    recovery = data["recovery"] or []
    view = epoch_view(data["epochs"])
    nonfinite = (metadata.get("training") or {}).get("nonfinite") or {}
    skipped_batches = nonfinite.get("skipped_batches")
    if skipped_batches is None:
        counted = [row.get("nonfinite_batches") for row in data["epochs"]
                   if isinstance(row.get("nonfinite_batches"), int)]
        skipped_batches = max(counted) if counted else None

    statistics = manifest.get("statistics") or {}
    coverage = manifest.get("class_coverage") or {}
    quarantine = manifest.get("quarantine") or {}
    signature = run.get("signature") or {}

    trainer_map = ((metadata.get("test_metrics") or {}).get("map"))
    trainer_map50 = ((metadata.get("test_metrics") or {}).get("map50"))
    trainer_op = ((metadata.get("test_metrics") or {}).get("operating_point") or {}).get("overall") or {}
    independent_map = (onnx_eval.get("coco") or {}).get("map")
    independent_map50 = (onnx_eval.get("coco") or {}).get("map50")
    independent_op = onnx_eval.get("operating_point") or {}
    delta = (abs(trainer_map - independent_map)
             if isinstance(trainer_map, (int, float)) and isinstance(independent_map, (int, float)) else None)

    top, bottom, evaluable, total_classes = per_class_rows(onnx_eval)

    lines = [f"### {spec['title']}", "",
             f"用途定位：{spec['role']}。推理尺寸 {spec['size']}×{spec['size']}，输出为 letterbox 坐标系下的 "
             f"xyxy 框 + 1 基类别 id + 分数，需按 metadata 中的预处理说明还原到原图坐标。", ""]

    lines += ["#### 数据来源与划分", "",
              table(["项目", "值"], [
                  ["数据源", f"{license_info.get('dataset', MISSING)}（{spec['dataset']}）"],
                  ["官方来源", str(license_info.get("source", license_info.get("download_url", MISSING)))],
                  ["归档字节", human_bytes(transport.get("archive_bytes", license_info.get("archive_bytes")))],
                  ["归档 sha256", str(transport.get("sha256", license_info.get("sha256", MISSING)))],
                  ["类别数", str(len(manifest["classes"])) if manifest.get("classes") else MISSING],
                  ["标注格式", json.dumps(manifest["annotation_formats"], ensure_ascii=False) if manifest.get("annotation_formats") else MISSING],
                  ["类别 id 基数（源）", str(manifest.get("voc_class_id_base", MISSING))],
                  ["划分策略", str(manifest.get("split_strategy", MISSING))],
                  ["seed", str(manifest.get("seed", MISSING))],
                  ["按字节去重移除", str((manifest.get("dedup") or {}).get("removed_count", MISSING))],
                  ["跨划分同源分组（应空）", json.dumps((manifest.get("source_group_audit") or {}).get("cross_split_groups"), ensure_ascii=False) if (manifest.get("source_group_audit") or {}).get("cross_split_groups") is not None else MISSING],
              ]), ""]

    if statistics:
        lines += ["各划分规模（来自 manifest.statistics，不是原始源计数）：", "",
                  table(["划分", "图像数", "无目标图像", "同源分组数", "目标框合计"],
                        [[split, str(value.get("images", MISSING)), str(value.get("empty_images", MISSING)),
                          str(value.get("source_groups", MISSING)),
                          str(sum((value.get("objects") or {}).values()) if value.get("objects") else MISSING)]
                         for split, value in statistics.items()]), ""]
    else:
        lines += [f"各划分规模：{MISSING}（{task}_manifest.json 缺失，无法给出划分计数）", ""]

    if quarantine:
        lines += ["隔离的无效源样本（未移动或修改任何源文件）：", "",
                  table(["项目", "值"], [
                      ["启用", num(quarantine.get("enabled"))],
                      ["数量", str(quarantine.get("count", MISSING))],
                      ["按原划分", json.dumps(quarantine.get("by_original_split") or {}, ensure_ascii=False)],
                      ["按原因", json.dumps(quarantine.get("by_reason") or {}, ensure_ascii=False)],
                  ]), ""]

    lines += ["#### 训练配置（读自 run.json 的 signature）", "",
              table(["项目", "值"], [
                  ["优化轮数", str(signature.get("epochs", MISSING))],
                  ["批大小", str(signature.get("batch", MISSING))],
                  ["学习率", str(signature.get("lr", MISSING))],
                  ["数据加载 worker", str(signature.get("workers", MISSING))],
                  ["随机种子", str(signature.get("seed", MISSING))],
                  ["训练样本上限", str(signature.get("limit_train", MISSING))],
                  ["验证样本上限", str(signature.get("limit_val", MISSING))],
                  ["manifest sha256", str(signature.get("manifest_sha256", MISSING))],
                  ["冒烟运行", num(run.get("smoke"))],
                  ["架构", str(metadata.get("architecture", MISSING))],
                  ["实现许可", str(metadata.get("implementation_license", MISSING))],
                  ["torch / onnxruntime", json.dumps(metadata["dependencies"], ensure_ascii=False) if metadata.get("dependencies") else MISSING],
              ]), ""]

    selection = metadata.get("selection") or {}
    val_metrics = metadata.get("val_metrics") or {}
    lines += ["#### 模型选择", "",
              table(["项目", "值"], [
                  ["选择划分", str(selection.get("split", MISSING))],
                  ["选择准则", str(selection.get("criterion", MISSING))],
                  ["入选轮次", str(selection.get("epoch", MISSING))],
                  ["入选轮 mAP(0.50:0.95)", num(val_metrics.get("map"))],
                  ["入选轮 mAP@.50", num(val_metrics.get("map50"))],
              ]), "",
              "测试集不参与任何模型选择：训练器只在保存最优权重之后才触碰一次测试划分，并写入不可变收据"
              "（重复评估会被 signature + checkpoint sha256 拒绝）。", ""]

    lines += ["#### 独立测试（训练器实现：torchmetrics + pycocotools）", "",
              table(["指标", "值"], [
                  ["测试状态", str(metadata.get("test_status", MISSING))],
                  ["mAP(0.50:0.95)", num(trainer_map)],
                  ["mAP@.50", num(trainer_map50)],
                  ["全类召回", num(trainer_op.get("recall"))],
                  ["误检占报出比例", num(trainer_op.get("false_positive_fraction"))],
                  ["每图误报数", num(trainer_op.get("false_alarms_per_image"))],
                  ["收据评估次数", str(receipt.get("evaluations", MISSING))],
                  ["收据 checkpoint sha256", str(receipt.get("checkpoint_sha256", MISSING))[:16] or MISSING],
              ]), ""]

    counts = onnx_eval.get("counts") or {}
    protocol = onnx_eval.get("protocol") or {}
    runtime = onnx_eval.get("runtime") or {}
    timing = onnx_eval.get("timing") or {}
    preprocessing = onnx_eval.get("preprocessing_parity") or {}
    lines += ["#### 独立复评（本项目 ONNX 评测器：onnxruntime + 直接调用 pycocotools COCOeval）", "",
              table(["指标", "值"], [
                  ["测试图像数", str(protocol.get("images", MISSING))],
                  ["类别数", str(protocol.get("classes", MISSING))],
                  ["mAP(0.50:0.95)", num(independent_map)],
                  ["mAP@.50", num(independent_map50)],
                  ["真值框总数", str(counts.get("ground_truth_boxes", MISSING))],
                  ["0.001 阈值以下检出数", str(counts.get("detections_at_floor", MISSING))],
                  ["可评估类别数", str(evaluable) if onnx_eval else MISSING],
                  ["不可评估类别数", str(len(counts.get("non_evaluable_classes") or [])) if onnx_eval else MISSING],
                  ["全类召回", num((independent_op.get("overall") or {}).get("recall") if independent_op.get("overall") else independent_op.get("recall"))],
                  ["每图误报数", num((independent_op.get("overall") or {}).get("false_alarms_per_image") if independent_op.get("overall") else independent_op.get("false_alarms_per_image"))],
                  ["onnxruntime", str(runtime.get("onnxruntime", MISSING))],
                  ["执行提供者", json.dumps(runtime["session_providers"], ensure_ascii=False) if runtime.get("session_providers") else MISSING],
                  ["预处理一致性（与训练器数据集类逐元素比较）", json.dumps(preprocessing, ensure_ascii=False) if preprocessing else MISSING],
                  ["耗时", f"{timing.get('seconds', MISSING)} 秒 / {timing.get('seconds_per_image', MISSING)} 秒每图"],
                  ["ONNX 字节 / sha256", f"{onnx_eval.get('artifact', {}).get('onnx_bytes', MISSING)} / {str(onnx_eval.get('artifact', {}).get('onnx_sha256', MISSING))[:16]}"],
              ]), ""]

    if top:
        lines += ["独立复评可评估类别中 mAP 最高与最低各若干：", "",
                  table(["类别", "mAP", "真值框数", "方向"],
                        [[name, num(value), str(gt), "最高"] for name, value, gt in top]
                        + [[name, num(value), str(gt), "最低"] for name, value, gt in bottom]), ""]

    lines += ["#### 逐轮曲线（训练器每轮 fsync 写入的 JSONL）", "",
              table(["项目", "值"], [
                  ["已记录轮数", str(view["epochs_logged"])],
                  ["首轮 mAP", num(view["first_map"])],
                  ["末轮 mAP", num(view["last_map"])],
                  ["末轮 mAP@.50", num(view["last_map50"])],
                  ["末轮 loss", num(view["last_loss"], 4)],
                  ["最优轮（按 val mAP）", str(view.get("best_epoch", MISSING))],
                  ["最优轮 mAP", num(view.get("best_map"))],
                  ["平均每轮耗时", num(view["mean_seconds"], 1) + " 秒"],
                  ["累计训练耗时", num(view["total_seconds"], 1) + " 秒"],
              ]), ""]

    if recovery:
        rows = []
        for entry in recovery:
            local = Path(entry.get("local", ""))
            size = local.stat().st_size if local.exists() else None
            rows.append([local.name or MISSING, human_bytes(size), str(entry.get("sha256", MISSING))[:16]])
        lines += ["#### 回收产物（本地重建自远端，逐文件与远端 sha256sum 比对）", "",
                  table(["文件", "本地字节", "sha256 前 16 位"], rows), ""]
    else:
        lines += [f"#### 回收产物{chr(10)}{chr(10)}{MISSING}（{task}_recovery.json 缺失，无法列出已校验产物）", ""]

    return "\n".join(lines), {
        "task": task, "dataset": spec["dataset"], "size": spec["size"],
        "trainer_test_map": trainer_map, "trainer_test_map50": trainer_map50,
        "independent_onnx_map": independent_map, "independent_onnx_map50": independent_map50,
        "map_absolute_delta": delta, "parity_warning": bool(delta is not None and delta > PARITY_TOLERANCE),
        "evaluable_classes": evaluable, "manifest_classes": total_classes,
        "test_images": protocol.get("images"), "epochs": view,
        "best_epoch": view.get("best_epoch"), "best_val_map": view.get("best_map"),
        "class_coverage_policy": coverage.get("policy"),
        "missing_class_ids": coverage.get("missing_ids"),
        "missing_class_names": coverage.get("missing_names"),
        "nonfinite_skipped": skipped_batches, "nonfinite_policy": nonfinite.get("policy"),
    }


def build_document(out_dir: Path, generated_at: str | None = None):
    gaps: list[str] = []
    blocks, summaries = [], []
    for spec in TASKS:
        data = collect(out_dir, spec, gaps)
        block, summary = task_block(out_dir, spec, data, gaps)
        blocks.append(block)
        summaries.append(summary)

    warnings = [item["task"] for item in summaries if item["parity_warning"]]
    header = [
        "# 训练交付：fire_smoke 与 crop_pest 检测器", "",
        "本文件由 `tools/write_model_report.py` 从回收制品自动生成。所有数字直接读自 JSON/JSONL，"
        "不手工誊写、不四舍五入到更「好看」的值；缺失的输入会显式标注为缺失并列在文末的「证据缺口」中。", "",
        f"生成时间：{generated_at or dt.datetime.now().isoformat(timespec='seconds')}", "",
        "## 0. 结论速览", "",
    ]
    overview = []
    for item in summaries:
        overview.append([
            item["task"],
            num(item["trainer_test_map"]),
            num(item["independent_onnx_map"]),
            num(item["map_absolute_delta"]),
            str(item["test_images"] if item["test_images"] is not None else MISSING),
            str(item["best_epoch"] if item["best_epoch"] is not None else MISSING),
        ])
    header += [table(["任务", "训练器自测 mAP", "独立 ONNX 复评 mAP", "绝对差", "测试图像数", "最优轮"], overview), "",
               f"两套实现相互独立：训练器走 torchmetrics(backend=pycocotools)，复评走 onnxruntime + 直接调用 "
               f"pycocotools COCOeval；两者共用同一测试划分与同一 ONNX 权重。绝对差超过 "
               f"{PARITY_TOLERANCE} 会被标记为告警（本次：{('、'.join(warnings) if warnings else '无')}）。", "",
               "**本文不含任何跨数据集、跨场景或跨机型的泛化结论**：数字只描述下列测试划分。", ""]

    gap_section = ["## 附录：证据缺口", ""]
    if gaps:
        gap_section += ["以下输入在生成时不存在或不可读，对应字段已在正文标注为缺失：", ""] + \
                       [f"- `{item}`" for item in gaps]
    else:
        gap_section += ["无：本报告全部输入均存在且可读。"]
    gap_section += ["", "各任务的 mAP 只对测试划分有效；训练与验证之间的差距、部署现场的域偏移均未评估。", ""]

    limitations = ["## 附录：能力边界与许可（逐条附来源）", ""]
    for spec, item in zip(TASKS, summaries):
        missing_ids = item["missing_class_ids"]
        limitations += [
            f"**{spec['task']}**",
            "",
            f"- 类别覆盖策略（来源 `{spec['task']}_manifest.json` 的 `class_coverage.policy`）："
            f"`{item['class_coverage_policy']}`。" if item["class_coverage_policy"] is not None else f"- 类别覆盖策略：{MISSING}（`{spec['task']}_manifest.json` 缺失）。",
            (f"- 源数据中不存在任何标注的类别（无法训练、也无法评估）：id {missing_ids}，"
             f"名称 {item['missing_class_names']}。" if missing_ids else
             ("- 源数据中不存在标注的类别：无。" if item["class_coverage_policy"] is not None else f"- 源数据中不存在标注的类别：{MISSING}（清单缺失，无法判断）。")),
            f"- 数据许可与再分发限制：以 `{spec['dataset']}_source_license.json` 的原文为准，"
            f"本文件不替代法律意见。",
            ("- 训练期非有限损失处理：未记录（元数据缺少 `training.nonfinite`，无法断言是否发生过）。"
             if item["nonfinite_skipped"] is None else
             "- 训练期非有限损失处理：无被跳过的批次（策略 "
             f"`{item['nonfinite_policy']}`，跳过 0 次）。" if not item["nonfinite_skipped"] else
             f"- 训练期非有限损失处理：策略 `{item['nonfinite_policy']}`，跳过 "
             f"{item['nonfinite_skipped']} 个批次；被跳过的批次从未参与梯度更新，该数字来自逐轮记录。"),
            "",
        ]
    limitations += [
        "以上「不可训练 / 不可评估」的表述来自准备阶段的源侧审计记录，不是对模型能力的推测。", ""]

    return "\n".join(header + blocks + limitations + gap_section), \
        {"generated_at": generated_at, "tasks": summaries, "gaps": gaps,
         "parity_tolerance": PARITY_TOLERANCE, "parity_warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "inspection_gpu")
    parser.add_argument("--dest", type=Path, default=ROOT / "docs" / "proposal" / "06d_trained_detectors.md")
    parser.add_argument("--json", type=Path, default=None,
                       help="machine-readable sidecar (default: <out-dir>/model_report.json)")
    parser.add_argument("--allow-partial", action="store_true",
                       help="write the document even when required model evidence is absent")
    args = parser.parse_args()

    document, payload = build_document(args.out_dir)
    required = [item for item in payload["tasks"]
                if item["trainer_test_map"] is None and item["independent_onnx_map"] is None]
    if required and not args.allow_partial:
        for item in required:
            print(f"refusing to write: {item['task']} has neither a trainer test receipt nor an ONNX evaluation")
        print("re-run with --allow-partial to emit an explicit gap report instead")
        return 2

    # docs/proposal/ 下所有文档都是 LF；显式 newline="\n" 免得 Windows 文本模式
    # 把生成出来的 06d 写成 CRLF，让它在目录里成为唯一的异类。
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with args.dest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    sidecar = args.json or (args.out_dir / "model_report.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"report": str(args.dest), "sidecar": str(sidecar),
                      "gaps": len(payload["gaps"]),
                      "parity_warnings": [item["task"] for item in payload["tasks"] if item["parity_warning"]]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
