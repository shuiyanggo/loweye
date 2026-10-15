#!/usr/bin/env python3
"""Independent evaluation of an exported detector artifact.

Why this exists
---------------
`train_inspection_fcos.py` scores the live PyTorch module through the
torchmetrics wrapper during and after training. That answers "how good is the
model I trained", but not "is the artifact I am shipping the same detector".
This script answers the second question by scoring the *exported ONNX file*:

  * inference runs in onnxruntime, not in PyTorch;
  * the split metric is produced by pycocotools' COCOeval directly, not by the
    torchmetrics wrapper that training used;
  * preprocessing is re-implemented here, and `--check-preprocessing` proves
    numerically that it agrees with the training dataset.

Disagreement between the two paths is evidence that the export changed the
model's behaviour; absent that, the numbers here are the ones that describe the
shipped file.

Usage on the training host
--------------------------
  /root/miniconda3/bin/python tools/gpu/evaluate_onnx_detector.py \
      --onnx /root/autodl-tmp/loweye/models/crop_pest_fcos_800.onnx \
      --manifest /root/loweye/data/crop_pest_manifest.json \
      --split test --out /root/loweye/logs/crop_pest_onnx_test.json

  # plumbing check against the smoke artifact before trusting a long run
  ... --onnx .../models_smoke/crop_pest_fcos_800.onnx --split val --limit 8 \
      --check-preprocessing 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

SCORE_FLOOR = 0.001          # the detector's own score_thresh; nothing is cut below it
OPERATING_CONF = 0.35        # fixed operating point reported alongside AP
OPERATING_IOU = 0.5
PAD_FILL = 114


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox(image, size: int, fill: int = PAD_FILL):
    """Scale by size/max(width, height) and centre-pad onto a size x size canvas.

    Independent implementation of the geometry in
    `InspectionDataset.__getitem__`; `--check-preprocessing` proves the two
    agree numerically instead of asking the reader to trust the symmetry.
    """
    from PIL import Image

    width, height = image.size
    if min(width, height, size) <= 0:
        raise ValueError("image dimensions and size must be positive")
    scale = size / max(width, height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR),
                 ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def to_nchw(canvas):
    import numpy as np
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


def box_iou(a, b) -> float:
    """IoU of two xyxy boxes. Written here so no training-code helper is reused."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def greedy_counts(boxes, labels, scores, gt_boxes, gt_labels, num_classes: int) -> dict:
    """Same-class score-ordered greedy one-to-one matching at the operating point."""
    counts = {c: {"gt": 0, "tp": 0, "fp": 0} for c in range(1, num_classes + 1)}
    for label in gt_labels:
        label = int(label)
        if label in counts:
            counts[label]["gt"] += 1
    used: set[int] = set()
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    for index in order:
        label = int(labels[index])
        if label not in counts or float(scores[index]) < OPERATING_CONF:
            continue
        best_iou, best_gt = 0.0, -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in used or int(gt_labels[gt_index]) != label:
                continue
            overlap = box_iou(boxes[index], gt_box)
            if overlap > best_iou:
                best_iou, best_gt = overlap, gt_index
        if best_gt >= 0 and best_iou >= OPERATING_IOU:
            used.add(best_gt)
            counts[label]["tp"] += 1
        else:
            counts[label]["fp"] += 1
    return counts


def summarize_counts(counts: dict, classes: list, images: int) -> dict:
    def rates(entry):
        gt, tp, fp = entry["gt"], entry["tp"], entry["fp"]
        return {**entry, "fn": gt - tp,
                "recall": tp / gt if gt else None,
                "false_positive_fraction": fp / (tp + fp) if gt and tp + fp else None,
                "false_alarms_per_image": fp / images if gt and images else None}
    total = {key: sum(entry[key] for entry in counts.values()) for key in ("gt", "tp", "fp")}
    return {"conf": OPERATING_CONF, "iou": OPERATING_IOU,
            "matching": "same-class score-ordered greedy one-to-one",
            "images": images, "overall": rates(total),
            "per_class": {name: rates(counts[index]) for index, name in enumerate(classes, 1)}}


def coco_evaluate(images, annotations, detections, category_ids):
    """Run pycocotools COCOeval over already-collected numpy results."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "low-altitude-eye independent ONNX evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": str(c)} for c in category_ids],
    }
    coco_gt.createIndex()
    if not detections:
        return {"available": False, "reason": "no detection above the score floor"}
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()

    def mean_ap(slice_):
        values = slice_[slice_ > -1]
        return float(values.mean()) if values.size else None

    precision = evaluator.eval["precision"]        # [T, R, K, A, M]
    recall = evaluator.eval["recall"]              # [T, K, A, M]
    cat_ids = list(evaluator.params.catIds)
    per_class = {}
    for index, category in enumerate(cat_ids):
        per_class[int(category)] = {
            "map": mean_ap(precision[:, :, index, 0, 2]),
            "map50": mean_ap(precision[0, :, index, 0, 2]),
            "ar100": float(recall[:, index, 0, 2].mean()) if recall[:, index, 0, 2].size else None,
        }
    return {"available": True,
            "map": mean_ap(precision[:, :, :, 0, 2]),
            "map50": mean_ap(precision[0, :, :, 0, 2]),
            "map75": mean_ap(precision[5, :, :, 0, 2]),
            "per_class_by_id": per_class,
            "max_dets": [int(v) for v in evaluator.params.maxDets],
            "iou_thresholds": [float(v) for v in evaluator.params.iouThrs],
            "area_range_all": [float(v) for v in evaluator.params.areaRng[0]]}


def geometry_transform(boxes, width: int, height: int, size: int):
    """Map manifest boxes (original pixels) onto the letterbox canvas the ONNX sees.

    The training dataset builds its targets with train_inspection_fcos.transform_boxes,
    so the exported graph emits boxes in that canvas space. Borrowing the training
    helper verbatim keeps both sides in one coordinate space instead of re-deriving
    the letterbox maths here, where it can silently drift.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_inspection_fcos as trainer            # training-side reference
    return trainer.transform_boxes(boxes, width, height, size, False)


def check_preprocessing(manifest_path: Path, samples, size: int, count: int) -> dict:
    """Prove this script's letterbox equals the training dataset's tensor."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import train_inspection_fcos as trainer            # training-side reference

    dataset = trainer.InspectionDataset(samples, size, train=False)
    probed = min(count, len(samples))
    worst = 0.0
    for index in range(probed):
        with_image = dataset[index][0].numpy()
        sample = samples[index]
        from PIL import Image
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB")
        mine = to_nchw(letterbox(image, size))[0]
        worst = max(worst, float(np.abs(mine - with_image).max()))
    return {"images_compared": probed, "max_abs_difference": worst,
            "reference": "train_inspection_fcos.InspectionDataset", "manifest": str(manifest_path),
            "agrees_within_1e-6": worst <= 1e-6}


def build_session(onnx_path: Path, threads: int, providers: list):
    import onnxruntime as ort
    available = ort.get_available_providers()
    chosen = [name for name in providers if name in available]
    if not chosen:
        chosen = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available[:1]
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=chosen)
    return ort, session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    parser.add_argument("--check-preprocessing", type=int, default=0, metavar="N",
                        help="compare letterbox output against the training dataset on N images")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--providers", nargs="+",
                        default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    import numpy as np

    onnx_path, manifest_path = Path(args.onnx), Path(args.manifest)
    for path in (onnx_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = list(data["classes"])
    samples = list(data[args.split])
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit(f"split {args.split} is empty")

    ort_module, session = build_session(onnx_path, args.threads, args.providers)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "images" or inputs[0].type != "tensor(float)":
        raise SystemExit(f"ONNX input contract mismatch: {[(v.name, v.type, v.shape) for v in inputs]}")
    if [v.name for v in outputs] != ["boxes", "labels", "scores"]:
        raise SystemExit(f"ONNX output contract mismatch: {[v.name for v in outputs]}")
    input_shape = list(inputs[0].shape)
    size = int(input_shape[2])
    if input_shape != [1, 3, size, size]:
        raise SystemExit(f"ONNX input shape is not a fixed square image: {input_shape}")

    parity = check_preprocessing(manifest_path, samples, size, args.check_preprocessing) \
        if args.check_preprocessing else None
    if parity and not parity["agrees_within_1e-6"]:
        raise SystemExit(f"preprocessing disagrees with the training dataset: {parity}")

    from PIL import Image

    coco_images, coco_annotations, coco_detections = [], [], []
    counts = {c: {"gt": 0, "tp": 0, "fp": 0} for c in range(1, len(classes) + 1)}
    annotation_id = 0
    total_gt, total_det, skipped_zero_area = 0, 0, 0
    started = time.time()
    for index, sample in enumerate(samples):
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        coco_images.append({"id": index, "width": size, "height": size,
                            "file_name": Path(sample["image"]).name})
        # Ground truth has to live in the same space as the detections: the graph
        # emits letterbox-canvas boxes, so transform the manifest boxes exactly as
        # the training dataset does before anything is scored.
        gt_boxes, gt_labels = [], []
        for box, label in zip(geometry_transform(sample["boxes"], width, height, size),
                              sample["labels"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                skipped_zero_area += 1
                continue
            label = int(label)
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(label)
            annotation_id += 1
            coco_annotations.append({"id": annotation_id, "image_id": index, "category_id": label,
                                     "bbox": [x1, y1, x2 - x1, y2 - y1],
                                     "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            if label in counts:
                counts[label]["gt"] += 1
            total_gt += 1

        tensor = to_nchw(letterbox(image, size))
        boxes, labels, scores = session.run(["boxes", "labels", "scores"], {"images": tensor})
        boxes = np.asarray(boxes, dtype=np.float64)
        labels = np.asarray(labels).astype(np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != scores.shape \
                or labels.ndim != 1 or boxes.shape[0] != labels.shape[0]:
            raise SystemExit(f"unexpected output shapes at image {index}: "
                             f"{boxes.shape} {labels.shape} {scores.shape}")
        keep = scores >= SCORE_FLOOR
        keep &= boxes[:, 2] > boxes[:, 0]
        keep &= boxes[:, 3] > boxes[:, 1]
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        for box, label, score in zip(boxes, labels, scores):
            if not 1 <= label <= len(classes):
                continue
            coco_detections.append({"image_id": index, "category_id": int(label),
                                    "bbox": [float(box[0]), float(box[1]),
                                             float(box[2] - box[0]), float(box[3] - box[1])],
                                    "score": float(score)})
        total_det += int(scores.size)

        matched = greedy_counts(boxes.tolist(), labels.tolist(), scores.tolist(),
                                gt_boxes, gt_labels, len(classes))
        for label, values in matched.items():
            for key in ("tp", "fp"):
                counts[label][key] += values[key]
        if args.progress_every and (index + 1) % args.progress_every == 0:
            elapsed = time.time() - started
            print(json.dumps({"progress": {"images": index + 1, "of": len(samples),
                                           "seconds": round(elapsed, 1),
                                           "seconds_per_image": round(elapsed / (index + 1), 3)}}),
                  flush=True)

    elapsed = time.time() - started
    coco = coco_evaluate(coco_images, coco_annotations, coco_detections,
                         list(range(1, len(classes) + 1)))
    per_class = {}
    by_id = coco.get("per_class_by_id", {})
    for label, name in enumerate(classes, 1):
        entry = by_id.get(label, {})
        per_class[name] = {"map": entry.get("map"), "map50": entry.get("map50"),
                           "gt": counts[label]["gt"],
                           "evaluable": bool(counts[label]["gt"]) and entry.get("map") is not None}
    report = {
        "artifact": {"onnx": str(onnx_path), "onnx_bytes": onnx_path.stat().st_size,
                     "onnx_sha256": sha256_file(onnx_path),
                     "manifest": str(manifest_path),
                     "manifest_sha256": sha256_file(manifest_path),
                     "manifest_classes": len(classes)},
        "runtime": {"onnxruntime": ort_module.__version__,
                    "available_providers": ort_module.get_available_providers(),
                    "session_providers": session.get_providers(),
                    "input_shape": input_shape,
                    "output_names_types": [(v.name, v.type) for v in outputs]},
        "protocol": {"split": args.split, "images": len(samples), "classes": len(classes),
                     "score_floor": SCORE_FLOOR,
                     "preprocessing": "letterbox to size/max(w,h), centred, 114 padding, RGB, /255, "
                                      "NCHW float32; ImageNet normalisation is inside the graph",
                     "metric_implementation": "pycocotools COCOeval, bbox, default maxDets",
                     "operating_point": {"conf": OPERATING_CONF, "iou": OPERATING_IOU}},
        "coco": {"map": coco.get("map"), "map50": coco.get("map50"), "map75": coco.get("map75"),
                 "max_dets": coco.get("max_dets"), "available": coco.get("available", False)},
        "per_class": per_class,
        "operating_point": summarize_counts(counts, classes, len(samples)),
        "counts": {"ground_truth_boxes": total_gt, "detections_at_floor": total_det,
                   "skipped_zero_area_gt": skipped_zero_area,
                   "non_evaluable_classes": [name for name, v in per_class.items() if not v["evaluable"]]},
        "preprocessing_parity": parity,
        "timing": {"seconds": round(elapsed, 1), "images": len(samples),
                   "seconds_per_image": round(elapsed / len(samples), 3)},
    }
    if not math.isfinite(float(report["coco"]["map"] or 0.0)):
        raise SystemExit("non-finite mAP; refusing to write a report")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    print(json.dumps({"onnx_evaluation": {"map": report["coco"]["map"],
                                          "map50": report["coco"]["map50"],
                                          "images": len(samples),
                                          "seconds": report["timing"]["seconds"],
                                          "report": str(destination)}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
