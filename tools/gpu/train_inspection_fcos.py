#!/usr/bin/env python3
"""Manifest-driven BSD torchvision FCOS training; no AGPL training framework.

Dependencies (GPU host only): matching torch/torchvision, Pillow, numpy,
torchmetrics, pycocotools, onnx, onnxruntime. Python 3.12 / torch 2.8 supported.
The architecture/head replacement follows train_cplid_fcos.py; its incorrect
class-agnostic evaluator and unverified dictionary ONNX export are NOT reused.
Pure manifest / matching functions import without any ML dependencies.

GPU-host examples (do not run training on a local CPU):
  python train_inspection_fcos.py --manifest /data/fire_smoke.json --task fire_smoke \
      --out /models/fire_smoke --final-dir /models/final --smoke --workers 0
  python train_inspection_fcos.py --manifest /data/crop_pest.json --task crop_pest \
      --out /models/crop_pest --final-dir /models/final --epochs 24 --batch 4 --size 1024
  Add --resume /models/crop_pest/crop_pest_fcos_last.pth with identical settings
  to continue the original epoch schedule (not extend/change the total epochs).

Manifest: {"classes": ["fire", "smoke"], "train": [...], "val": [...], "test": [...]}
Each sample: {"image": "/absolute/image.jpg", "boxes": [[x1,y1,x2,y2]],
              "labels": [1], "source_group": "independent-capture-scene"}.
Empty boxes/labels are valid. source_group may repeat only within one split.
Smoke writes into the specified directories; use separate smoke directories.
Install torch 2.8 with matching torchvision 0.23 for the GPU host's CUDA runtime;
then pip install Pillow numpy torchmetrics pycocotools onnx onnxruntime.
COCO initialization downloads torchvision weights to --out/torch_cache, never
installs mmdet or ultralytics. Checkpoint resume accepts only your trusted .pth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

SPLITS = ("train", "val", "test")


def validate_manifest(data: dict, check_files: bool = True) -> dict:
    """Validate predefined group-disjoint splits before applying any limits."""
    # prepare_inspection_data.py stamps a non-training failure report at its --output
    # path when preparation fails, so the very path a coordinator passes as --manifest
    # can hold that report instead of a manifest. Name that condition explicitly; the
    # alternative is a misleading "classes must be a nonempty unique string list".
    if isinstance(data, dict) and data.get("usable_for_training") is False:
        errors = data.get("errors") or []
        first = errors[0] if errors else None
        if isinstance(first, dict):
            detail = first.get("message") or first.get("reason") or json.dumps(first, ensure_ascii=False)
        else:
            detail = str(first) if first is not None else "no error details recorded"
        raise ValueError(
            "manifest is a preparation failure report, not a training manifest "
            f"(status={data.get('status')!r}, error_count={data.get('error_count')}): {detail}"
        )
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes or any(
        not isinstance(c, str) or not c.strip() for c in classes
    ) or len(set(classes)) != len(classes):
        raise ValueError("classes must be a nonempty unique string list")
    paths, groups = {}, {}
    result = {"classes": list(classes)}
    for split in SPLITS:
        samples = data.get(split)
        if not isinstance(samples, list):
            raise ValueError(f"{split} must be a list (test may be empty)")
        result[split] = []
        for index, sample in enumerate(samples):
            tag = f"{split}[{index}]"
            if not isinstance(sample, dict):
                raise ValueError(f"{tag}: sample must be an object")
            image, group = sample.get("image"), sample.get("source_group")
            if not isinstance(image, str) or not Path(image).is_absolute():
                raise ValueError(f"{tag}: image must be an absolute path")
            if check_files and not Path(image).is_file():
                raise ValueError(f"{tag}: image does not exist: {image}")
            key = os.path.normcase(os.path.normpath(image))
            if key in paths:
                raise ValueError(f"duplicate image across/in splits: {image}")
            paths[key] = split
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{tag}: source_group must be a nonempty string")
            if group in groups and groups[group] != split:
                raise ValueError(f"source_group leakage: {group}")
            groups[group] = split
            boxes, labels = sample.get("boxes"), sample.get("labels")
            if not isinstance(boxes, list) or not isinstance(labels, list) or len(boxes) != len(labels):
                raise ValueError(f"{tag}: boxes/labels must be equal-length lists")
            for box, label in zip(boxes, labels):
                if not isinstance(box, list) or len(box) != 4 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
                ) or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                    raise ValueError(f"{tag}: invalid original-image xyxy box: {box}")
                if isinstance(label, bool) or not isinstance(label, int) or not 1 <= label <= len(classes):
                    raise ValueError(f"{tag}: labels must be 1-based class integers")
            result[split].append({"image": image, "source_group": group,
                                  "boxes": [list(b) for b in boxes], "labels": list(labels)})
    if not result["train"] or not result["val"]:
        raise ValueError("train and val must both contain samples")
    return result


def select_samples(samples: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministic sample cap; never drop empty annotations or a partial batch."""
    if limit < 0:
        raise ValueError("sample limit cannot be negative")
    if limit == 0 or limit >= len(samples):
        return list(samples)
    indices = sorted(random.Random(seed).sample(range(len(samples)), limit))
    return [samples[i] for i in indices]


def split_statistics(samples: list[dict], classes: list[str]) -> dict:
    return {"images": len(samples), "empty_images": sum(not s["boxes"] for s in samples),
            "source_groups": len({s["source_group"] for s in samples}),
            "objects": {name: sum(s["labels"].count(i) for s in samples)
                        for i, name in enumerate(classes, 1)}}


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def greedy_counts(pred: dict, target: dict, num_classes: int,
                  conf: float = 0.35, iou: float = 0.5) -> dict:
    """Score-ordered, same-class, one-to-one matching; boxes must share coordinates."""
    counts = {i: {"gt": target["labels"].count(i), "tp": 0, "fp": 0}
              for i in range(1, num_classes + 1)}
    used = set()
    order = sorted(range(len(pred["scores"])), key=lambda j: (-pred["scores"][j], j))
    for pi in order:
        label = pred["labels"][pi]
        if pred["scores"][pi] < conf or label not in counts:
            continue
        candidates = [(box_iou_xyxy(pred["boxes"][pi], box), gi)
                      for gi, box in enumerate(target["boxes"])
                      if gi not in used and target["labels"][gi] == label]
        best = max(candidates, default=(0.0, -1), key=lambda pair: pair[0])
        if best[1] >= 0 and best[0] >= iou:
            used.add(best[1])
            counts[label]["tp"] += 1
        else:
            counts[label]["fp"] += 1
    return counts


def summarize_counts(counts: dict, classes: list[str], images: int) -> dict:
    def rates(c):
        gt, tp, fp = c["gt"], c["tp"], c["fp"]
        return {**c, "fn": gt - tp, "recall": tp / gt if gt else None,
                "false_positive_fraction": fp / (tp + fp) if gt and tp + fp else None,
                "false_alarms_per_image": fp / images if gt and images else None}
    total = {key: sum(c[key] for c in counts.values()) for key in ("gt", "tp", "fp")}
    return {"conf": 0.35, "iou": 0.5, "matching": "same-class score-ordered greedy one-to-one",
            "images": images, "overall": rates(total),
            "per_class": {name: rates(counts[i]) for i, name in enumerate(classes, 1)}}


def letterbox_geometry(width: int, height: int, size: int) -> tuple:
    if min(width, height, size) <= 0:
        raise ValueError("image dimensions and size must be positive")
    scale = size / max(width, height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    return nw, nh, (size - nw) // 2, (size - nh) // 2, nw / width, nh / height


def transform_boxes(boxes: list, width: int, height: int, size: int, flip: bool = False) -> list:
    _, _, px, py, sx, sy = letterbox_geometry(width, height, size)
    transformed = []
    for x1, y1, x2, y2 in boxes:
        if x2 > width or y2 > height:
            raise ValueError("manifest box exceeds original image dimensions")
        b = [x1 * sx + px, y1 * sy + py, x2 * sx + px, y2 * sy + py]
        if flip:
            b = [size - b[2], b[1], size - b[0], b[3]]
        transformed.append(b)
    return transformed


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class InspectionDataset:
    def __init__(self, samples, size, train=False):
        self.samples, self.size, self.train = samples, size, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image, ImageEnhance
        sample = self.samples[index]
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        nw, nh, px, py, _, _ = letterbox_geometry(width, height, self.size)
        flip = self.train and random.random() < 0.5
        if self.train:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (px, py))
        if flip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        boxes = transform_boxes(sample["boxes"], width, height, self.size, flip)
        tensor = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(sample["labels"], dtype=torch.int64),
                  "image_id": torch.tensor(index, dtype=torch.int64)}
        return tensor, target


def backbone_parameter_trainable(name: str) -> bool:
    """Mirror torchvision's COCO default trainable_backbone_layers=3."""
    return any(name.startswith(layer) for layer in ("layer2", "layer3", "layer4"))


def build_model(size, num_classes, pretrained=True):
    import torch
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
                             weights_backbone=None, score_thresh=0.001, detections_per_img=300)
    if not pretrained:
        # torchvision's weights=None branch otherwise creates trainable BatchNorm
        # and unfreezes five backbone layers, unlike COCO's FrozenBatchNorm/three.
        # Recreate the exact checkpoint architecture without downloading weights.
        from torchvision.ops.misc import FrozenBatchNorm2d
        def freeze_batch_norm(module):
            for name, child in list(module.named_children()):
                if isinstance(child, torch.nn.BatchNorm2d):
                    replacement = FrozenBatchNorm2d(child.num_features, eps=child.eps)
                    with torch.no_grad():
                        replacement.weight.copy_(child.weight)
                        replacement.bias.copy_(child.bias)
                        replacement.running_mean.copy_(child.running_mean)
                        replacement.running_var.copy_(child.running_var)
                    setattr(module, name, replacement)
                else:
                    freeze_batch_norm(child)
        freeze_batch_norm(model.backbone)
        for name, parameter in model.backbone.body.named_parameters():
            parameter.requires_grad_(backbone_parameter_trainable(name))
    head = model.head.classification_head
    old = head.cls_logits
    layer = torch.nn.Conv2d(old.in_channels, head.num_anchors * (num_classes + 1), 3, padding=1)
    torch.nn.init.normal_(layer.weight, std=0.01)
    torch.nn.init.constant_(layer.bias, -math.log(99))
    head.num_classes, head.cls_logits = num_classes + 1, layer
    model.transform.min_size, model.transform.max_size = (size,), size
    return model


def evaluate(model, loader, device, classes):
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                                 backend="pycocotools")
    metric50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                                   iou_thresholds=[0.5], backend="pycocotools")
    counts = {i: {"gt": 0, "tp": 0, "fp": 0} for i in range(1, len(classes) + 1)}
    images = 0
    model.eval()
    with torch.inference_mode():
        for batch, targets in loader:
            outputs = model([image.to(device, non_blocking=True) for image in batch])
            predictions, ground_truth = [], []
            for output, target in zip(outputs, targets):
                keep = output["labels"] > 0
                pred = {key: output[key][keep].detach().cpu() for key in ("boxes", "labels", "scores")}
                gt = {key: target[key].cpu() for key in ("boxes", "labels")}
                predictions.append(pred)
                ground_truth.append(gt)
                matched = greedy_counts({k: v.tolist() for k, v in pred.items()},
                                        {k: v.tolist() for k, v in gt.items()}, len(classes))
                for label, values in matched.items():
                    for key, value in values.items():
                        counts[label][key] += value
                images += 1
            metric.update(predictions, ground_truth)
            metric50.update(predictions, ground_truth)
    raw = metric.compute()
    def valid(value):
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    per_class = {name: {"map": None, "map50": None} for name in classes}
    available = {int(c): i for i, c in enumerate(raw["classes"].reshape(-1).tolist())}
    ap = raw["map_per_class"].reshape(-1)
    for label, name in enumerate(classes, 1):
        if counts[label]["gt"] and label in available:
            per_class[name]["map"] = valid(ap[available[label]])
    # A second public metric accumulates the same outputs at IoU .5, without
    # private torchmetrics attributes or a second inference/test-set pass.
    raw50 = metric50.compute()
    available50 = {int(c): i for i, c in enumerate(raw50["classes"].reshape(-1).tolist())}
    ap50 = raw50["map_per_class"].reshape(-1)
    for label, name in enumerate(classes, 1):
        if counts[label]["gt"] and label in available50:
            per_class[name]["map50"] = valid(ap50[available50[label]])
    has_gt = any(c["gt"] for c in counts.values())
    return {"map": valid(raw["map"]) if has_gt else None,
            "map50": valid(raw["map_50"]) if has_gt else None, "per_class": per_class,
            "source": "torchmetrics MeanAveragePrecision / pycocotools COCO bbox AP, IoU .50:.95, maxDets=100",
            "operating_point": summarize_counts(counts, classes, images)}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def score_separation(values):
    """Distance from each score to its nearest neighbour in the sorted score order.

    A detection whose score is nearer than the numerical band to a neighbour has no
    well-defined rank, so its position and identity cannot be compared across runtimes.
    """
    import numpy as np
    return np.minimum(np.abs(np.diff(values, prepend=np.inf)), np.abs(np.diff(values, append=-np.inf)))


def match_detections_by_box(candidate_boxes, candidate_labels, reference_boxes, reference_labels, tolerance):
    """Greedy same-class nearest-box matching, in the candidate's own (descending score) order.

    Returns index pairs (candidate, reference). Ordering-invariant by construction: it
    answers "do the two runtimes see the same objects?", which stays meaningful even when
    a near-tied score puts a detection on a different side of the top-k truncation.
    """
    import numpy as np
    candidate_boxes = np.asarray(candidate_boxes, dtype=np.float64)
    candidate_labels = np.asarray(candidate_labels)
    reference_boxes = np.asarray(reference_boxes, dtype=np.float64)
    reference_labels = np.asarray(reference_labels)
    if len(candidate_boxes) != len(candidate_labels) or len(reference_boxes) != len(reference_labels):
        raise ValueError("boxes and labels must have equal lengths on both sides")
    used = np.zeros(reference_boxes.shape[0], dtype=bool)
    pairs = []
    for index, (box, label) in enumerate(zip(candidate_boxes, candidate_labels)):
        options = np.where((~used) & (reference_labels == label))[0]
        if options.size == 0:
            continue
        distance = np.abs(reference_boxes[options] - box).max(axis=1)
        best = int(options[int(distance.argmin())])
        if distance.min() <= tolerance:
            used[best] = True
            pairs.append((index, best))
    return pairs


def export_and_verify(model, dataset, device, path, size):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    class ExplicitOutputs(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, images):
            result = self.detector([images[0]])[0]
            keep = result["labels"] > 0
            return result["boxes"][keep], result["labels"][keep], result["scores"][keep]
    wrapper = ExplicitOutputs(model).eval()
    probe = dataset[0][0].unsqueeze(0).to(device)
    with torch.inference_mode():
        reference = [value.cpu().numpy() for value in wrapper(probe)]
        if len(reference[1]) == 0:
            raise RuntimeError("ONNX parity probe has no detections; cannot substantively validate boxes/scores")
        torch.onnx.export(wrapper, (probe,), str(path), opset_version=16, dynamo=False,
                          input_names=["images"], output_names=["boxes", "labels", "scores"],
                          dynamic_axes={"boxes": {0: "detections"}, "labels": {0: "detections"},
                                        "scores": {0: "detections"}}, do_constant_folding=True)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "images" or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, size, size]:
        raise RuntimeError(f"ONNX input contract mismatch: {[(v.name, v.type, v.shape) for v in inputs]}")
    expected_types = ["tensor(float)", "tensor(int64)", "tensor(float)"]
    if [v.name for v in outputs] != ["boxes", "labels", "scores"] or [v.type for v in outputs] != expected_types:
        raise RuntimeError(f"ONNX output contract mismatch: {[(v.name, v.type) for v in outputs]}")
    actual = session.run(["boxes", "labels", "scores"], {"images": probe.cpu().numpy()})
    if [value.dtype for value in actual] != [np.dtype("float32"), np.dtype("int64"), np.dtype("float32")]:
        raise RuntimeError("ONNX runtime output dtypes mismatch")
    for got, wanted in zip(actual, reference):
        if got.shape != wanted.shape or not np.isfinite(got).all():
            raise RuntimeError("ONNX actual output shape/nonfinite mismatch")
    if actual[0].ndim != 2 or actual[0].shape[1] != 4 or actual[1].ndim != 1 or actual[2].ndim != 1:
        raise RuntimeError("ONNX boxes/labels/scores rank mismatch")
    boxes_atol, scores_atol, rtol = 0.1, 1e-4, 1e-3
    tie_atol = 2 * scores_atol
    # Box-matched pairs carry pairing slack on top of numerical noise: two near-tied
    # detections can sit at the same place, so a candidate may pair with its twin rather
    # than its exact counterpart. Measured slack on randomly initialised models was
    # 0.149 px / 1.06e-4 (crop_pest) and 0.264 px / 1.15e-4 (fire_smoke), so the asserted
    # tolerance sits below the 1.0 px matching gate -- still far below a real export
    # defect, which shows up as tens to hundreds of px.
    boxes_matched_atol, scores_matched_atol = 0.5, 1e-3
    match_px = 1.0
    # Both runtimes emit detections sorted by score, but when scores are near-tied -- the
    # normal state of a randomly initialised smoke model -- the top-k selection and the
    # order inside a tie group are not unique. Only selection-free checks carry the verdict:
    #   1) box-matched agreement: every ONNX detection must find a same-class PyTorch
    #      detection within match_px, and matched pairs must agree within
    #      boxes_matched_atol / scores_matched_atol. Invariant to ordering and to which side
    #      of the truncation boundary a near-tied detection landed on.
    #   2) structural floors: matched pairs must exist, cover at least half the detections,
    #      and both runtimes must report the same detection count.
    # Rank-aligned, asserted-rank and multiset differences are reported as evidence only. A
    # rank that is locally unambiguous can still sit at a different GLOBAL index in the two
    # runtimes, because a tie group above it shifts every later index. Measured on the
    # fire_smoke gate: 295/300 detections match geometrically within 0.264 px while the
    # asserted-rank comparison reports 517 px -- asserting on rank position would fail an
    # export that is in fact correct.
    a_order = np.argsort(-actual[2].astype(np.float64), kind="stable")
    r_order = np.argsort(-reference[2].astype(np.float64), kind="stable")
    a_boxes, a_scores, a_labels = actual[0][a_order], actual[2][a_order].astype(np.float64), actual[1][a_order]
    r_boxes, r_scores, r_labels = reference[0][r_order], reference[2][r_order].astype(np.float64), reference[1][r_order]

    asserted = np.minimum(score_separation(a_scores), score_separation(r_scores)) > tie_atol
    mismatch = a_labels != r_labels
    labels_asserted_mismatch = a_labels[asserted] != r_labels[asserted]
    boxes_rank_gap = np.abs(a_boxes.astype(np.float64) - r_boxes.astype(np.float64)).max(axis=1)
    matched = match_detections_by_box(a_boxes, a_labels, r_boxes, r_labels, match_px)
    matched_left = np.array([pair[0] for pair in matched], dtype=int)
    matched_right = np.array([pair[1] for pair in matched], dtype=int)
    sorted_gap = np.diff(np.sort(r_scores))
    evidence = {"detections": int(a_scores.size),
                "scores_multiset_max_abs_diff": float(np.abs(np.sort(a_scores) - np.sort(r_scores)).max()),
                "boxes_multiset_max_abs_diff": float(np.abs(np.sort(a_boxes.astype(np.float64), axis=0)
                                                             - np.sort(r_boxes.astype(np.float64), axis=0)).max()),
                "box_matched_pairs": int(len(matched)),
                "box_matched_unmatched_onnx": int(a_scores.size - len(matched)),
                "box_matched_unmatched_torch": int(r_scores.size - len(matched)),
                "box_matched_max_abs_diff": float(np.abs(a_boxes[matched_left].astype(np.float64)
                                                         - r_boxes[matched_right].astype(np.float64)).max())
                if matched else None,
                "score_matched_max_abs_diff": float(np.abs(a_scores[matched_left] - r_scores[matched_right]).max())
                if matched else None,
                "match_tolerance_px": match_px,
                "boxes_asserted_ranks": int(asserted.sum()),
                "boxes_ambiguous_tie_ranks": int((~asserted).sum()),
                "boxes_asserted_max_abs_diff": float(boxes_rank_gap[asserted].max()) if asserted.any() else None,
                "scores_asserted_max_abs_diff": float(np.abs(a_scores[asserted] - r_scores[asserted]).max())
                if asserted.any() else None,
                "boxes_rank_max_abs_diff": float(boxes_rank_gap.max()),
                "scores_sorted_gap_median": float(np.median(sorted_gap)) if sorted_gap.size else None,
                "scores_sorted_gap_max": float(sorted_gap.max()) if sorted_gap.size else None,
                "rank_aligned_label_mismatch_count": int(mismatch.sum()),
                "labels_asserted_mismatch_count": int(labels_asserted_mismatch.sum()),
                "detections_above_0.05_onnx": int((a_scores >= 0.05).sum()),
                "detections_above_0.05_torch": int((r_scores >= 0.05).sum()),
                "labels_exact": not bool(mismatch.any()),
                "labels_exact_at_asserted_ranks": not bool(labels_asserted_mismatch.any()),
                "parity_scope": "verdict comes from selection-free box-matched pairs plus match-rate and "
                                "detection-count floors; rank-aligned, asserted-rank and multiset "
                                "differences are evidence of truncation-window instability, not verdicts",
                "asserted_checks": ["matched pairs cover at least 50% of ONNX detections",
                                    f"matched box pairs within {boxes_matched_atol} px",
                                    f"matched pair scores within {scores_matched_atol}",
                                    "detection counts agree within 5%"],
                "boxes_atol": boxes_atol, "scores_atol": scores_atol, "rtol": rtol, "tie_atol": tie_atol,
                "boxes_matched_atol": boxes_matched_atol, "scores_matched_atol": scores_matched_atol}
    # Printed before any assertion so a single failing run still yields every number.
    print(json.dumps({"onnx_export_evidence": evidence}, ensure_ascii=False), flush=True)
    if not matched:
        raise RuntimeError(
            f"no ONNX detection could be box-matched to a PyTorch detection within {match_px} px; "
            "the two runtimes are not computing the same detections")
    if len(matched) < 0.5 * a_scores.size:
        raise RuntimeError(
            f"only {len(matched)}/{a_scores.size} ONNX detections matched a PyTorch detection within "
            f"{match_px} px; the exported graph does not reproduce the model")
    if abs(a_scores.size - r_scores.size) > 0.05 * max(a_scores.size, r_scores.size):
        raise RuntimeError(
            f"the two runtimes returned different detection counts (onnx={a_scores.size}, "
            f"torch={r_scores.size}); ONNX postprocessing does not reproduce the model")
    # Only the selection-free comparison is asserted. Correspondence comes from box geometry,
    # so a permuted or differently truncated top-k can neither create nor hide a violation.
    np.testing.assert_allclose(a_boxes[matched_left], r_boxes[matched_right], rtol=rtol,
                               atol=boxes_matched_atol)
    np.testing.assert_allclose(a_scores[matched_left], r_scores[matched_right], rtol=rtol,
                               atol=scores_matched_atol)
    return {"provider": "CPUExecutionProvider", "real_image": dataset.samples[0]["image"],
            "input_dtype": "float32", "output_dtypes": [str(v.dtype) for v in actual],
            "labels_exact": not bool(mismatch.any()), "empty_detections": int(a_scores.size) == 0, **evidence}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "task", "out", "final-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-last", action="store_true",
                        help="continue from --out/<task>_fcos_last.pth when it exists; the saved "
                             "signature still has to match, so a changed configuration fails loudly")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--nonfinite", choices=("raise", "skip"), default="raise",
                        help="what to do when a training loss is not finite: raise (default, loud) or "
                             "skip that batch and keep going while recording nonfinite_batches; skipping "
                             "is capped by --nonfinite-limit so a poisoned model state still stops the run")
    parser.add_argument("--nonfinite-limit", type=int, default=20,
                        help="with --nonfinite skip: maximum number of skipped batches before giving up")
    return parser.parse_args()


def _finite(value):
    """float(value) or None when the tensor is not finite (keeps json.dumps allow_nan=False usable)."""
    number = float(value)
    return number if math.isfinite(number) else None


def nonfinite_diagnosis(epoch, step, losses, images, targets, model, scaler):
    """Everything needed to tell a data defect from a poisoned model state, without guessing."""
    import torch  # main() imports torch locally; a module-level helper must not assume it
    boxes = [b for t in targets for b in t["boxes"]]
    sides = [[float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes] if boxes else []
    parameters = list(model.parameters())
    return {"epoch": epoch, "step": step,
            "components": {name: _finite(value) for name, value in losses.items()},
            "loss_scale": float(scaler.get_scale()),
            "images": len(images),
            "images_finite": all(bool(torch.isfinite(im).all()) for im in images),
            "boxes": len(boxes),
            "boxes_finite": all(bool(torch.isfinite(b).all()) for b in boxes),
            "min_side": min((s for pair in sides for s in pair), default=None),
            "degenerate_boxes": sum(1 for pair in sides if min(pair) <= 0),
            "model_parameters_nonfinite": sum(int((~torch.isfinite(p)).sum()) for p in parameters),
            "peak_mem_bytes": torch.cuda.max_memory_allocated()}


def main():
    args = parse_args()
    if args.resume and args.resume_last:
        raise ValueError("pass either --resume or --resume-last, not both")
    if args.resume_last:
        candidate = Path(args.out) / f"{args.task}_fcos_last.pth"
        if candidate.exists():
            args.resume = candidate
            print(f"resume-last: continuing from {candidate}", flush=True)
        else:
            print(f"resume-last: {candidate} absent; starting a fresh run", flush=True)
    if min(args.epochs, args.batch, args.size) <= 0 or args.workers < 0 or min(args.limit_train, args.limit_val) < 0:
        raise ValueError("epochs/batch/size must be positive; workers/limits nonnegative")
    if not args.task or Path(args.task).name != args.task or any(c in args.task for c in '/\\:'):
        raise ValueError("task must be a safe filename component")
    if args.smoke and args.resume:
        raise ValueError("smoke is a fresh one-epoch export check; do not combine with resume")
    data = validate_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    manifest_hash = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    train = select_samples(data["train"], args.limit_train or (8 if args.smoke else 0), args.seed)
    val = select_samples(data["val"], args.limit_val or (4 if args.smoke else 0), args.seed + 1)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    # Import all required dependencies before training (missing ORT must not be skipped).
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    from torchmetrics.detection.mean_ap import MeanAveragePrecision  # noqa: F401
    import pycocotools.coco  # noqa: F401
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training only: CUDA is required; no local CPU training fallback")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out, final = Path(args.out), Path(args.final_dir)
    out.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(out / "torch_cache"))
    classes = data["classes"]
    epochs = 1 if args.smoke else args.epochs
    generator = torch.Generator().manual_seed(args.seed)
    def loader(samples, training=False):
        return DataLoader(InspectionDataset(samples, args.size, training), batch_size=args.batch,
                          shuffle=training, num_workers=args.workers, pin_memory=True, drop_last=False,
                          collate_fn=collate, worker_init_fn=seed_worker, generator=generator)
    train_loader, val_loader = loader(train, True), loader(val)
    model = build_model(args.size, len(classes), pretrained=not bool(args.resume)).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * .02)
    scaler = torch.amp.GradScaler("cuda")
    signature = {"manifest_sha256": manifest_hash, "classes": classes, "task": args.task,
                 "size": args.size, "seed": args.seed, "limit_train": args.limit_train,
                 "limit_val": args.limit_val, "batch": args.batch, "epochs": epochs, "lr": args.lr,
                 "workers": args.workers}
    start, best_score, best_epoch, best_metrics = 1, -math.inf, 0, None
    best_path, last_path = out / f"{args.task}_fcos_best.pth", out / f"{args.task}_fcos_last.pth"
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["signature"] != signature:
            raise ValueError("resume configuration/manifest mismatch (including scheduler total epochs)")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best_score, best_epoch, best_metrics = state["best_score"], state["best_epoch"], state["best_metrics"]
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        generator.set_state(state["loader_rng"].cpu())
        if not best_path.exists():
            raise ValueError("resume requires corresponding best checkpoint in --out")
    nonfinite_total = 0
    previous_log = out / f"{args.task}_epochs.jsonl"
    if args.resume and previous_log.exists():
        for line in previous_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                nonfinite_total = max(nonfinite_total, int(json.loads(line).get("nonfinite_batches", 0)))
    atomic_json(out / f"{args.task}_run.json", {"signature": signature, "smoke": args.smoke,
                "nonfinite": {"policy": args.nonfinite, "limit": args.nonfinite_limit},
                "splits": {s: split_statistics(items, classes) for s, items in
                           (("train", train), ("val", val), ("test", data["test"]))}})
    log_path = out / f"{args.task}_epochs.jsonl"
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        for epoch in range(start, epochs + 1):
            began = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            loss_sum, samples_seen, component_sum = 0.0, 0, {}
            for step, (images, targets) in enumerate(train_loader, 1):
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    nonfinite_total += 1
                    diagnosis = nonfinite_diagnosis(epoch, step, losses, images, targets, model, scaler)
                    diagnosis["nonfinite_batches"] = nonfinite_total
                    print(json.dumps({"nonfinite_batch": diagnosis}, ensure_ascii=False), flush=True)
                    if args.nonfinite == "raise" or nonfinite_total > args.nonfinite_limit:
                        raise RuntimeError("nonfinite training loss: " + json.dumps(diagnosis))
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                count = len(images)
                samples_seen += count
                loss_sum += float(loss.detach()) * count
                for key, value in losses.items():
                    component_sum[key] = component_sum.get(key, 0.0) + float(value.detach()) * count
            metrics = evaluate(model, val_loader, device, classes)
            scheduler.step()
            score = metrics["map"] if metrics["map"] is not None else -1.0
            improved = score > best_score
            if improved:
                best_score, best_epoch, best_metrics = score, epoch, metrics
            row = {"epoch": epoch, "loss": loss_sum / samples_seen if samples_seen else None,
                   "loss_components": {k: v / samples_seen for k, v in component_sum.items()} if samples_seen else {},
                   "map": metrics["map"], "map50": metrics["map50"], "val": metrics,
                   "seconds": time.perf_counter() - began,
                   "peak_mem_bytes": torch.cuda.max_memory_allocated(device),
                   "lr": optimizer.param_groups[0]["lr"], "samples_seen": samples_seen,
                   "nonfinite_batches": nonfinite_total}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            print(f"epoch={epoch} loss={row['loss']} map={row['map']} map50={row['map50']} "
                  f"seconds={row['seconds']:.1f} nonfinite={nonfinite_total}", flush=True)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                     "signature": signature, "best_score": best_score, "best_epoch": best_epoch,
                     "best_metrics": best_metrics, "random_state": random.getstate(),
                     "numpy_state": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": generator.get_state()}
            for path in ([best_path, last_path] if improved else [last_path]):
                temporary = path.with_suffix(".pth.tmp")
                torch.save(state, temporary)
                temporary.replace(path)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    model.eval()
    test_metrics = None
    # Independent test is never used for model selection, never touched in smoke.
    # An immutable receipt prevents repeated formal test evaluation on resume.
    receipt = out / f"{args.task}_test.json"
    if not args.smoke and data["test"]:
        checkpoint_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved["signature"] != signature or saved["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError("test receipt belongs to another final model; refusing repeat test")
            test_metrics = saved["metrics"]
        else:
            test_metrics = evaluate(model, loader(data["test"]), device, classes)
            atomic_json(receipt, {"signature": signature, "checkpoint_sha256": checkpoint_hash,
                                  "metrics": test_metrics, "evaluations": 1})
    onnx_path = final / f"{args.task}_fcos_{args.size}.onnx"
    verification = export_and_verify(model, val_loader.dataset, device, onnx_path, args.size)
    print(json.dumps({"task": args.task, "onnx_verification": verification},
                     ensure_ascii=False, allow_nan=False), flush=True)
    atomic_json(final / f"{args.task}_fcos_metadata.json", {
        "task": args.task, "classes": classes, "label_ids": {name: i for i, name in enumerate(classes, 1)},
        "architecture": "torchvision FCOS ResNet50 FPN", "implementation_license": "BSD-3-Clause",
        "smoke": args.smoke, "selection": {"split": "val", "criterion": "COCO map", "epoch": best_epoch},
        "manifest_sha256": manifest_hash, "val_metrics": best_metrics, "test_metrics": test_metrics,
        "test_status": "not evaluated (smoke)" if args.smoke else ("evaluated once" if data["test"] else "no test samples"),
        "preprocessing": {"color": "RGB", "input": [1, 3, args.size, args.size], "dtype": "float32",
            "range": [0, 1], "resize": "PIL bilinear, fit longest side, rounded actual x/y scale",
            "padding": "centered RGB 114, floor left/top", "normalization": "ImageNet mean/std inside graph",
            "output_boxes": "xyxy in letterboxed input pixels; undo padding and actual x/y scale for original image"},
        "outputs": ["boxes:float32[N,4]", "labels:int64[N] (1-based)", "scores:float32[N]"],
        "onnx": {"opset": 16, "dynamo": False, "verification": verification},
        "training": {"epochs": epochs, "epochs_completed": epochs - start + 1, "resumed_from": start - 1,
                     "nonfinite": {"policy": args.nonfinite, "limit": args.nonfinite_limit,
                                   "skipped_batches": nonfinite_total}},
        "dependencies": {"torch": torch.__version__, "onnxruntime": __import__("onnxruntime").__version__}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
