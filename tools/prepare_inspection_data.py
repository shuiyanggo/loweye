#!/usr/bin/env python3
"""Prepare CPU-only D-Fire/IP102 detection manifests; never download or train.

Layout/conversion conventions follow the open-source YOLO dataset format:
https://docs.ultralytics.com/datasets/detect/ (no Ultralytics dependency).
IP102's official VOC-style detection mirror is also supported.
Requires Pillow and PyYAML. Strict by default; explicit quarantine audits exclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "trainval": "train", "training": "train",
                 "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
REFERENCES = {
    "dfire": {
        "source": "https://github.com/gaia-solutions-on-demand/DFireDataset",
        "license_reference": "https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/LICENSE",
        "terms": "Collection CC0-1.0; upstream disclaims original image ownership. Verify original-image rights.",
        "approximate_download": "~21k images / ~3GB; mirror-dependent, not a measured input count",
    },
    "ip102": {
        "source": "https://github.com/xpwu95/IP102",
        "license_reference": "https://github.com/xpwu95/IP102/blob/master/README.md",
        "classes_reference": "https://github.com/xpwu95/IP102/blob/master/classes.txt",
        "terms": "Free for academic usage; contact Xiaoping Wu for other purposes. Mirror PDDL does not supersede original terms.",
        "approximate_download": "~19k detection images / ~0.8GB; mirror-dependent, not a measured input count",
    },
}


class DataValidationError(ValueError):
    def __init__(self, errors, quarantine=None):
        self.errors = errors
        self.quarantine = quarantine
        super().__init__("Dataset validation failed: " + "; ".join(errors[:5]))


def convert_yolo_labels(text: str, width: int, height: int, num_classes: int):
    """Pure strict normalized YOLO -> pixel xyxy, foreground labels start at 1."""
    if width <= 0 or height <= 0 or num_classes <= 0:
        raise ValueError("Invalid image dimensions or class count")
    boxes, labels = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue  # An existing empty file is a valid negative sample.
        tokens = line.split()
        if len(tokens) != 5 or not re.fullmatch(r"\d+", tokens[0]):
            raise ValueError(f"line {number}: expected integer 0-based class and four YOLO coordinates")
        label = int(tokens[0])
        if not 0 <= label < num_classes:
            raise ValueError(f"line {number}: class {label} outside 0..{num_classes - 1}")
        try:
            x, y, w, h = map(float, tokens[1:])
        except ValueError as exc:
            raise ValueError(f"line {number}: invalid coordinates") from exc
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"line {number}: nonfinite/out-of-range coordinates or empty box")
        coords = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if any(v < -1e-9 or v > 1 + 1e-9 for v in coords):
            raise ValueError(f"line {number}: box extends beyond image")
        boxes.append([max(0., coords[0]) * width, max(0., coords[1]) * height,
                      min(1., coords[2]) * width, min(1., coords[3]) * height])
        labels.append(label + 1)
    return boxes, labels


def convert_voc_labels(text: str, width: int, height: int, classes: list[str], *, class_id_base: int = 1):
    """VOC numeric class IDs use an explicit base; boxes stay 1-based inclusive."""
    if class_id_base not in (0, 1):
        raise ValueError("VOC class ID base must be 0 or 1")
    xml = ET.fromstring(text)
    size = xml.find("size")
    if size is not None:
        if int(size.findtext("width", "0")) != width or int(size.findtext("height", "0")) != height:
            raise ValueError("VOC dimensions disagree with decoded image")
    boxes, labels = [], []
    for obj in xml.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if re.fullmatch(r"\d+", name):
            source_label = int(name)
            if not class_id_base <= source_label < class_id_base + len(classes):
                raise ValueError(f"VOC class {name!r} outside {class_id_base}..{class_id_base + len(classes) - 1}")
            label = source_label - class_id_base + 1
        elif name in classes:
            label = classes.index(name) + 1
        else:
            raise ValueError(f"Unknown VOC class {name!r}")
        bounds = obj.find("bndbox")
        if bounds is None:
            raise ValueError("Missing VOC bndbox")
        coords = [float(bounds.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = coords
        if not all(math.isfinite(v) for v in coords) or not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
            raise ValueError("Invalid VOC 1-based inclusive box")
        boxes.append([x1 - 1, y1 - 1, x2, y2])
        labels.append(label)
    return boxes, labels


def infer_source_group(image: Path, root: Path, digest: str) -> str:
    """Only explicit frame markers are heuristic video evidence, not plain digits."""
    match = re.fullmatch(r"(.+?)[_-](?:frame|frm)[_-]?\d+", image.stem, flags=re.IGNORECASE)
    if match:
        # Same video basename across split directories may indicate a shared source.
        return "filename-video:" + match.group(1).casefold()
    return "sha256:" + digest


def deduplicate_splits(splits: dict):
    """Pure dedup, byte-identical SHA256 priority test > val > train (also within split)."""
    kept = {name: [] for name in SPLITS}
    seen, audit = {}, []
    for split in reversed(SPLITS):
        for item in sorted(splits.get(split, []), key=lambda x: x["image"]):
            digest = item["sha256"]
            if digest in seen:
                winner_split, winner = seen[digest]
                audit.append({"sha256": digest, "removed_split": split, "removed_image": item["image"],
                              "kept_split": winner_split, "kept_image": winner["image"],
                              "annotation_conflict": (item["boxes"], item["labels"]) != (winner["boxes"], winner["labels"])})
            else:
                seen[digest] = (split, item)
                kept[split].append(dict(item))
    return kept, audit


def grouped_partition(items: list, fractions: tuple, seed: int = 42):
    """Pure seeded partition of hash/source groups; ratios approximate for large groups."""
    groups = defaultdict(list)
    for item in items:
        groups[item["source_group"]].append(item)
    for values in groups.values():
        values.sort(key=lambda item: item["image"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < len(fractions):
        raise ValueError("Too few independent hash/source groups for nonempty splits")
    total = len(items)
    result = [[] for _ in fractions]
    # Reserve at least one group per split; greedily fill each requested fraction.
    remaining = keys[:]
    for idx, fraction in enumerate(fractions[:-1]):
        target = max(1, round(total * fraction))
        while remaining and (len(result[idx]) < target or not result[idx]):
            if len(remaining) <= len(fractions) - idx - 1:
                break
            result[idx].extend(groups[remaining.pop(0)])
    for key in remaining:
        result[-1].extend(groups[key])
    if any(not split for split in result):
        raise ValueError("Unable to create nonempty grouped splits")
    return result


def _names_from_file(path: Path):
    rows = [row.strip() for row in path.read_text(encoding="utf-8-sig").splitlines() if row.strip()]
    numbered = [re.fullmatch(r"(\d+)\s+(.+)", row) for row in rows]
    if any(numbered):
        if not all(numbered):
            raise ValueError("classes file mixes indexed and unindexed names")
        indices = [int(m.group(1)) for m in numbered]
        if indices not in (list(range(len(rows))), list(range(1, len(rows) + 1))):
            raise ValueError("classes file indices must be contiguous 0-based or official 1-based")
        return [m.group(2).strip() for m in numbered]
    return rows


def resolve_classes(root: Path, dataset: str, explicit=None, classes_file=None):
    import yaml
    paths = sorted(root.rglob("data.yaml")) + sorted(root.rglob("data.yml"))
    yaml_records = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or "names" not in data:
            raise ValueError(f"{path}: YAML must contain names")
        names = data["names"]
        if isinstance(names, dict):
            try:
                indexed = {int(k): v for k, v in names.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid names indices") from exc
            if len(indexed) != len(names) or sorted(indexed) != list(range(len(indexed))):
                raise ValueError(f"{path}: names indices must start at zero and be contiguous")
            names = [indexed[i] for i in range(len(indexed))]
        if not isinstance(names, list):
            raise ValueError(f"{path}: names must be a string list or 0-based mapping")
        if "nc" in data and data["nc"] != len(names):
            raise ValueError(f"{path}: nc disagrees with names")
        yaml_records.append((path, data, names))
    if yaml_records:
        names = yaml_records[0][2]
        if any(record[2] != names for record in yaml_records):
            raise ValueError("Conflicting data.yaml class order; select a single dataset root")
        source = {"kind": "data.yaml", "paths": [str(p.resolve()) for p, _, _ in yaml_records]}
        if explicit is not None and explicit != names:
            raise ValueError("--classes conflicts with YAML order")
        if classes_file is not None and _names_from_file(Path(classes_file)) != names:
            raise ValueError("--classes-file conflicts with YAML order")
    elif dataset == "dfire":
        if explicit is None or len(explicit) != 2 or set(explicit) != {"smoke", "fire"}:
            raise ValueError("D-Fire without YAML requires explicit --classes smoke fire (or verified reversed order)")
        names, source = list(explicit), {"kind": "explicit-cli", "order_verified_by": "caller"}
    else:
        if explicit is not None:
            raise ValueError("IP102 requires official classes.txt or --classes-file, not guessed --classes")
        candidates = [Path(classes_file)] if classes_file else sorted(root.rglob("classes.txt"))
        if len(candidates) != 1:
            raise ValueError("IP102 without YAML requires one official classes.txt or explicit --classes-file")
        names = _names_from_file(candidates[0])
        source = {"kind": "classes-file", "path": str(candidates[0].resolve()),
                  "provenance": "caller must verify official class order"}
    expected = 2 if dataset == "dfire" else 102
    if len(names) != expected or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{dataset} requires {expected} unique nonempty string classes")
    if dataset == "dfire" and {n.casefold() for n in names} != {"smoke", "fire"}:
        raise ValueError("D-Fire YAML names must identify smoke and fire; do not guess numeric names")
    return names, source, yaml_records


def _annotation_path(image: Path, root: Path, dataset: str):
    candidates = []
    parts = image.relative_to(root).parts
    for index, part in enumerate(parts[:-1]):
        if part.casefold() in {"images", "jpegimages"}:
            candidates.append(root.joinpath(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt"))
            # Accommodate Windows/Unix mirrors that capitalize the directory.
            for sibling in image.parents[len(parts) - index - 2].parent.iterdir():
                if sibling.is_dir() and sibling.name.casefold() in ({"labels", "annotations"} if dataset == "ip102" else {"labels"}):
                    extensions = (".xml", ".txt") if sibling.name.casefold() == "annotations" else (".txt",)
                    for extension in extensions:
                        candidates.append(sibling.joinpath(*parts[index + 1:]).with_suffix(extension))
    existing = sorted({p.resolve() for p in candidates if p.is_file()}, key=str)
    if len(existing) != 1:
        exc = ValueError(f"Missing or ambiguous annotation for {image}: found {len(existing)}")
        exc.annotation_candidates = [str(path) for path in existing]
        raise exc
    return existing[0]


def _official_memberships(root: Path, yaml_records, images: list):
    membership = {}
    def assign(image, split):
        image = image.resolve()
        if image in membership and membership[image] != split:
            raise ValueError(f"Image listed in multiple official splits: {image}")
        membership[image] = split
    # Honor YAML split paths or YOLO image-list txt files, not just directory names.
    for yaml_path, data, _ in yaml_records:
        base = yaml_path.parent
        if data.get("path"):
            candidate = Path(str(data["path"]))
            candidate = candidate if candidate.is_absolute() else base / candidate
            if candidate.is_dir():
                base = candidate
            # Mirrors often retain an obsolete machine-specific 'path'.
        for split in SPLITS:
            entries = data.get(split)
            if entries is None:
                continue
            for entry in (entries if isinstance(entries, list) else [entries]):
                target = Path(str(entry))
                target = target if target.is_absolute() else base / target
                if not target.exists():
                    raise ValueError(f"YAML split path does not exist: {target}")
                if target.is_dir():
                    for image in images:
                        if image.is_relative_to(target.resolve()):
                            assign(image, split)
                else:
                    for line in target.read_text(encoding="utf-8-sig").splitlines():
                        if not line.strip():
                            continue
                        p = Path(line.strip())
                        if not p.is_absolute():
                            choices = [target.parent / p, base / p, root / p]
                            found = {choice.resolve() for choice in choices if choice.is_file()}
                            if len(found) != 1:
                                raise ValueError(f"Unresolved/ambiguous image-list entry: {line}")
                            p = found.pop()
                        if p.resolve() not in images:
                            raise ValueError(f"Official image-list entry missing from image tree: {p}")
                        assign(p, split)
    # VOC detection split lists: trainval is a training pool, never merge test.
    by_stem = defaultdict(list)
    for image in images:
        by_stem[image.stem].append(image)
    split_directories = {p for p in root.rglob("Main") if p.parent.name.casefold() == "imagesets"}
    # Official detection mirrors also place trainval.txt/test.txt next to JPEGImages.
    # Only image-container parents qualify, avoiding unrelated text files.
    for image in images:
        for parent in image.parents:
            if parent.name.casefold() in {"images", "jpegimages"}:
                candidate = parent.parent
                if (candidate / "trainval.txt").is_file() and (candidate / "test.txt").is_file():
                    split_directories.add(candidate)
                break
    for directory in sorted(split_directories):
        filenames = ("train.txt", "val.txt", "test.txt") if (directory / "train.txt").exists() else ("trainval.txt", "test.txt")
        for filename in filenames:
            path = directory / filename
            if not path.is_file():
                continue
            split = SPLIT_ALIASES[path.stem]
            for row in path.read_text(encoding="utf-8-sig").splitlines():
                if not row.strip():
                    continue
                stem = Path(row.split()[0]).stem
                if len(by_stem[stem]) != 1:
                    raise ValueError(f"VOC split image missing/ambiguous: {stem}")
                assign(by_stem[stem][0], split)
    for image in images:
        if image in membership:
            continue
        split_names = {SPLIT_ALIASES[p.casefold()] for p in image.relative_to(root).parts[:-1] if p.casefold() in SPLIT_ALIASES}
        if len(split_names) > 1:
            raise ValueError(f"Ambiguous split directories: {image}")
        membership[image] = next(iter(split_names), "unsplit")
    return membership


def build_manifest(root, dataset: str, classes=None, classes_file=None, *, voc_class_id_base: int = 1,
                   quarantine_invalid: bool = False, allow_unobserved_classes: bool = False):
    from PIL import Image
    root = Path(root).resolve()
    if voc_class_id_base not in (0, 1):
        raise DataValidationError(["VOC class ID base must be 0 or 1"])
    if allow_unobserved_classes and dataset != "ip102":
        raise DataValidationError(["Allowing unobserved classes is only supported for IP102"])
    if not root.is_dir() or dataset not in REFERENCES:
        raise DataValidationError(["Root must exist and dataset must be dfire or ip102"])
    try:
        names, class_source, yaml_records = resolve_classes(root, dataset, classes, classes_file)
        images = sorted((p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
                         and any(part.casefold() in {"images", "jpegimages"} for part in p.relative_to(root).parts[:-1])), key=str)
        if not images:
            raise ValueError("No images found beneath an images/Images/JPEGImages directory")
        membership = _official_memberships(root, yaml_records, images)
    except (ValueError, OSError) as exc:
        raise DataValidationError([str(exc)]) from exc
    pools = {name: [] for name in (*SPLITS, "unsplit")}
    errors, formats = [], Counter()
    quarantine = {"enabled": quarantine_invalid, "count": 0, "records": [],
                  "by_original_split": {}, "by_reason": {}}
    for image in images:
        annotation = None
        try:
            annotation = _annotation_path(image, root, dataset)
            with Image.open(image) as decoded:
                decoded.verify()
            with Image.open(image) as decoded:
                decoded.load()
                width, height = decoded.size
            sha = hashlib.sha256()
            with image.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    sha.update(chunk)
            digest = sha.hexdigest()
            text = annotation.read_text(encoding="utf-8-sig")
            if annotation.suffix == ".xml":
                boxes, labels = convert_voc_labels(text, width, height, names, class_id_base=voc_class_id_base)
                formats[f"VOC_{voc_class_id_base}based_inclusive"] += 1
            else:
                if annotation.parent.name.casefold() == "annotations" and text.lstrip().startswith("<"):
                    raise ValueError(f"Missing or ambiguous annotation for {image}: XML content in YOLO .txt source")
                boxes, labels = convert_yolo_labels(text, width, height, len(names))
                formats["YOLO_0based_normalized"] += 1
            pools[membership[image]].append({"image": str(image), "width": width, "height": height,
                "boxes": boxes, "labels": labels, "source_group": infer_source_group(image, root, digest),
                "sha256": digest, "annotation": str(annotation), "original_split": membership[image]})
        except (ValueError, OSError, ET.ParseError, Image.DecompressionBombError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if quarantine_invalid:
                quarantine["records"].append({"image": str(image), "reason": reason,
                                              "original_split": membership[image],
                                              "annotation_candidates": getattr(exc, "annotation_candidates",
                                                                                 [str(annotation)] if annotation else [])})
            else:
                errors.append(f"{image}: {reason}")
    quarantine["count"] = len(quarantine["records"])
    quarantine["by_original_split"] = dict(Counter(item["original_split"] for item in quarantine["records"]))
    quarantine["by_reason"] = dict(Counter(item["reason"] for item in quarantine["records"]))
    if errors:
        raise DataValidationError(errors, quarantine)
    # Quarantine excludes invalid sources from the manifest; never moves or changes input files.
    official = {split: pools[split] for split in SPLITS}
    if pools["unsplit"] and any(official.values()):
        raise DataValidationError(["Mixed split and unsplit images; choose a complete dataset root"], quarantine)
    try:
        if pools["unsplit"]:
            official["train"] = pools["unsplit"]
            official, audit = deduplicate_splits(official)
            partitions = grouped_partition(official["train"], (.70, .15, .15))
            final = dict(zip(SPLITS, partitions))
            strategy = "generated seed42 grouped 70/15/15; no official split found"
        else:
            final, audit = deduplicate_splits(official)
            if final["train"] and final["test"] and not final["val"] and not pools["val"]:
                final["train"], final["val"] = grouped_partition(final["train"], (.85, .15))
                strategy = "official test preserved; seed42 grouped 15% validation from official training pool"
            else:
                strategy = "official split preserved"
        if any(not final[s] for s in SPLITS):
            raise ValueError("All train/val/test splits must contain samples after deduplication")
        observed = {label for split in final.values() for item in split for label in item["labels"]}
        missing = sorted(set(range(1, len(names) + 1)) - observed)
        if missing and not allow_unobserved_classes:
            raise ValueError(f"Dataset has too few observed classes; missing foreground labels: {missing}")
    except ValueError as exc:
        raise DataValidationError([str(exc)], quarantine) from exc
    group_splits = defaultdict(set)
    for split, items in final.items():
        for item in items:
            group_splits[item["source_group"]].add(split)
    overlap = [{"source_group": key, "splits": sorted(value)} for key, value in sorted(group_splits.items()) if len(value) > 1]
    statistics = {}
    for split, items in final.items():
        counts = Counter(label for item in items for label in item["labels"])
        statistics[split] = {"images": len(items), "empty_images": sum(not i["labels"] for i in items),
            "source_groups": len({i["source_group"] for i in items}),
            "objects": {name: counts[index + 1] for index, name in enumerate(names)}}
    class_coverage = {
        "policy": "explicit_allow_unobserved_ip102" if allow_unobserved_classes else "strict_all_classes_observed",
        "allow_unobserved_classes": allow_unobserved_classes,
        "scope": "valid retained samples after quarantine and deduplication; not a raw-source audit",
        "foreground_id_base": 1, "source_class_count": len(names),
        "observed_ids": sorted(observed), "observed_names": [names[label - 1] for label in sorted(observed)],
        "missing_ids": missing, "missing_names": [names[label - 1] for label in missing],
        "all_source_classes_evaluable": not missing,
        "missing_classes": [{"id": label, "name": names[label - 1], "evaluable": False,
                             "reason": "No ground-truth objects in retained dataset; this class cannot be learned or evaluated from this dataset."}
                            for label in missing],
        "limitation": "Original class order and IDs are unchanged. Observed labels do not guarantee per-split training/evaluation coverage; missing classes have no demonstrated recognition capability."
    }
    local_records = sorted(root.rglob("source_license.json"))
    return {"schema_version": 1, "status": "ok", "dataset": dataset, "root": str(root), "classes": names,
        **final, "class_source": class_source, "annotation_formats": dict(formats), "statistics": statistics,
        "voc_class_id_base": voc_class_id_base, "quarantine": quarantine, "class_coverage": class_coverage,
        "split_strategy": strategy, "seed": 42,
        "dedup": {"algorithm": "sha256 original image bytes", "priority": ["test", "val", "train"],
                  "removed_count": len(audit), "records": audit},
        "source_group_audit": {"cross_split_groups": overlap},
        "license": {**REFERENCES[dataset], "local_records": [str(p.resolve()) for p in local_records],
                    "policy": "Academic local preparation only; no raw-image redistribution; not a legal clearance."},
        "limitations": ["Byte hashing catches exact files, not re-encoded/cropped duplicates.",
            "Only explicit _frameNN/_frmNN filename markers are heuristic video groups; other files use hashes. Filename groups are not verified provenance.",
            "Official splits are preserved, not a guarantee of video/source isolation; inspect cross_split_groups before training.",
            "Generated ratios are approximate at whole hash/source-group granularity, without class stratification.",
            "Local mirrors and classes-file provenance require verification; no aerial-domain validation is established."]}


def make_group_safe_manifest(manifest: dict):
    """Make an explicitly nonofficial seeded split for identifiable groups only."""
    import copy
    result = copy.deepcopy(manifest)
    items = [item for split in SPLITS for item in manifest[split]]
    assigned_before = {item["image"]: split for split in SPLITS for item in manifest[split]}
    partitions = grouped_partition(items, (.70, .15, .15))
    changes = []
    for split, samples in zip(SPLITS, partitions):
        result[split] = samples
        for item in samples:
            previous = assigned_before[item["image"]]
            if previous != split:
                changes.append({"image": item["image"], "previous_split": previous, "new_split": split})
    result["split_strategy"] = "NONOFFICIAL group-safe seed42 70/15/15 from all original splits"
    result["official_split_changes"] = changes
    result["source_group_audit"] = {"cross_split_groups": []}
    for split in SPLITS:
        samples = result[split]
        counts = Counter(label for item in samples for label in item["labels"])
        result["statistics"][split] = {"images": len(samples), "empty_images": sum(not i["labels"] for i in samples),
            "source_groups": len({i["source_group"] for i in samples}),
            "objects": {name: counts[index + 1] for index, name in enumerate(result["classes"])}}
    result["limitations"].append("group-safe applies only to inferred groups; unknown video provenance cannot be isolated or guaranteed.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("dfire", "ip102"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--classes-file", type=Path)
    parser.add_argument("--voc-class-id-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--quarantine-invalid", action="store_true",
                        help="Exclude invalid/missing source samples and record every exclusion; default is strict")
    parser.add_argument("--allow-unobserved-classes", action="store_true",
                        help="IP102 only: preserve source class IDs and disclose classes with no retained ground truth")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    # A JSON output must never replace an input image, annotation or YAML.
    if output.suffix.casefold() != ".json" or output.name == "source_license.json":
        parser.error("--output must be a separate manifest .json, not source_license.json")
    try:
        manifest = build_manifest(args.root, args.dataset, args.classes, args.classes_file,
                                  voc_class_id_base=args.voc_class_id_base,
                                  quarantine_invalid=args.quarantine_invalid,
                                  allow_unobserved_classes=args.allow_unobserved_classes)
        code = 0
    except DataValidationError as exc:
        manifest = {"status": "error", "dataset": args.dataset, "root": str(args.root.resolve()),
                    "errors": exc.errors, "error_count": len(exc.errors), "usable_for_training": False}
        if exc.quarantine is not None:
            manifest["quarantine"] = exc.quarantine
        print(str(exc), file=sys.stderr)
        code = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: {output}")
    if code == 0 and manifest["source_group_audit"]["cross_split_groups"]:
        safe_output = output.with_name(output.stem + "_group_safe_manifest.json")
        try:
            safe = make_group_safe_manifest(manifest)
        except ValueError as exc:
            print(f"Official manifest retained, but group-safe training split cannot be made: {exc}", file=sys.stderr)
            return 2
        safe_output.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"NONOFFICIAL inferred-group-safe manifest: {safe_output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
