"""Read-only IP102 source class audit; no preparation, mutation, or training."""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def audit(root):
    result = {}
    combined = Counter()
    for suffix in ("xml", "txt"):
        paths = sorted(root.glob(f"*.{suffix}"))
        counts = Counter()
        failures = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8-sig")
                if suffix == "xml":
                    names = [(obj.findtext("name") or "").strip()
                             for obj in ET.fromstring(text).findall("object")]
                else:
                    names = []
                    for number, line in enumerate(text.splitlines(), 1):
                        if not line.strip():
                            continue
                        tokens = line.split()
                        if len(tokens) != 5:
                            raise ValueError(f"line {number}: expected five YOLO fields")
                        coords = [float(value) for value in tokens[1:]]
                        if not all(math.isfinite(value) for value in coords):
                            raise ValueError(f"line {number}: nonfinite YOLO coordinates")
                        names.append(tokens[0])
                labels = []
                for name in names:
                    if not re.fullmatch(r"\d+", name) or not 0 <= int(name) <= 101:
                        raise ValueError(f"source class {name!r} outside integer 0..101")
                    labels.append(int(name))
                counts.update(labels)
            except (ValueError, OSError, ET.ParseError) as exc:
                failures.append({"annotation": str(path), "reason": f"{type(exc).__name__}: {exc}"})
        combined.update(counts)
        result[suffix] = {"files": len(paths), "valid_files": len(paths) - len(failures),
                          "counts": {str(i): counts[i] for i in range(102)},
                          "missing0based": [i for i in range(102) if not counts[i]],
                          "failures": failures}
    result["combined"] = {"counts": {str(i): combined[i] for i in range(102)},
                          "missing0based": [i for i in range(102) if not combined[i]]}
    result["root"] = str(root)
    result["scope"] = "All source annotation files before quarantine or dedup; counts include dual annotations, not unique objects."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/root/loweye/data/external/ip102/Annotations"))
    parser.add_argument("--output", type=Path,
                        default=Path("/root/loweye/logs/ip102_source_class_audit.json"))
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be the existing source Annotations directory")
    if args.output.suffix.lower() != ".json" or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("--output must be a separate JSON outside the source annotation directory")
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {kind: {"files": result[kind]["files"], "valid_files": result[kind]["valid_files"],
                      "missing0based": result[kind]["missing0based"], "failure_count": len(result[kind]["failures"])}
               for kind in ("xml", "txt")}
    summary.update({"missing0based": result["combined"]["missing0based"], "output": str(args.output)})
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
