"""Summarize a prepare_inspection_data.py outcome without printing the whole file.

Works on both shapes the preparation tool can leave behind:
  * the failure report (status != ok, an "errors" list of "<image>: <reason>")
  * the success manifest (status ok, split lists, quarantine audit, class coverage)

Prints counts only, so the output stays readable over an SSH capture.
"""
import collections
import json
import sys
from pathlib import Path


def bucket(lines):
    buckets = collections.Counter()
    for line in lines:
        if not isinstance(line, str):
            line = json.dumps(line, ensure_ascii=False)
        reason = line.split(": ", 1)[-1]
        buckets[reason[:150]] += 1
    return buckets


def main():
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"file={path} top_level_keys={sorted(data)}")
    if data.get("status") == "ok":
        print("status=ok")
        for split in ("train", "val", "test"):
            print(f"  {split}: {len(data.get(split, []))}")
        print(f"  classes: {data.get('classes')}")
        print(f"  annotation_formats: {data.get('annotation_formats')}")
        q = data.get("quarantine") or {}
        print(f"  quarantine enabled={q.get('enabled')} count={q.get('count')}")
        print(f"  quarantine by_split: {q.get('by_original_split')}")
        for reason, count in (q.get("by_reason") or {}).items():
            print(f"    {count:5d}  {reason[:150]}")
        cov = data.get("class_coverage") or {}
        print(f"  class_coverage policy: {cov.get('policy')}")
        print(f"  missing_classes: {[c.get('name') for c in (cov.get('missing_classes') or [])]}")
        audit = data.get("source_group_audit") or {}
        print(f"  cross_split_groups: {audit.get('cross_split_groups')}")
    else:
        errors = data.get("errors") or []
        if isinstance(errors, dict):
            errors = [f"{k}: {v}" for k, v in errors.items()]
        print(f"status=failed errors={len(errors)}")
        for reason, count in bucket(errors).most_common(12):
            print(f"    {count:5d}  {reason}")
        q = data.get("quarantine") or {}
        print(f"  quarantine: count={q.get('count')} by_reason={q.get('by_reason')}")


if __name__ == "__main__":
    main()
