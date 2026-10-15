"""交付物本地验收：回收回来的 ONNX / best / last 是否真的能跑。

付费主机证明的是「那边对」：tools/gpu/train_inspection_fcos.py 里 torch 与
onnxruntime 的逐检测框 parity，加上 tools/gpu/evaluate_onnx_detector.py 的独立
COCOeval。这个脚本证明的是「这边可用」：

1. 磁盘上的字节仍等于回收收据记录的 SHA256（传输没坏、回收后没人动过）；
2. metadata 的类别表/label id 与 manifest 自洽；
3. ONNX 声明的输入输出契约正是训练器承诺的那一份；
4. 项目自己的后端 loweye/detect/onnx_detector.py 能加载它并返回 DetectResult。

检测用合成图像，不需要数据集原图，也不会再从付费主机拉任何东西。

用法：
    python -m tools.verify_delivered_detectors
    python -m tools.verify_delivered_detectors --only crop_pest
    # 对探针产物排练，不碰正式路径：
    python -m tools.verify_delivered_detectors --out-dir out/inspection_gpu/_localcheck \
        --onnx fire_smoke=out/inspection_gpu/_localcheck/fire_smoke_fcos_640.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "out" / "inspection_gpu"
TASKS = (("fire_smoke", 640), ("crop_pest", 800))
OUTPUT_TYPES = {"boxes": "tensor(float)", "labels": "tensor(int64)", "scores": "tensor(float)"}
MODEL_SUFFIXES = (".onnx", ".pth")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_entries(out_dir: Path, task: str) -> list[dict]:
    """回收收据里的条目，接受三种真实出现过的形状。

    * tools/gpu/monitor_inspection.py 直接写一个 JSON 列表（真实回收的唯一形态）；
    * 早期/手工演练写成 {"recovered": [...]} 或 {"recovered": {...}}；
    * 单条目 dict（自身带 remote/sha256）。
    收据缺失或不是合法 JSON 时返回空列表（调用方转成 not_recovered）。
    """
    path = out_dir / f"{task}_recovery.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        recovered = payload.get("recovered")
        if isinstance(recovered, list):
            return [entry for entry in recovered if isinstance(entry, dict)]
        if isinstance(recovered, dict):
            return [recovered]
        if "remote" in payload and "sha256" in payload:
            return [payload]
    return []


def entry_for(entries: list[dict], suffix: str) -> dict:
    for entry in entries:
        if str(entry.get("remote", "")).endswith(suffix):
            return entry
    return {}


def local_for(remote: str) -> Path:
    """回收落点约定（与 tools/gpu/monitor_inspection.py 一致）。"""
    name = Path(str(remote)).name
    if name.endswith(MODEL_SUFFIXES):
        return PROJECT / "models" / "detector" / name
    return DEFAULT_OUT / name


def verify_task(task: str, size: int, out_dir: Path, onnx_override: Path | None) -> dict:
    checks: list[dict] = []

    def note(name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
        checks.append({"check": name, "ok": bool(ok), "required": required, "detail": detail})
        return bool(ok)

    entries = receipt_entries(out_dir, task)
    if not entries:
        note("recovery_receipt", False, f"没有回收收据：{out_dir / f'{task}_recovery.json'}")
        return {"task": task, "ok": False, "status": "not_recovered", "checks": checks}

    report: dict = {}
    for name in (f"{task}_onnx_eval.json", f"{task}_eval.json"):
        report = load_json(out_dir / name)
        if report:
            break

    # ---- 权重字节 ----------------------------------------------------------
    onnx_entry = entry_for(entries, ".onnx")
    onnx_path = onnx_override or (local_for(onnx_entry["remote"]) if onnx_entry else Path())
    if not note("onnx_in_receipt", bool(onnx_entry), str(onnx_entry.get("remote", ""))):
        pass
    onnx_ok = note("onnx_file_exists", bool(onnx_path) and Path(onnx_path).exists(), str(onnx_path))
    if onnx_ok:
        actual = sha256_file(onnx_path)
        note(
            "onnx_sha256_matches_receipt",
            actual == onnx_entry.get("sha256"),
            f"local {actual[:16]}… receipt {str(onnx_entry.get('sha256'))[:16]}…",
        )
        report_sha = str(report.get("artifact", {}).get("onnx_sha256", ""))
        note(
            "onnx_sha256_matches_eval_report",
            actual == report_sha,
            f"eval {report_sha[:16]}…" if report_sha else "本地还没有独立评测报告",
            required=False,
        )

    for tag in ("best", "last"):
        entry = entry_for(entries, f"fcos_{tag}.pth")
        path = local_for(entry["remote"]) if entry else Path()
        exists = note(f"{tag}_file_exists", bool(entry) and path.exists(), str(path))
        if exists:
            note(
                f"{tag}_sha256_matches_receipt",
                sha256_file(path) == entry.get("sha256"),
                f"{entry.get('sha256', '')[:16]}…",
            )

    # ---- 契约与类别表 ------------------------------------------------------
    metadata = load_json(out_dir / f"{task}_fcos_metadata.json")
    manifest = load_json(out_dir / f"{task}_manifest.json")
    classes = list(metadata.get("classes") or [])
    label_ids = list(metadata.get("label_ids") or [])
    note("metadata_present", bool(metadata), f"classes={len(classes)} label_ids={len(label_ids)}")
    note("label_ids_match_classes", len(classes) == len(label_ids) and bool(classes), "")
    if manifest.get("classes"):
        note(
            "classes_match_manifest",
            list(manifest["classes"]) == classes,
            f"manifest={len(manifest['classes'])} metadata={len(classes)}",
        )

    checks_by_name = {item["check"]: item for item in checks}
    onnx_ready = all(
        checks_by_name[name]["ok"]
        for name in (
            "onnx_file_exists",
            "onnx_sha256_matches_receipt",
            "metadata_present",
            "label_ids_match_classes",
        )
    )
    if not onnx_ready:
        return {"task": task, "ok": False, "status": "artifact_problem", "checks": checks}

    # ---- 项目自己的后端 ----------------------------------------------------
    from loweye.detect.onnx_detector import OnnxDetector

    detector = OnnxDetector(
        onnx_path=str(Path(onnx_path).resolve()),
        input_size=size,
        class_names=classes,
        slice_enabled=False,
        model_version="verify_delivered_detectors",
    )
    session = detector.session
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    note("single_input", len(inputs) == 1, ",".join(item.name for item in inputs))
    if inputs:
        shape = [str(dim) for dim in inputs[0].shape]
        note(
            "input_contract",
            inputs[0].type == "tensor(float)" and shape[1:] == ["3", str(size), str(size)],
            f"{inputs[0].name} {inputs[0].type} {shape}",
        )
    produced = {item.name: item.type for item in outputs}
    note(
        "output_contract",
        produced == OUTPUT_TYPES,
        json.dumps(produced, ensure_ascii=False),
    )
    note("layout_is_tv_dets", detector._layout == "tv_dets", detector._layout)

    import numpy as np
    from PIL import Image

    probe_image = PROJECT / "out" / "_localcheck_input.png"
    probe_image.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), mode="RGB").save(probe_image)

    result = detector.detect(probe_image, "localcheck")
    detections = list(getattr(result, "detections", []))
    unknown = sorted({item.cls for item in detections if item.cls not in classes})
    note("project_detector_ran", result.detector == "onnx", f"detector={result.detector}")
    note("detections_in_class_table", not unknown, f"unknown={unknown}")
    top = max(detections, key=lambda item: item.score, default=None)

    return {
        "task": task,
        "ok": all(item["ok"] for item in checks if item["required"]),
        "status": "ok",
        "onnx": str(onnx_path),
        "classes": len(classes),
        "layout": detector._layout,
        "detections": len(detections),
        "elapsed_ms": float(getattr(result, "elapsed_ms", 0.0)),
        "top_detection": None if top is None else {"cls": top.cls, "score": round(float(top.score), 6)},
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--onnx", action="append", default=[], help="task=path 覆盖（排练用）")
    parser.add_argument("--json", default="")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    overrides = {}
    for item in arguments.onnx:
        task, _, path = item.partition("=")
        overrides[task.strip()] = Path(path.strip())

    wanted = set(arguments.only)
    results = [
        verify_task(task, size, out_dir, overrides.get(task))
        for task, size in TASKS
        if not wanted or task in wanted
    ]

    payload = {"out_dir": str(out_dir), "tasks": results}
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in results:
        summary = json.dumps(
            {key: item[key] for key in ("task", "status", "ok", "classes", "layout", "detections", "elapsed_ms") if key in item},
            ensure_ascii=False,
        )
        print(summary)
        for check in item["checks"]:
            if not check["ok"]:
                mark = "FAIL" if check["required"] else "gap "
                print(f"  {mark} {check['check']}: {check['detail']}")

    if all(item["status"] == "not_recovered" for item in results):
        print("nothing recovered yet")
        return 3
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
