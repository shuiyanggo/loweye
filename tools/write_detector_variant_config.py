"""写一份检测器变体配置（config/app.<task>_gpu.yaml），以 config/app.cplid_gpu.yaml 为模板。

为什么用工具而不是手改：crop_pest 有 102 个类别名，必须与训练清单的**顺序**完全一致；而
onnx_path / input_size / class_names / model_version 这四项只要有一项不对，构建期不会报错，
只有真正推理时才会暴露（这正是 tools/check_detector_config.py 存在的理由）。所以这四项由
元数据与命令行参数一次性写出，不留手改空间。

detect: 块用「文本整块替换」而不是 YAML round-trip，模板里其余的键与注释原样保留。

用法：
    python -m tools.write_detector_variant_config --task crop_pest \
        --onnx models/detector/crop_pest_fcos_800.onnx \
        --metadata out/inspection_gpu/crop_pest_fcos_metadata.json \
        --model-version crop-pest-fcos-r50fpn-800 --port 8853 \
        --out-file config/app.crop_pest_gpu.yaml

    # 交付后核对：committed 配置是否仍与最终元数据一致（一致则 identical=true）
    python -m tools.write_detector_variant_config ... --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT / "config" / "app.cplid_gpu.yaml"
INLINE_CLASS_LIMIT = 6
EXIT_OK = 0
EXIT_INPUT_PROBLEM = 2


def load_metadata(path: Path) -> tuple[list[str], int | None]:
    """返回 (类别名列表, 导出输入边长)；边长取自 preprocessing.input 的第 3 位。"""
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"metadata has no usable classes list: {path}")
    size: int | None = None
    shape = (data.get("preprocessing") or {}).get("input")
    if isinstance(shape, list) and len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
        size = int(shape[2])
    return list(classes), size


def _find_block(lines: list[str], key: str) -> tuple[int, int]:
    """返回顶层 `key:` 块的 [start, end)：end 是下一个顶层键（列 0 的非注释行）。"""
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = index
            break
    if start is None:
        raise KeyError(f"{key}: block not found in base config")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped and not lines[index][0].isspace() and not lines[index].startswith("#"):
            end = index
            break
    return start, end


def _detect_block(onnx_path: str, input_size: int, class_names: list[str], model_version: str) -> list[str]:
    """生成 detect: 块。类别少用行内列表（与模板一致），类别多用块状列表（102 类行内不可读）。"""
    if len(class_names) <= INLINE_CLASS_LIMIT:
        class_lines = [f"  class_names: [{', '.join(class_names)}]"]
    else:
        class_lines = ["  class_names:"] + [f"    - {name}" for name in class_names]
    return [
        "detect:",
        "  detector: onnx",
        f"  onnx_path: {onnx_path}",
        *class_lines,
        f"  input_size: {input_size}               # 与导出时一致；导出图里的尺寸是常量，喂别的尺寸会算错",
        "  nms_iou: 0.50",
        "  conf_threshold: 0.25",
        "  human_review_threshold: 0.60",
        f"  model_version: {model_version}",
        "  slice:",
        "    enabled: false               # 与独立评测协议一致（整幅 letterbox）；切片会改变指标口径",
        f"    size: {input_size}",
        "    overlap: 0.2",
    ]


def render_variant(
    base_text: str,
    *,
    onnx_path: str,
    input_size: int,
    class_names: list[str],
    model_version: str,
    port: int,
    title: str,
) -> str:
    """纯函数：把模板文本渲染成变体配置文本。"""
    if not class_names:
        raise ValueError("class_names must not be empty")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    lines = base_text.splitlines()

    start, end = _find_block(lines, "detect")
    lines = lines[:start] + _detect_block(onnx_path, input_size, class_names, model_version) + lines[end:]

    server_start, server_end = _find_block(lines, "server")
    for index in range(server_start + 1, server_end):
        if lines[index].startswith("  port:"):
            lines[index] = f"  port: {port}"
            break
    else:
        raise KeyError("server.port not found in base config")

    for index, line in enumerate(lines):
        if line.startswith("  subtitle:"):
            lines[index] = f"  subtitle: {title}"
            break
    else:
        raise KeyError("app.subtitle not found in base config")

    header_end = 0
    while header_end < len(lines) and (lines[header_end].startswith("#") or not lines[header_end].strip()):
        header_end += 1
    header = [
        f"# 低空之眼 · GPU 微调检测权重配置（FCOS-ResNet50-FPN，{title}）",
        "#",
        "# 权重来源：AutoDL GPU（tools/gpu/run_inspection_tasks.py 协调 prepare → 冒烟 → 24 轮训练），",
        "# 导出 ONNX 后由 tools/gpu/monitor_inspection.py 逐件校验 SHA256 再回收。",
        "#",
        "# 下面四项必须彼此一致，任何一项不对都只在推理期报错（先跑 tools/check_detector_config.py 可提前发现）：",
        f"#   onnx_path     = {onnx_path}",
        f"#   input_size    = {input_size}",
        f"#   class_names   = {len(class_names)} 类，顺序与训练清单一致",
        f"#   model_version = {model_version}",
        "#",
        "# 本文件由 tools/write_detector_variant_config.py 生成，不要手改（改参数后重新生成）。",
        "#",
        "# 用法：",
        "#   python -m tools.check_detector_config --config config/app.<task>_gpu.yaml",
        "#   python -m loweye.cli.serve --config config/app.<task>_gpu.yaml --no-reload",
        "#",
    ]
    return "\n".join(header + lines[header_end:]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="写一份检测器变体配置")
    parser.add_argument("--task", required=True)
    parser.add_argument("--onnx", required=True, help="仓库内相对路径：回收后的 ONNX 落点")
    parser.add_argument("--metadata", required=True, help="训练元数据 JSON（提供类别顺序与输入边长）")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--input-size", type=int, default=None, help="默认取元数据声明的边长；显式给出则必须与之一致")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--title", default=None, help="配置标题与 app.subtitle")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--base", default=str(BASE_CONFIG))
    parser.add_argument("--check", action="store_true", help="只核对已存在的文件是否与给定参数一致，不写入")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = PROJECT / metadata_path
    try:
        classes, declared_size = load_metadata(metadata_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}, ensure_ascii=False))
        return EXIT_INPUT_PROBLEM
    title = args.title or args.task
    out_path = Path(args.out_file)
    if not out_path.is_absolute():
        out_path = PROJECT / out_path
    onnx_path = args.onnx
    input_size = declared_size if declared_size is not None else args.input_size
    if input_size is None:
        print(json.dumps({"ok": False, "error": "metadata 未声明输入边长，请用 --input-size 显式指定"}, ensure_ascii=False))
        return EXIT_INPUT_PROBLEM
    if args.input_size is not None and int(args.input_size) != input_size:
        print(json.dumps({"ok": False, "error": f"input_size 不一致：命令行 {args.input_size} vs 元数据 {input_size}"}, ensure_ascii=False))
        return EXIT_INPUT_PROBLEM
    base_text = Path(args.base).read_text(encoding="utf-8")
    rendered = render_variant(
        base_text,
        onnx_path=onnx_path,
        input_size=input_size,
        class_names=classes,
        model_version=args.model_version,
        port=args.port,
        title=title,
    )
    existing = out_path.read_text(encoding="utf-8") if out_path.is_file() else None
    identical = existing == rendered
    if args.check:
        print(json.dumps({
            "ok": identical,
            "file": str(out_path.relative_to(PROJECT)) if out_path.is_absolute() else str(out_path),
            "exists": existing is not None,
            "identical": identical,
            "classes": len(classes),
            "input_size": input_size,
            "model_version": args.model_version,
        }, ensure_ascii=False))
        return EXIT_OK if identical else 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "file": str(out_path.relative_to(PROJECT)),
        "classes": len(classes),
        "input_size": input_size,
        "port": args.port,
        "model_version": args.model_version,
        "previous_identical": identical,
    }, ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
