"""核对 README 里写到的文件 / 目录路径是否真的存在于仓库里。

起因：README 第 11.2 节声称参赛方案书 PDF 在 `out/低空之眼-参赛方案书.pdf`（10 章 / 59 页），
而全项目 `**/*.pdf` 只有两份巡检报告——文档里写着的交付物并不存在。这类偏差不会让任何测试
变红，只会在读者去找那个文件时暴露，所以做成一条可重复执行的检查。

判定规则（明确、可测）：
- 只取 Markdown 反引号里的 token；
- 排除 URL、绝对路径（主机侧路径如 /root/loweye/...）、含通配符或占位符（{} * ? <> |）的 token；
- 目录候选：token 以 `/` 结尾，要求 is_dir()；
- 文件候选：token 含 `/` 且后缀在白名单内，要求 exists()；带 `:行号` 或 `:起-止` 的写法先剥掉行号。

用法：
    python -m tools.audit_readme_paths
    python -m tools.audit_readme_paths --file README.md --json-out out/readme_paths.json
退出码：0 = 全部命中；3 = 有缺失；2 = 文件或参数问题。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

EXIT_OK = 0
EXIT_MISSING = 3
EXIT_INPUT_PROBLEM = 2

BACKTICK = re.compile(r"`([^`\n]+)`")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")

FILE_SUFFIXES = {
    ".md", ".py", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".sh", ".ps1",
    ".png", ".jpg", ".jpeg", ".pdf", ".onnx", ".pth", ".html", ".css", ".js",
    ".csv", ".ipynb", ".toml", ".ini", ".log", ".mp4",
}


def strip_line_suffix(token: str) -> str:
    """把 `foo/bar.py:12-18` 还原成 `foo/bar.py`。"""
    return LINE_SUFFIX.sub("", token)


def is_directory_candidate(token: str) -> bool:
    if not token.endswith("/"):
        return False
    if token.startswith(("/", "http", "#")):
        return False
    return not any(ch in token for ch in "*?{}<>|")


def is_file_candidate(token: str) -> bool:
    if token.startswith(("/", "http", "#")):
        return False
    if any(ch in token for ch in "*?{}<>| \t"):
        return False
    if "/" not in token:
        return False
    return Path(token).suffix.lower() in FILE_SUFFIXES


def audit(text: str, root: Path, alt_roots: tuple[Path, ...] = ()) -> dict:
    """核对 text 里的路径候选。

    alt_roots 用来消化 README「目录结构」那一节：那里的路径是相对 `loweye/` 写的
    （`agent/pipeline.py` 指的是 `loweye/agent/pipeline.py`）。第一版只按项目根解析，
    结果 8 条缺失里有 6 条是这个原因造成的误报——判定规则宁可多看一眼基准目录，
    也不要给读者一堆不存在的「坏路径」。
    """
    roots = (root,) + tuple(alt_roots)
    checked: dict[str, dict] = {}
    for raw in BACKTICK.findall(text):
        token = strip_line_suffix(raw.strip())
        kind = None
        if is_directory_candidate(token):
            kind = "dir"
            probe = token.rstrip("/")
        elif is_file_candidate(token):
            kind = "file"
            probe = token
        if kind is None:
            continue
        targets = [(base / probe).resolve() for base in roots]
        hit = next((t for t in targets if t.exists()), None)
        checked[token] = {
            "kind": kind,
            "exists": hit is not None,
            "path": str(hit if hit is not None else targets[0]),
            "searched": [str(t) for t in targets],
        }
    missing = sorted(t for t, info in checked.items() if not info["exists"])
    return {
        "file": str(root / "README.md"),
        "root": str(root),
        "alt_roots": [str(p) for p in alt_roots],
        "checked_count": len(checked),
        "missing_count": len(missing),
        "missing": missing,
        "checked": checked,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="核对文档（README 或方案书章节等）中提到的路径是否存在")
    ap.add_argument("--file", default=str(PROJECT / "README.md"))
    ap.add_argument("--root", default=str(PROJECT), help="相对路径的解析基准")
    ap.add_argument(
        "--alt-root",
        action="append",
        default=None,
        help="附加解析基准（默认自动加入 <root>/loweye，因为 README 的目录结构一节以它为基准）",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    source = Path(args.file)
    if not source.is_file():
        print(f"找不到文件：{source}", file=sys.stderr)
        return EXIT_INPUT_PROBLEM
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"基准目录不存在：{root}", file=sys.stderr)
        return EXIT_INPUT_PROBLEM

    if args.alt_root is None:
        fallback = root / "loweye"
        alt_roots = (fallback,) if fallback.is_dir() else ()
    else:
        alt_roots = tuple(Path(p).resolve() for p in args.alt_root)
    report = audit(source.read_text(encoding="utf-8"), root, alt_roots)
    report["file"] = str(source)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("file", "checked_count", "missing_count", "missing")},
                     ensure_ascii=False, indent=2))
    if report["missing"]:
        print(f"{Path(report['file']).name} 提到但磁盘上不存在的路径：{report['missing_count']} 条", file=sys.stderr)
        return EXIT_MISSING
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
