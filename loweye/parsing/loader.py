"""文档读取：PDF / TXT / Markdown → 纯文本 + 页码映射。

页码映射是「可溯源」的一部分：引用的条款要能告诉用户它出自第几页。
扫描件（无文字层）会被识别出来并给出明确警告，而不是静默产出空知识库 ——
这一类文档需要先走 OCR（方案 v2 里列的 MinerU / PaddleOCR），
本模块不假装能读扫描件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass
class LoadedDoc:
    doc_id: str
    title: str
    text: str
    page_map: dict[int, tuple[int, int]] = field(default_factory=dict)  # 行号 → (页码, 页内行号)
    n_pages: int = 0
    source_version: str = ""
    source_date: str = ""
    source_url: str = ""
    path: str = ""
    warnings: list[str] = field(default_factory=list)


def load_source_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """读取来源登记表，统一归一化为 {"文件名": {doc_id, title, source_version, ...}}。

    支持两种写法：列表式（每条带 path，推荐，顺序稳定）与字典式（以文件名为键）。
    （方案 v2 M2：每一份知识文档都必须记录来源、版本、发布日期，
    否则引用无法被验证，答辩时也说不清知识库的权威性。）
    """
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    docs = data.get("documents", {}) or {}

    registry: dict[str, dict[str, Any]] = {}
    if isinstance(docs, dict):
        for name, meta in docs.items():
            if isinstance(meta, dict):
                registry[str(name)] = _normalize_meta(meta)
        return registry
    if isinstance(docs, list):
        for meta in docs:
            if not isinstance(meta, dict):
                continue
            raw_path = meta.get("path") or meta.get("file") or meta.get("name")
            if not raw_path:
                continue
            registry[Path(str(raw_path)).name] = _normalize_meta(meta)
    return registry


def _normalize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """把 source_version / version 这类同义字段归一，避免登记表写法影响入库结果。"""
    out = dict(meta)
    for canonical, aliases in (
        ("source_version", ("version",)),
        ("source_date", ("date",)),
        ("source_url", ("url",)),
    ):
        if not out.get(canonical):
            for alias in aliases:
                if out.get(alias):
                    out[canonical] = out[alias]
                    break
    return out


def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> tuple[str, dict[int, tuple[int, int]], int, list[str]]:
    import fitz  # PyMuPDF

    warnings: list[str] = []
    pages: list[str] = []
    page_map: dict[int, tuple[int, int]] = {}
    line_cursor = 0
    empty_pages = 0

    with fitz.open(path) as doc:
        n_pages = doc.page_count
        for pno in range(n_pages):
            page = doc.load_page(pno)
            text = page.get_text("text")
            if not text.strip():
                empty_pages += 1
            lines = text.split("\n")
            for i, _ in enumerate(lines):
                page_map[line_cursor + i] = (pno + 1, i + 1)
            line_cursor += len(lines)
            pages.append(text)

    if empty_pages == n_pages and n_pages > 0:
        warnings.append(
            f"{path.name}：{n_pages} 页均无可提取文字层，判定为扫描件，"
            "需要先做 OCR（MinerU / PaddleOCR）后才能入库"
        )
    elif empty_pages:
        warnings.append(f"{path.name}：{empty_pages}/{n_pages} 页无文字层，这些页不会产出条款")
    return "\n".join(pages), page_map, n_pages, warnings


def load_document(
    path: str | Path,
    *,
    registry: dict[str, dict[str, Any]] | None = None,
    doc_id: str | None = None,
) -> LoadedDoc:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"知识文档不存在：{p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")

    meta = (registry or {}).get(p.name, {})
    warnings: list[str] = []

    if suffix == ".pdf":
        text, page_map, n_pages, pdf_warnings = _load_pdf(p)
        warnings.extend(pdf_warnings)
    else:
        text = _read_text_file(p)
        page_map = {}
        n_pages = 0

    if not text.strip():
        warnings.append(f"{p.name}：未提取到任何文本，请检查文件是否损坏或为纯图片")

    return LoadedDoc(
        doc_id=doc_id or meta.get("doc_id") or p.stem,
        title=meta.get("title") or p.stem,
        text=text,
        page_map=page_map,
        n_pages=n_pages,
        source_version=str(meta.get("source_version", "") or ""),
        source_date=str(meta.get("source_date", "") or ""),
        source_url=str(meta.get("source_url", "") or ""),
        path=str(p),
        warnings=warnings,
    )


def discover_documents(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def dumps_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
