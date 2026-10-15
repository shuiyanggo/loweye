"""知识库构建：文档 → 条款 → JSONL 落盘。

产出 data/processed/clauses.jsonl，之后检索层只读这个文件，
保证「切分结果」是一个可被检查、可被评审看见的中间产物，
而不是藏在内存里的一次性动作。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..schemas import Clause
from ..text import dedupe_clauses, split_clauses
from .loader import LoadedDoc, discover_documents, load_document, load_source_registry


@dataclass
class BuildReport:
    docs: int = 0
    chunks: int = 0
    numbered: int = 0
    per_doc: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def numbered_ratio(self) -> float:
        return (self.numbered / self.chunks) if self.chunks else 0.0

    def to_dict(self) -> dict:
        return {
            "docs": self.docs,
            "chunks": self.chunks,
            "numbered": self.numbered,
            "numbered_ratio": round(self.numbered_ratio, 4),
            "per_doc": self.per_doc,
            "warnings": self.warnings,
        }


def doc_to_clauses(doc: LoadedDoc, *, min_chars: int = 12, max_chars: int = 1200) -> list[Clause]:
    raw = dedupe_clauses(
        split_clauses(doc.text, page_map=doc.page_map, min_chars=min_chars, max_chars=max_chars)
    )
    out: list[Clause] = []
    for i, rc in enumerate(raw, start=1):
        out.append(
            Clause(
                chunk_id=f"{doc.doc_id}#{i:04d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                clause_no=rc.clause_no,
                hierarchy_path=rc.hierarchy_path,
                page=rc.page,
                text=rc.text,
                source_version=doc.source_version,
                source_date=doc.source_date,
            )
        )
    return out


def build_clauses(cfg: Config, *, raw_dir: Path | None = None) -> tuple[list[Clause], BuildReport]:
    raw_dir = raw_dir or cfg.path("data_raw")
    registry = load_source_registry(cfg.root / "config" / "sources.yaml")
    report = BuildReport()
    all_clauses: list[Clause] = []

    for path in discover_documents(raw_dir):
        doc = load_document(path, registry=registry)
        report.warnings.extend(doc.warnings)
        clauses = doc_to_clauses(doc)
        all_clauses.extend(clauses)
        report.docs += 1
        report.per_doc[doc.doc_id] = len(clauses)
        report.chunks += len(clauses)
        report.numbered += sum(1 for c in clauses if not c.clause_no.startswith("段落"))
        if not clauses:
            report.warnings.append(f"{doc.title}：未切出任何条款")
    return all_clauses, report


def write_clauses_jsonl(clauses: list[Clause], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in clauses:
            fh.write(c.model_dump_json() + "\n")
    return p


def read_clauses_jsonl(path: str | Path) -> list[Clause]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[Clause] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(Clause.model_validate(json.loads(line)))
    return out
