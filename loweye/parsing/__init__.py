"""M2 解析层：把规程文档变成可检索、可引用的条款。"""

from .loader import LoadedDoc, discover_documents, load_document, load_source_registry
from .builder import BuildReport, build_clauses, write_clauses_jsonl, read_clauses_jsonl

__all__ = [
    "LoadedDoc",
    "load_document",
    "discover_documents",
    "load_source_registry",
    "BuildReport",
    "build_clauses",
    "write_clauses_jsonl",
    "read_clauses_jsonl",
]
