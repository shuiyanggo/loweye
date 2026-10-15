"""金标问答集的加载与判定。

判定规则写在这里（而不是散在脚本里）：
- 域内问题：返回结果的 (doc_id, clause_no) 之一命中期望集合即算命中；
  用 doc 限定的原因是条款号跨文档重号（1.1.3 在两篇演示文档里都存在）。
- 越界问题：期望 covered=False，与 Recall 分开统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GoldItem:
    id: str
    question: str
    doc: str
    clauses: list[str] = field(default_factory=list)
    note: str = ""
    group: str = "literal"  # literal=用文档原词提问；paraphrase=现场口吻改写

    def matches(self, doc_id: str, clause_no: str) -> bool:
        return doc_id == self.doc and clause_no in self.clauses


@dataclass
class RefusalItem:
    id: str
    question: str
    why: str = ""


@dataclass
class GoldSet:
    in_scope: list[GoldItem]
    out_of_scope: list[RefusalItem]

    @property
    def n_in_scope(self) -> int:
        return len(self.in_scope)

    def by_group(self, group: str) -> list[GoldItem]:
        return [g for g in self.in_scope if g.group == group]

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []
        for g in self.in_scope:
            if g.group not in seen:
                seen.append(g.group)
        return seen

    @property
    def n_out_of_scope(self) -> int:
        return len(self.out_of_scope)

    def to_dict(self) -> dict:
        return {
            "in_scope": [g.id for g in self.in_scope],
            "out_of_scope": [r.id for r in self.out_of_scope],
            "n_in_scope": self.n_in_scope,
            "n_out_of_scope": self.n_out_of_scope,
        }


def load_goldset(path: str | Path) -> GoldSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    in_scope = [
        GoldItem(
            id=str(item["id"]),
            question=str(item["question"]),
            doc=str(item["doc"]),
            clauses=[str(c) for c in (item.get("clauses") or [])],
            note=str(item.get("note", "")),
            group=str(item.get("group", "literal")),
        )
        for item in (data.get("in_scope") or [])
    ]
    out_of_scope = [
        RefusalItem(id=str(item["id"]), question=str(item["question"]), why=str(item.get("why", "")))
        for item in (data.get("out_of_scope") or [])
    ]
    if not in_scope:
        raise ValueError(f"金标集 {path} 里没有 in_scope 题目")
    return GoldSet(in_scope=in_scope, out_of_scope=out_of_scope)
