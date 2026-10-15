"""领域术语归一化与查询扩展。

一线提问用的是口语，知识库文档用的是规程术语。两者词面不对齐时，纯词法检索
（BM25 + 字符 n-gram 重排）会失分；而单纯降低拒答阈值只会让错误条款被当成
处置依据，比拒答更糟。因此这里做的是**查询侧对齐**：把提问里的口语说法换成
规范术语（归一化），再把命中的规范术语追加进查询（扩展）。

两件事都发生在检索前，不改动知识库原文与索引，因此可随时关掉做消融对照。
表里没有的写法不会被猜译 —— 归一化只替换确实命中的别名，宁可扩不出来，
也不做模糊联想。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TermEntry:
    canonical: str
    aliases: list[str] = field(default_factory=list)

    @property
    def all_forms(self) -> list[str]:
        return [self.canonical, *self.aliases]


class TermLexicon:
    """术语表。`normalize` 做替换，`expand` 做替换 + 追加规范术语。"""

    def __init__(self, entries: list[TermEntry] | None = None) -> None:
        self.entries: list[TermEntry] = list(entries or [])
        # 长别名优先匹配，避免「碎了一片」被更短的「碎了」抢先切断
        pairs: list[tuple[str, str]] = []
        for entry in self.entries:
            for alias in entry.aliases:
                pairs.append((alias, entry.canonical))
        pairs.sort(key=lambda kv: (-len(kv[0]), kv[0]))
        self._alias_to_canonical = pairs
        # 折叠用的查表：任何形式（规范词或别名）→ 规范词，O(1) 命中。
        # collapse() 在每次检索的每个词上都会调用，线性扫 47 条术语表太慢。
        self._form_to_canonical: dict[str, str] = {}
        for entry in self.entries:
            self._form_to_canonical[entry.canonical] = entry.canonical
            for alias in entry.aliases:
                self._form_to_canonical.setdefault(alias, entry.canonical)

    # ---------------------------------------------------------------- 构造

    @classmethod
    def from_dict(cls, data: dict) -> "TermLexicon":
        entries: list[TermEntry] = []
        for raw in (data or {}).get("terms") or []:
            canonical = str(raw.get("canonical", "")).strip()
            if not canonical:
                continue
            aliases = [str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip()]
            entries.append(TermEntry(canonical=canonical, aliases=aliases))
        return cls(entries)

    @classmethod
    def load(cls, path: str | Path | None) -> "TermLexicon":
        if not path:
            return cls([])
        p = Path(path)
        if not p.exists():
            return cls([])
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    # ---------------------------------------------------------------- 能力

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def alias_count(self) -> int:
        return len(self._alias_to_canonical)

    def canonicals_in(self, text: str) -> list[str]:
        """返回文本里出现的规范术语（保持术语表顺序，去重）。"""
        if not text:
            return []
        hits: list[str] = []
        for entry in self.entries:
            if entry.canonical in text:
                hits.append(entry.canonical)
                continue
            if any(alias in text for alias in entry.aliases):
                hits.append(entry.canonical)
        return hits

    def normalize(self, text: str) -> str:
        """把口语别名替换成规范术语（长别名优先，替换后不重复替换）。"""
        if not text or not self._alias_to_canonical:
            return text or ""
        out = text
        for alias, canonical in self._alias_to_canonical:
            if alias in out and canonical not in out:
                out = out.replace(alias, canonical)
        return out

    def expand(self, text: str) -> str:
        """原始提问 + 追加命中的规范术语，供 BM25 与重排共用。

        这里刻意**不改写原词**，只做追加。曾经试过「把口语别名替换成规范术语」，
        结果是把判别性最强的词也一起弄丢了：例如「明火扑灭后还要监视多久」里的
        「明火」被换成「火情」之后，恰好命中不了写着「明火扑灭后」的那一条。
        追加不丢信息，代价只是查询串变长、绝对分被摊薄一点 —— 用阈值消化即可。
        """
        canonicals = self.canonicals_in(text or "")
        base = (text or "").strip()
        if not canonicals:
            return base
        return f"{base} {' '.join(canonicals)}".strip()

    def collapse(self, tokens) -> set[str]:
        """把一批分词结果按「术语概念」折叠，同一概念的多种说法只算一次。

        为什么需要：查询扩展会把「瓷瓶」和追加的「绝缘子」同时放进查询串，
        若不折叠，同一个概念会占两个名额，覆盖率的分母被虚增，正确条款的
        绝对分反而被拉低。折叠之后，覆盖率回答的是「用户在说几个概念、
        其中几个在这条里出现了」，而不是「用户打了多少个词」。
        """
        out: set[str] = set()
        for token in tokens or []:
            if not token:
                continue
            canonical = self._canonical_of(token)
            out.add(canonical or token)
        return out

    def _canonical_of(self, token: str) -> str | None:
        return self._form_to_canonical.get(token)

    def describe(self, text: str) -> dict:
        """给可解释性用：原查询、归一化结果、实际用于检索的查询串、命中的术语。"""
        normalized = self.normalize(text or "")
        expanded = self.expand(text or "")
        return {
            "query": text or "",
            "normalized": normalized,
            "expanded": expanded,
            "matched_terms": self.canonicals_in(text or ""),
        }


def build_lexicon(cfg=None) -> TermLexicon:
    """按配置构造术语表；未配置或文件缺失时返回空表（等价于关闭该功能）。"""
    if cfg is None:
        return TermLexicon([])
    path_value = cfg.get("retrieval.lexicon", "config/lexicon.yaml")
    if not path_value:
        return TermLexicon([])
    return TermLexicon.load(cfg.abs_path(path_value))
