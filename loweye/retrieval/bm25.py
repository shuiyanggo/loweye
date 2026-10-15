"""BM25 召回。

优先使用开源实现 rank_bm25；若环境里没有该库，自动退回到本文件内的
Okapi BM25 实现（同样是标准公式，含单测），保证系统在任何环境都能跑。
降级行为会写进返回结果，不静默。
"""

from __future__ import annotations

import math
from collections import Counter

from ..text import tokenize

K1 = 1.5
B = 0.75


def tokenize_for_bm25(text: str) -> list[str]:
    return tokenize(text)


class _OkapiBM25Fallback:
    """标准 Okapi BM25（无第三方依赖的兜底实现）。"""

    def __init__(self, corpus: list[list[str]], k1: float = K1, b: float = B) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1)))
                scores[i] += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return scores


class BM25Index:
    """对外统一接口，屏蔽底层实现差异。"""

    def __init__(self, texts: list[str]) -> None:
        self.tokenized = [tokenize_for_bm25(t) for t in texts]
        self.backend_name = "rank_bm25"
        try:  # pragma: no cover - 取决于环境是否装了 rank_bm25
            from rank_bm25 import BM25Okapi

            self._impl = BM25Okapi(self.tokenized, k1=K1, b=B)
        except Exception:
            self._impl = _OkapiBM25Fallback(self.tokenized)
            self.backend_name = "builtin_okapi"

    def scores(self, query: str) -> list[float]:
        tokens = tokenize_for_bm25(query)
        if not tokens or not self.tokenized:
            return [0.0] * len(self.tokenized)
        raw = self._impl.get_scores(tokens)
        # rank_bm25 的 idf 在词频极高时可能为负，负分对检索没有意义，截断到 0
        return [max(0.0, float(s)) for s in raw]

    @staticmethod
    def normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        hi = max(scores)
        if hi <= 0:
            return [0.0] * len(scores)
        return [s / hi for s in scores]
