"""重排（Reranker）。

默认后端 lexical —— 确定性词法重排，输出 **绝对** 分数（不做 min-max 归一化）。
这一点很关键：只有绝对分数才能用来判断「知识库到底有没有覆盖这个问题」，
归一化后最高分永远是 1.0，拒答机制就成了摆设。

装上 torch + sentence-transformers 后可切到 cross_encoder（bge-reranker-v2-m3），
logits 经 sigmoid 转成 0~1 的绝对分数，语义一致。
"""

from __future__ import annotations

import math
import re
from typing import Protocol, runtime_checkable

from ..schemas import Clause
from ..text import char_ngrams, tokenize

_CLAUSE_NO_RE = re.compile(r"\d+(?:\.\d+){1,3}")


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, query: str, clauses: list[Clause]) -> list[float]:
        """返回与 clauses 等长的绝对相关度分数，取值 0~1。"""
        ...


class LexicalReranker:
    """词法重排：条款号命中 > 词覆盖率 > 字级重合，最后做轻微长度惩罚。

    权重写死在类属性里便于单测与答辩解释；不做学习，因而不需要训练数据。

    可选传入术语表：覆盖率按「术语概念」折叠后计算（见 TermLexicon.collapse），
    这样「瓷瓶」与追加进来的「绝缘子」不会被算成两个概念。
    """

    name = "lexical"
    W_COVERAGE = 0.55
    W_BIGRAM = 0.25
    W_CLAUSE_NO = 0.20

    def __init__(self, lexicon=None) -> None:
        self.lexicon = lexicon

    def _tokens(self, text: str) -> set[str]:
        raw = tokenize(text)
        if self.lexicon is None:
            return set(raw)
        return self.lexicon.collapse(raw)

    def score(self, query: str, clauses: list[Clause]) -> list[float]:
        if not clauses:
            return []
        q_tokens = self._tokens(query)
        q_bigrams = set(char_ngrams(query, 2, 3))
        q_clause_nos = set(_CLAUSE_NO_RE.findall(query))

        scores: list[float] = []
        for clause in clauses:
            doc_text = f"{clause.clause_no} {clause.hierarchy_path} {clause.text}"
            d_tokens = self._tokens(doc_text)
            coverage = (len(q_tokens & d_tokens) / len(q_tokens)) if q_tokens else 0.0

            d_bigrams = set(char_ngrams(clause.text, 2, 3))
            bigram = (len(q_bigrams & d_bigrams) / len(q_bigrams)) if q_bigrams else 0.0

            clause_hit = 0.0
            if q_clause_nos:
                clause_hit = 1.0 if clause.clause_no in q_clause_nos else 0.0
                if not clause_hit and any(
                    clause.clause_no.startswith(no) for no in q_clause_nos
                ):
                    clause_hit = 0.6  # 问到某一节，节下的条也算相关

            raw = (
                self.W_COVERAGE * coverage
                + self.W_BIGRAM * bigram
                + self.W_CLAUSE_NO * clause_hit
            )
            # 长度惩罚：很长的条款只在内容确实命中时才值得排前面
            length_penalty = 1.0 / (1.0 + math.log1p(max(0, len(clause.text) - 200)) / 12.0)
            scores.append(max(0.0, min(1.0, raw * length_penalty)))
        return scores


class CrossEncoderReranker:
    """bge-reranker-v2-m3 等交叉编码器（可选增强）。

    坑：sentence-transformers 的 CrossEncoder 在 num_labels=1 时**默认已经套了一层
    Sigmoid**（activation_fn 默认就是 Sigmoid）。如果这里再套一层，两次 sigmoid 会把
    分数压到 0.5 附近——实测域内题 0.51~0.73、域外题 0.50~0.66，两边几乎重叠，
    阈值拒答直接失效（拒答率从 100% 掉到 38%）。所以显式把激活换成 Identity，
    由本类自己套一层 sigmoid，语义清楚且不随库版本漂移。
    """

    name = "cross_encoder"

    def __init__(self, model_name_or_path: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder  # 延迟导入

        self._model = CrossEncoder(model_name_or_path, activation_fn=torch.nn.Identity())

    def score(self, query: str, clauses: list[Clause]) -> list[float]:
        if not clauses:
            return []
        pairs = [[query, f"{c.clause_no} {c.hierarchy_path}\n{c.text}"] for c in clauses]
        logits = self._model.predict(pairs)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def build_reranker(
    kind: str,
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    model_path: str | None = None,
    lexicon=None,
):
    """返回 (重排器, 降级原因)。"""
    kind = (kind or "lexical").lower()
    if kind in ("cross_encoder", "bge_reranker", "reranker"):
        try:
            return CrossEncoderReranker(model_path or model_name), None
        except Exception as exc:
            return (
                LexicalReranker(lexicon),
                f"交叉编码器重排不可用（{type(exc).__name__}: {exc}），已降级为词法重排",
            )
    return LexicalReranker(lexicon), None
