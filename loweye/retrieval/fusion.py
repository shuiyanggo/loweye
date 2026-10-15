"""RRF（Reciprocal Rank Fusion）融合。

BM25 分与向量余弦分的量纲不同，直接加权求和需要调参且不稳定；
RRF 只用名次，天然免调参，是「两路召回 → 单一排序」最省事也最稳的做法。
公式：score(d) = Σ_r 1 / (k + rank_r(d))，k 取 60。
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, k: int = 60, weights: list[float] | None = None
) -> dict[int, float]:
    """rankings：每路召回的 doc 下标列表，已按各自分数从高到低排列。"""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 长度必须与 rankings 一致")

    fused: dict[int, float] = {}
    for path_idx, ranking in enumerate(rankings):
        weight = float(weights[path_idx])
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return fused


def top_indices(scores: list[float], k: int) -> list[int]:
    """按分数取前 k 个下标，同分时按下标升序，保证结果可复现。"""
    if k <= 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def ranks_from_order(order: list[int]) -> dict[int, int]:
    return {doc: rank for rank, doc in enumerate(order, start=1)}
