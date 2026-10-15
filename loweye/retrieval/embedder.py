"""向量召回：可插拔嵌入后端。

默认后端 hashing_tfidf —— 特征哈希 + IDF 加权的稀疏向量，纯 numpy 实现，
零下载、零显存、结果确定。它不是神经网络嵌入，但它是一套真实可复现的
向量检索，且把「词级 + 字级 n-gram」两类特征都用上了，中文规程检索的
召回质量足够撑起演示与评测。

装上 torch + sentence-transformers 后，改 config/app.yaml 的
retrieval.embedder 为 sentence_transformer 即可切到 BGE-M3，
两者接口一致，可以跑同一套评测脚本做对照（方案 v2 第 4 章 M6）。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..text import char_ngrams, tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # 每行 L2 归一化
        ...


class HashingTfidfEmbedder:
    """特征哈希 + IDF 加权，输出 L2 归一化的稠密向量（可直接点积算余弦）。"""

    name = "hashing_tfidf"

    def __init__(self, dim: int = 4096, n_min: int = 2, n_max: int = 3) -> None:
        self.dim = int(dim)
        self.n_min = n_min
        self.n_max = n_max
        self._df = np.zeros(self.dim, dtype=np.float64)
        self._n_docs = 0
        self._fitted = False

    # ---- 特征 ----
    def _features(self, text: str) -> list[str]:
        return tokenize(text) + char_ngrams(text, self.n_min, self.n_max)

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dim

    # ---- 训练 / 编码 ----
    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for bucket in {self._bucket(f) for f in set(self._features(text))}:
                df[bucket] += 1.0
        self._df = df
        self._n_docs = len(texts)
        self._fitted = True
        return self

    def _idf(self, bucket: int) -> float:
        return math.log((1.0 + self._n_docs) / (1.0 + self._df[bucket])) + 1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            features = self._features(text)
            if not features:
                continue
            counts: dict[int, float] = {}
            for feature in features:
                bucket = self._bucket(feature)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            for bucket, tf in counts.items():
                out[row, bucket] = (1.0 + math.log(tf)) * self._idf(bucket)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    # ---- 持久化 ----
    def state(self) -> dict:
        return {
            "kind": self.name,
            "dim": self.dim,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "n_docs": self._n_docs,
            "df": self._df.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "HashingTfidfEmbedder":
        obj = cls(dim=int(state["dim"]), n_min=int(state.get("n_min", 2)), n_max=int(state.get("n_max", 3)))
        obj._df = np.asarray(state["df"], dtype=np.float64)
        obj._n_docs = int(state.get("n_docs", 0))
        obj._fitted = True
        return obj


class SentenceTransformerEmbedder:
    """BGE-M3 等神经网络嵌入（可选增强，需 torch + sentence-transformers）。"""

    name = "sentence_transformer"

    def __init__(self, model_name_or_path: str, dim: int = 1024, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self._model = SentenceTransformer(model_name_or_path)
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension() or dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)


def build_embedder(
    kind: str,
    texts: list[str],
    *,
    dim: int = 4096,
    model_name: str = "BAAI/bge-m3",
    model_path: str | None = None,
    state: dict | None = None,
) -> tuple[Embedder, str | None]:
    """返回 (嵌入器, 降级原因)。降级原因非空表示没走成用户配置的后端。"""
    kind = (kind or "hashing_tfidf").lower()
    if kind in ("sentence_transformer", "bge", "bge_m3"):
        try:
            embedder = SentenceTransformerEmbedder(model_path or model_name)
            return embedder, None
        except Exception as exc:  # 缺库、缺权重、显存不足都走这里
            reason = f"神经网络嵌入不可用（{type(exc).__name__}: {exc}），已降级为 hashing_tfidf"
            fallback = HashingTfidfEmbedder(dim=dim)
            fallback.fit(texts)
            return fallback, reason

    embedder = HashingTfidfEmbedder(dim=dim)
    if state and state.get("kind") == embedder.name and int(state.get("dim", 0)) == dim:
        embedder = HashingTfidfEmbedder.from_state(state)
    else:
        embedder.fit(texts)
    return embedder, None


def save_embedder_state(embedder: Embedder, path: str | Path) -> None:
    if not isinstance(embedder, HashingTfidfEmbedder):
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(embedder.state()), encoding="utf-8")


def load_embedder_state(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
