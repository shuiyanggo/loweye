"""知识库：条款 + BM25 索引 + 向量矩阵 + 重排器 = 一个可检索对象。

检索链路（每一步都能在返回结果里看到）：
    BM25 top-k ┐
               ├─ RRF 融合 → 候选 → 重排 → 最终 top-k → 是否覆盖判定
    向量 top-k ┘
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..schemas import Clause, ScoredClause, SearchHit, SearchResponse
from ..text import is_exclusion_text, is_numbered_clause, is_scope_statement
from .bm25 import BM25Index
from .embedder import (
    Embedder,
    build_embedder,
    load_embedder_state,
    save_embedder_state,
)
from .fusion import reciprocal_rank_fusion, top_indices
from .lexicon import TermLexicon, build_lexicon
from .rerank import Reranker, build_reranker


@dataclass
class KnowledgeBaseStats:
    chunks: int
    docs: int
    bm25_backend: str
    embedder: str
    reranker: str
    dim: int

    def to_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "docs": self.docs,
            "bm25_backend": self.bm25_backend,
            "embedder": self.embedder,
            "reranker": self.reranker,
            "dim": self.dim,
        }


class KnowledgeBase:
    # 范围排除闸门在召回阶段往前看几条：排除声明常常不是召回第一，但基本都在前三。
    # 取 3 是保守值——宁可多拒一条，也不要答一条超出覆盖范围的问题。
    EXCLUSION_PROBE_K = 3

    def __init__(
        self,
        clauses: list[Clause],
        *,
        bm25: BM25Index,
        embedder: Embedder,
        vectors: np.ndarray | None,
        reranker: Reranker,
        bm25_top_k: int = 30,
        vector_top_k: int = 30,
        fused_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
        min_relevance: float = 0.18,
        require_clause_no: bool = True,
        lexicon: TermLexicon | None = None,
        degradations: list[str] | None = None,
    ) -> None:
        self.clauses = clauses
        self.bm25 = bm25
        self.embedder = embedder
        self.vectors = vectors
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.fused_top_k = fused_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.min_relevance = min_relevance
        self.require_clause_no = require_clause_no
        self.lexicon = lexicon or TermLexicon([])
        self.degradations = list(degradations or [])

    # ------------------------------------------------------------ 构建

    @classmethod
    def from_clauses(
        cls,
        clauses: list[Clause],
        cfg: Config | None = None,
        *,
        embedder_state: dict | None = None,
    ) -> "KnowledgeBase":
        degradations: list[str] = []
        texts = [f"{c.clause_no} {c.hierarchy_path}\n{c.text}" for c in clauses]

        bm25 = BM25Index([f"{c.clause_no} {c.text}" for c in clauses])

        kind = cfg.get("retrieval.embedder", "hashing_tfidf") if cfg else "hashing_tfidf"
        dim = int(cfg.get("retrieval.embed_dim", 4096)) if cfg else 4096
        # 本地权重目录走 abs_path：配置里写项目内相对路径（models/bge-m3）也能解析，
        # 直接交给 SentenceTransformer 的话会按进程 CWD 去找，换目录启动就找不到模型。
        emb_path = cfg.get("retrieval.embedder_model_path") if cfg else None
        embedder, why = build_embedder(
            kind,
            texts,
            dim=dim,
            model_name=(cfg.get("retrieval.embedder_model", "BAAI/bge-m3") if cfg else "BAAI/bge-m3"),
            model_path=(str(cfg.abs_path(emb_path)) if (cfg and emb_path) else None),
            state=embedder_state,
        )
        if why:
            degradations.append(why)
        vectors = embedder.encode(texts) if texts else None

        rkind = cfg.get("retrieval.reranker", "lexical") if cfg else "lexical"
        lexicon = build_lexicon(cfg)
        rrk_path = cfg.get("retrieval.reranker_model_path") if cfg else None
        reranker, why2 = build_reranker(
            rkind,
            model_name=(cfg.get("retrieval.reranker_model", "BAAI/bge-reranker-v2-m3") if cfg else "BAAI/bge-reranker-v2-m3"),
            model_path=(str(cfg.abs_path(rrk_path)) if (cfg and rrk_path) else None),
            lexicon=lexicon,
        )
        if why2:
            degradations.append(why2)

        def g(key: str, default):
            return cfg.get(f"retrieval.{key}", default) if cfg else default

        return cls(
            clauses,
            bm25=bm25,
            embedder=embedder,
            vectors=vectors,
            reranker=reranker,
            bm25_top_k=int(g("bm25_top_k", 30)),
            vector_top_k=int(g("vector_top_k", 30)),
            fused_top_k=int(g("fused_top_k", 20)),
            final_top_k=int(g("final_top_k", 5)),
            rrf_k=int(g("rrf_k", 60)),
            min_relevance=float(g("min_relevance", 0.18)),
            require_clause_no=bool(g("require_clause_no", True)),
            lexicon=build_lexicon(cfg),
            degradations=degradations,
        )

    # ------------------------------------------------------------ 检索

    def set_lexicon(self, lexicon: TermLexicon | None) -> None:
        """切换术语表（消融对照用）。重排器手里也有一份，必须一起换掉，
        否则会出现「检索按术语表扩展、打分却按原词折叠」的错配。"""
        self.lexicon = lexicon or TermLexicon([])
        if hasattr(self.reranker, "lexicon"):
            self.reranker.lexicon = self.lexicon

    def search(
        self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> SearchResponse:
        """检索条款。

        doc_ids：把检索范围限定在这些文档内（None = 全库）。场景化问答用它——
        输电缺陷的工单不该引用《植保作业指引》里的通用报告条款。
        实现方式是先把范围外文档的两路得分清零、再从候选里剔掉，
        而不是事后过滤结果：否则范围外的文档会先占满 top-k，把范围内的条款挤出去。
        """
        started = time.perf_counter()
        k_final = int(top_k or self.final_top_k)
        n = len(self.clauses)
        if n == 0:
            return SearchResponse(query=query, hits=[], covered=False, top_score=0.0, elapsed_ms=0.0)
        allowed = {str(d) for d in doc_ids} if doc_ids else None

        # 0) 术语归一化 + 查询扩展：把口语说法对齐到文档术语，再追加命中的规范术语。
        #    用户看到的仍是原问题，检索用的是 search_query（在响应里回传，便于解释）。
        described = self.lexicon.describe(query)
        search_query = described["expanded"] or query
        matched_terms = described["matched_terms"]

        # 1) 两路召回
        bm25_raw = self.bm25.scores(search_query)
        vec_scores: list[float]
        if self.vectors is not None and self.vectors.size:
            qvec = self.embedder.encode([search_query])[0]
            vec_scores = np.asarray(self.vectors @ qvec, dtype=np.float64).tolist()
        else:
            vec_scores = [0.0] * n
        if allowed is not None:
            bm25_raw = [
                s if self.clauses[i].doc_id in allowed else 0.0 for i, s in enumerate(bm25_raw)
            ]
            vec_scores = [
                s if self.clauses[i].doc_id in allowed else 0.0 for i, s in enumerate(vec_scores)
            ]
        bm25_rank = top_indices(bm25_raw, self.bm25_top_k)
        vec_rank = top_indices(vec_scores, self.vector_top_k)

        # 2) 候选集：两路召回**取并集**，按「两路中较好的名次」排序。
        #    以前只用 RRF 的前 20 名当候选集，实测有硬伤：换 BGE 向量后，
        #    留出题的正确答案常出现在向量通道第 1~8 名，却被 BM25 的强命中用
        #    RRF 名次压出候选集，重排再强也救不回来（留出题 R@1 反而从 8/10 掉到 4/10）。
        #    取并集后，任一路的强命中都能进重排，RRF 分仍照算，只用于展示与打平。
        best_rank: dict[int, int] = {}
        for rank_list in (bm25_rank, vec_rank):
            for pos, idx in enumerate(rank_list):
                if pos < best_rank.get(idx, 10**9):
                    best_rank[idx] = pos
        fused_order = sorted(best_rank.items(), key=lambda kv: (kv[1], kv[0]))[: self.fused_top_k]
        cand_idx = [
            i
            for i, _ in fused_order
            if allowed is None or self.clauses[i].doc_id in allowed
        ]
        # RRF 分照旧算出来：前端要展示两路融合分，同分时也用它打平
        fused = reciprocal_rank_fusion([bm25_rank, vec_rank], k=self.rrf_k)
        if self.require_clause_no:
            # 只允许真正带条款号的内容参与作答；文档开头的免责声明等无编号内容
            # 会被拦在这里。若整库都没有编号内容，则退回原候选集（并保持拒答阈值生效）。
            numbered = [i for i in cand_idx if is_numbered_clause(self.clauses[i].clause_no)]
            if numbered:
                cand_idx = numbered
        if not cand_idx:
            return SearchResponse(
                query=query,
                hits=[],
                covered=False,
                top_score=0.0,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        # 3) 重排（绝对分数）
        bm25_norm = BM25Index.normalize(bm25_raw)
        candidates = [self.clauses[i] for i in cand_idx]
        rerank_scores = self.reranker.score(search_query, candidates)

        scored: list[ScoredClause] = []
        for pos, idx in enumerate(cand_idx):
            scored.append(
                ScoredClause(
                    clause=self.clauses[idx],
                    bm25=round(float(bm25_norm[idx]), 6),
                    vector=round(float(vec_scores[idx]), 6),
                    rrf=round(float(fused.get(idx, 0.0)), 6),
                    rerank=round(float(rerank_scores[pos]), 6),
                    stages=["bm25", "vector", "rrf", "rerank"],
                )
            )
        scored.sort(key=lambda s: (-s.rerank, -s.rrf, s.clause.chunk_id))
        # 范围排除闸门：最相似的那条若是「本导则不适用于……」，说明该问题落在知识库
        # 明确排除的范围里。这类问题词面高度重合、分数往往还不低，只靠阈值拦不住。
        #
        # 判据取「召回阶段的第一名」而不是「重排后的第一名」：召回（BM25/向量）衡量的是
        # 主题相近，重排衡量的是「这条能不能当答案」，两者不是一回事。实测换 BGE 重排后，
        # 排除声明会被重排从第 1 名挤到后面（`demo_line_guide:1.1.2` → 排到第 2、3 名），
        # 于是 scope_excluded 变 false、分数又高于阈值，边界题就被错误作答了。
        # 两处任一命中都算排除，宁可少答。
        exclusion_hit = any(
            is_exclusion_text(self.clauses[i].text) for i in cand_idx[: self.EXCLUSION_PROBE_K]
        )
        # 顺序不能反：一条里同时出现「本导则适用于」与「不适用于」时，
        # 它既是元信息条款又是排除声明，先清理就丢掉拒答信号了。
        scope_excluded = exclusion_hit or (bool(scored) and is_exclusion_text(scored[0].clause.text))
        # 适用范围 / 术语定义 / 编制目的 只是自我介绍，不能当处置依据排在前面。
        # 只在它已经排到第一时才清理，避免无谓改动正常排序；清理后一条不剩则退回原排序。
        if scored and is_scope_statement(scored[0].clause.text):
            usable = [s for s in scored if not is_scope_statement(s.clause.text)]
            if usable:
                scored = usable
        top = scored[:k_final]

        top_score = float(top[0].rerank) if top else 0.0
        covered = (top_score >= self.min_relevance) and not scope_excluded
        hits = [
            SearchHit(
                rank=i + 1,
                clause=s.clause,
                bm25=s.bm25,
                vector=s.vector,
                rrf=s.rrf,
                rerank=s.rerank,
            )
            for i, s in enumerate(top)
        ]
        return SearchResponse(
            query=query,
            hits=hits,
            covered=covered,
            top_score=round(top_score, 6),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            scope_excluded=scope_excluded,
            search_query=search_query,
            matched_terms=matched_terms,
            scope_docs=sorted(allowed) if allowed is not None else [],
        )

    def search_scored(
        self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> list[ScoredClause]:
        """给编排层用：直接拿到带分数的条款对象。"""
        resp = self.search(query, top_k=top_k, doc_ids=doc_ids)
        return [
            ScoredClause(
                clause=h.clause,
                bm25=h.bm25,
                vector=h.vector,
                rrf=h.rrf,
                rerank=h.rerank,
                stages=["bm25", "vector", "rrf", "rerank"],
            )
            for h in resp.hits
        ]

    # ------------------------------------------------------------ 统计与持久化

    def stats(self) -> KnowledgeBaseStats:
        return KnowledgeBaseStats(
            chunks=len(self.clauses),
            docs=len({c.doc_id for c in self.clauses}),
            bm25_backend=self.bm25.backend_name,
            embedder=getattr(self.embedder, "name", type(self.embedder).__name__),
            reranker=getattr(self.reranker, "name", type(self.reranker).__name__),
            dim=int(getattr(self.embedder, "dim", 0)),
        )

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "clauses.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.clauses:
                fh.write(c.model_dump_json() + "\n")
        if self.vectors is not None and self.vectors.size:
            np.save(d / "vectors.npy", self.vectors)
        save_embedder_state(self.embedder, d / "embedder.json")
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "stats": self.stats().to_dict(),
                    "params": {
                        "bm25_top_k": self.bm25_top_k,
                        "vector_top_k": self.vector_top_k,
                        "fused_top_k": self.fused_top_k,
                        "final_top_k": self.final_top_k,
                        "rrf_k": self.rrf_k,
                        "min_relevance": self.min_relevance,
                    },
                    "degradations": self.degradations,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return d

    @classmethod
    def load(cls, directory: str | Path, cfg: Config | None = None) -> "KnowledgeBase":
        from ..parsing.builder import read_clauses_jsonl

        d = Path(directory)
        clauses = read_clauses_jsonl(d / "clauses.jsonl")
        state = load_embedder_state(d / "embedder.json")
        kb = cls.from_clauses(clauses, cfg, embedder_state=state)
        vectors_path = d / "vectors.npy"
        if vectors_path.exists():
            cached = np.load(vectors_path)
            if cached.shape == (len(clauses), int(getattr(kb.embedder, "dim", 0))):
                kb.vectors = cached
        return kb
