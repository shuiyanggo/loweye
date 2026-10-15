"""M6 评估：把「效果如何」变成能复现的数字。

一次评估跑四件事：
1. 检索消融 —— 同一批金标问题，分别只看 BM25 / 只看向量 / RRF 融合 / 融合+重排，
   比 Recall@1/@3/@5 与 MRR。目的：证明「加融合、加重排」不是装饰。
2. 拒答能力 —— 越界问题必须被拒答，否则系统会拿相似词面硬凑条款。
3. 时延 —— 检索与端到端分析的 P50/P95，给现场演示留够余量。
4. 三组对照 —— A 纯检测 / B 直答无知识库 / C 本系统，比建议可执行率、可溯源率、
   引用可核验率。

口径统一（否则数字没有意义）：
- 检索消融的四种配置都对「无条款号内容」做同样过滤，差异只来自召回与排序策略；
- 三组对照的分母完全一致：每个「(图片, 缺陷类别)」算一个样本，三组都在同一批样本上打分。

写死在报告里的两条边界（不允许删）：
- 演示图为合成图、条款为自编占位文档，因此本报告**不给出检测精度结论**；
- arm B 若为脚本化代理，报告里必须标注它不是真实模型实测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config
from ..detect.base import build_detector
from ..eval.baselines import ArmAnswer, arm_a_detection_only, arm_b_direct_answer
from ..eval.goldset import GoldSet, load_goldset
from ..eval.metrics import mrr, percentile, rate, recall_at_k
from ..llm.template import is_actionable
from ..retrieval.fusion import reciprocal_rank_fusion, top_indices
from ..retrieval.index import KnowledgeBase
from ..retrieval.lexicon import TermLexicon
from ..text import is_numbered_clause

RETRIEVAL_CONFIGS = [
    "bm25_only",
    "vector_only",
    "hybrid_rrf",
    "hybrid_rerank",
    "hybrid_rerank_no_lexicon",
]
CONFIG_LABELS = {
    "bm25_only": "仅 BM25",
    "vector_only": "仅向量",
    "hybrid_rrf": "RRF 融合（无重排）",
    "hybrid_rerank": "融合 + 重排（本系统）",
    "hybrid_rerank_no_lexicon": "融合 + 重排（关闭术语表）",
}
GROUP_LABELS = {
    "literal": "原词题（使用文档原词提问）",
    "paraphrase": "改写题（现场口吻，不含文档原词）",
}


def _ranks_summary(ranks: list[int | None]) -> dict:
    return {
        "recall@1": round(recall_at_k(ranks, 1), 4),
        "recall@3": round(recall_at_k(ranks, 3), 4),
        "recall@5": round(recall_at_k(ranks, 5), 4),
        "mrr": round(mrr(ranks), 4),
        "hit_rate@50": round(rate(sum(1 for r in ranks if r is not None), len(ranks)), 4),
    }


@dataclass
class ArmComparison:
    answers: dict[str, list[ArmAnswer]]
    actionable_rate: dict[str, float]
    traceable_rate: dict[str, float]
    citation_verified_rate: float
    citation_checked: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": {arm: len(v) for arm, v in self.answers.items()},
            "actionable_rate": {k: round(v, 4) for k, v in self.actionable_rate.items()},
            "traceable_rate": {k: round(v, 4) for k, v in self.traceable_rate.items()},
            "citation_verified_rate": round(self.citation_verified_rate, 4),
            "citation_checked": self.citation_checked,
            "notes": list(self.notes),
            "answers": {arm: [a.to_dict() for a in v] for arm, v in self.answers.items()},
        }


@dataclass
class EvalReport:
    created_at: str
    kb: dict
    retrieval: dict
    refusal: dict
    latency: dict
    arms: ArmComparison
    detection: dict
    goldset: dict
    retrieval_rows: list[dict] = field(default_factory=list)
    refusal_rows: list[dict] = field(default_factory=list)
    stage_latency: dict = field(default_factory=dict)
    degradation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "kb": self.kb,
            "retrieval": self.retrieval,
            "refusal": self.refusal,
            "latency": self.latency,
            "arms": self.arms.to_dict(),
            "detection": self.detection,
            "goldset": self.goldset,
            "retrieval_rows": self.retrieval_rows,
            "refusal_rows": self.refusal_rows,
            "stage_latency": self.stage_latency,
            "degradation_notes": self.degradation_notes,
        }


class Evaluator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.kb: KnowledgeBase | None = None
        self.goldset: GoldSet | None = None
        self._pipe = None

    # ------------------------------------------------------------- 准备

    def _load_kb(self) -> KnowledgeBase:
        if self.kb is None:
            index_dir = self.cfg.path("index_dir")
            self.kb = KnowledgeBase.load(index_dir, self.cfg)
        return self.kb

    def _load_goldset(self) -> GoldSet:
        if self.goldset is None:
            path = self.cfg.abs_path(self.cfg.get("eval.goldset", "eval/gold_qa.yaml"))
            self.goldset = load_goldset(path)
        return self.goldset

    def _pipeline(self):
        if self._pipe is None:
            from ..agent.pipeline import Pipeline

            self._pipe = Pipeline.from_config(self.cfg)
        return self._pipe

    # ------------------------------------------------------------- 检索消融

    def _rankings(self, query: str) -> dict[str, list[int]]:
        kb = self._load_kb()
        clauses = kb.clauses
        eligible = [i for i, c in enumerate(clauses) if is_numbered_clause(c.clause_no)] or list(range(len(clauses)))

        # 术语归一化 + 查询扩展属于「查询侧预处理」，所有配置共用同一个查询串，
        # 这样单路/融合/重排之间的差异只来自召回与排序策略本身；
        # 术语表的单独效果由 hybrid_rerank_no_lexicon 那一列隔离出来
        # （那一列会把 kb.lexicon 临时换成空表，因此这里必须动态读 kb.lexicon）。
        search_query = kb.lexicon.expand(query) or query

        bm25_raw = kb.bm25.scores(search_query)
        bm25_rank = top_indices(bm25_raw, kb.bm25_top_k)
        if kb.vectors is not None and kb.vectors.size:
            # 向量这一路原本误用了未扩展的 query，等于两路看的不是同一句话，
            # 消融对照就不成立了（对比的是「两路输入不同」，不是「融合有没有用」）。
            qvec = kb.embedder.encode([search_query])[0]
            vec_scores = np.asarray(kb.vectors @ qvec, dtype=np.float64).tolist()
        else:
            vec_scores = [0.0] * len(clauses)
        vec_rank = top_indices(vec_scores, kb.vector_top_k)
        fused = reciprocal_rank_fusion([bm25_rank, vec_rank], k=kb.rrf_k)

        def order_by(scores) -> list[int]:
            if isinstance(scores, dict):
                keyed = [(float(scores.get(i, 0.0)), i) for i in eligible]
            else:
                keyed = [(float(scores[i]), i) for i in eligible]
            keyed.sort(key=lambda kv: (-kv[0], kv[1]))
            return [i for _, i in keyed]

        # 融合+重排走主流程自己的排序（内部即 融合→候选过滤→重排），
        # top_k 给到全库长度以获得完整排序。
        by_chunk = {c.chunk_id: i for i, c in enumerate(clauses)}
        full_order = [by_chunk[h.clause.chunk_id] for h in kb.search(query, top_k=len(clauses)).hits]

        # 术语表消融：同一条链路、同一个阈值，只把查询归一化/扩展关掉，
        # 差异就只能来自这一层预处理。跑完必须还原，避免污染后续问题。
        saved_lexicon = kb.lexicon
        kb.set_lexicon(TermLexicon([]))
        try:
            plain_order = [by_chunk[h.clause.chunk_id] for h in kb.search(query, top_k=len(clauses)).hits]
        finally:
            kb.set_lexicon(saved_lexicon)

        return {
            "bm25_only": order_by(bm25_raw),
            "vector_only": order_by(vec_scores),
            "hybrid_rrf": order_by(fused),
            "hybrid_rerank": full_order,
            "hybrid_rerank_no_lexicon": plain_order,
        }

    def eval_retrieval(self) -> tuple[dict, list[dict]]:
        kb = self._load_kb()
        gold = self._load_goldset()
        ranks: dict[str, list[int | None]] = {c: [] for c in RETRIEVAL_CONFIGS}
        covered_flags: list[bool] = []
        rows: list[dict] = []

        for item in gold.in_scope:
            ranking = self._rankings(item.question)
            resp = kb.search(item.question)
            covered_flags.append(bool(resp.covered))
            row = {
                "id": item.id,
                "group": item.group,
                "question": item.question,
                "expected": f"{item.doc}:{'/'.join(item.clauses)}",
                "covered": bool(resp.covered),
                "scope_excluded": bool(resp.scope_excluded),
                "top1": f"{resp.hits[0].clause.doc_id}:{resp.hits[0].clause.clause_no}" if resp.hits else "",
                "top_score": round(float(resp.top_score), 4),
            }
            for cfg_name in RETRIEVAL_CONFIGS:
                hit_rank: int | None = None
                for pos, idx in enumerate(ranking[cfg_name][:50], start=1):
                    if item.matches(kb.clauses[idx].doc_id, kb.clauses[idx].clause_no):
                        hit_rank = pos
                        break
                ranks[cfg_name].append(hit_rank)
                row[cfg_name] = hit_rank
            rows.append(row)

        summary: dict = {}
        for cfg_name in RETRIEVAL_CONFIGS:
            summary[cfg_name] = _ranks_summary(ranks[cfg_name])
        summary["n_questions"] = len(gold.in_scope)
        summary["covered_rate"] = round(rate(sum(covered_flags), len(covered_flags)), 4)

        # 分组统计：原词题 vs 改写题。原词题对任何检索方式都太容易，
        # 分开报才能看出「只靠词面」与「词面+语义+重排」的差别。
        by_group: dict[str, dict] = {}
        for group in gold.groups:
            idxs = [i for i, item in enumerate(gold.in_scope) if item.group == group]
            if not idxs:
                continue
            g: dict = {"n_questions": len(idxs)}
            for cfg_name in RETRIEVAL_CONFIGS:
                g[cfg_name] = _ranks_summary([ranks[cfg_name][i] for i in idxs])
            g["covered_rate"] = round(rate(sum(1 for i in idxs if covered_flags[i]), len(idxs)), 4)
            by_group[group] = g
        summary["by_group"] = by_group
        return summary, rows

    # ------------------------------------------------------------- 拒答

    def eval_refusal(self) -> tuple[dict, list[dict]]:
        kb = self._load_kb()
        gold = self._load_goldset()
        rows = []
        for item in gold.out_of_scope:
            resp = kb.search(item.question)
            rows.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "why": item.why,
                    "covered": bool(resp.covered),
                    "top_score": round(float(resp.top_score), 4),
                    "n_hits": len(resp.hits),
                    "top1": f"{resp.hits[0].clause.doc_id}:{resp.hits[0].clause.clause_no}" if resp.hits else "",
                }
            )
        refused = sum(1 for r in rows if not r["covered"])
        summary = {
            "n_questions": len(rows),
            "refusal_rate": round(rate(refused, len(rows)), 4),
            "false_accept": [r["id"] for r in rows if r["covered"]],
            "min_relevance": round(float(kb.min_relevance), 4),
        }
        return summary, rows

    # ------------------------------------------------------------- 时延

    def eval_latency(self, *, search_repeat: int = 3, analyze_repeat: int = 2) -> tuple[dict, dict, list[str]]:
        kb = self._load_kb()
        gold = self._load_goldset()
        notes: list[str] = []
        search_ms: list[float] = []
        for _ in range(max(1, search_repeat)):
            for item in gold.in_scope:
                search_ms.append(float(kb.search(item.question).elapsed_ms))

        analyze_ms: list[float] = []
        stage_ms: dict[str, list[float]] = {}
        try:
            pipe = self._pipeline()
            images = sorted(self.cfg.path("assets").glob("*.png"))
            if not images:
                notes.append("data/assets 下没有演示图，端到端时延未测")
            for _ in range(max(1, analyze_repeat)):
                for img in images:
                    result = pipe.run(img)
                    analyze_ms.append(float(result.total_ms))
                    for timing in result.timings or []:
                        stage_ms.setdefault(timing.stage, []).append(float(timing.ms))
        except Exception as exc:  # noqa: BLE001 - 端到端拉不起来不应让整份评估失败
            notes.append(f"端到端时延未测：{type(exc).__name__}: {exc}")
            return {"search_ms": _lat(search_ms), "analyze_ms": None, "analyze_error": notes[-1]}, {}, notes

        return (
            {"search_ms": _lat(search_ms), "analyze_ms": _lat(analyze_ms) if analyze_ms else None},
            {stage: _lat(v) for stage, v in stage_ms.items()},
            notes,
        )

    # ------------------------------------------------------------- 三组对照

    def _samples(self) -> tuple[list[dict], list[ArmAnswer], int, int, list[str]]:
        """每个 (图片, 缺陷类别) 一个样本；三组 arm 共用这批样本作为分母。"""
        pipe = self._pipeline()
        samples: list[dict] = []
        arm_c: list[ArmAnswer] = []
        checked = verified = 0
        notes: list[str] = []

        images = sorted(self.cfg.path("assets").glob("*.png"))
        if not images:
            return [], [], 0, 0, ["data/assets 下没有演示图，三组对照无法进行"]

        for img in images:
            result = pipe.run(img)
            grouped: dict[str, dict] = {}
            for diag in result.diagnoses:
                entry = grouped.setdefault(
                    diag.cls,
                    {
                        "label": diag.label,
                        "n": 0,
                        "confidence": 0.0,
                        "severity": diag.severity,
                        "finding": diag.finding,
                        "generator": diag.generator,
                        "actions": [],
                        "citations": [],
                        "clause_nos": [],
                    },
                )
                entry["n"] += 1
                entry["confidence"] = max(entry["confidence"], float(diag.confidence))
                for action in diag.actions:
                    if action not in entry["actions"]:
                        entry["actions"].append(action)
                for citation in diag.citations:
                    entry["citations"].append(citation)
                    if citation.clause_no not in entry["clause_nos"]:
                        entry["clause_nos"].append(citation.clause_no)
            for cls, entry in sorted(grouped.items()):
                checked += len(entry["citations"])
                verified += sum(1 for c in entry["citations"] if _citation_verifies(c, pipe.kb))
                samples.append({"image_id": result.image_id, "cls": cls, **{k: entry[k] for k in ("label", "n", "confidence", "severity")}})
                arm_c.append(
                    ArmAnswer(
                        arm="C",
                        cls=cls,
                        label=entry["label"],
                        n_detections=entry["n"],
                        confidence=entry["confidence"],
                        severity=entry["severity"],
                        text=entry["finding"],
                        actions=list(entry["actions"]),
                        citations=list(entry["clause_nos"]),
                        source=str(entry["generator"]),
                    )
                )
        return samples, arm_c, checked, verified, notes

    def eval_arms(self, *, arm_b_mode: str = "scripted") -> tuple[ArmComparison, list[str]]:
        pipe = self._pipeline()
        samples, arm_c, checked, verified, notes = self._samples()
        grading_map = {
            s["cls"]: (s["label"], s["severity"]) for s in samples
        }
        arm_a = arm_a_detection_only(samples, grading=grading_map)
        arm_b, note_b = arm_b_direct_answer(samples, grading=grading_map, mode=arm_b_mode, cfg=self.cfg)

        answers = {"A": arm_a, "B": arm_b, "C": arm_c}
        comparison = ArmComparison(
            answers=answers,
            actionable_rate={arm: _actionable_rate(v) for arm, v in answers.items()},
            traceable_rate={arm: _traceable_rate(v) for arm, v in answers.items()},
            citation_verified_rate=rate(verified, checked),
            citation_checked=checked,
            notes=notes
            + [
                note_b,
                "三组使用完全相同的样本（每个「图片 × 缺陷类别」算一个），"
                "因此比率可直接横比。",
                "「建议可执行率」三组共用同一判据（loweye.llm.template.is_actionable），"
                "它只反映输出形态，不构成质量优势；差别在可溯源率与引用可核验率。",
                "arm A 结构上不产生处置建议；arm B 结构上没有引用来源，故两者的可溯源率天然为 0。",
                f"引用核验：核验 {checked} 条引用，其中 {verified} 条能在知识库条款原文中逐字找到。",
            ],
        )
        return comparison, notes

    # ------------------------------------------------------------- 检测自检

    def eval_detection_selfcheck(self) -> tuple[dict, list[str]]:
        """用检测器输出对比同一份 sidecar 标注 —— 自证循环，只证明脚本能算。"""
        from ..eval.metrics import DetBox, detection_report

        assets = self.cfg.path("assets")
        detector, why = build_detector(self.cfg)
        preds: list[DetBox] = []
        gts: list[DetBox] = []
        for sidecar in sorted(assets.glob("*.detections.json")):
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            for d in data.get("detections", []):
                gts.append(_det_box(d))
            image_path = sidecar.with_name(sidecar.name.replace(".detections.json", ".png"))
            if not image_path.exists():
                continue
            res = detector.detect(image_path, image_path.stem)
            for d in res.detections:
                preds.append(
                    DetBox(
                        cls=d.cls,
                        x1=float(d.bbox.x1),
                        y1=float(d.bbox.y1),
                        x2=float(d.bbox.x2),
                        y2=float(d.bbox.y2),
                        score=float(d.score),
                    )
                )
        report = detection_report(preds, gts, iou_threshold=0.5)
        report["detector"] = getattr(detector, "name", type(detector).__name__)
        report["detector_reason"] = why
        report["n_pred"] = len(preds)
        report["n_gt"] = len(gts)
        report["conclusion"] = (
            "自证循环：预测来自 sidecar 文件，真值也是同一份 sidecar，因此 mAP 必然接近 1。"
            "本项只验证评测脚本可运行，禁止作为模型精度结论引用。"
        )
        return report, []

    # ------------------------------------------------------------- 主入口

    def run(self, *, arm_b_mode: str = "scripted", search_repeat: int = 3, analyze_repeat: int = 2) -> EvalReport:
        kb = self._load_kb()
        gold = self._load_goldset()
        retrieval, retrieval_rows = self.eval_retrieval()
        refusal, refusal_rows = self.eval_refusal()
        latency, stage_latency, latency_notes = self.eval_latency(
            search_repeat=search_repeat, analyze_repeat=analyze_repeat
        )
        arms, arm_notes = self.eval_arms(arm_b_mode=arm_b_mode)
        detection, det_notes = self.eval_detection_selfcheck()
        return EvalReport(
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            kb=kb.stats().to_dict(),
            retrieval=retrieval,
            refusal=refusal,
            latency=latency,
            arms=arms,
            detection=detection,
            goldset=gold.to_dict(),
            retrieval_rows=retrieval_rows,
            refusal_rows=refusal_rows,
            stage_latency=stage_latency,
            degradation_notes=latency_notes + arm_notes + det_notes,
        )


# ------------------------------------------------------------------ 工具


def _det_box(d: dict):
    """把 sidecar 标注行转成评估框。bbox 兼容 [x1,y1,x2,y2] 与 {x1,y1,x2,y2} 两种写法。"""
    from ..eval.metrics import DetBox

    raw = d.get("bbox") or [0, 0, 0, 0]
    if isinstance(raw, dict):
        coords = [raw.get("x1", 0), raw.get("y1", 0), raw.get("x2", 0), raw.get("y2", 0)]
    else:
        coords = list(raw)[:4]
    return DetBox(
        cls=str(d.get("cls")),
        x1=float(coords[0]),
        y1=float(coords[1]),
        x2=float(coords[2]),
        y2=float(coords[3]),
        score=float(d.get("score", 1.0)),
    )


def _lat(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "max": round(max(values), 2) if values else 0.0,
        "mean": round(sum(values) / len(values), 2) if values else 0.0,
    }


def _citation_verifies(citation, kb: KnowledgeBase) -> bool:
    """引用核验：条款必须存在，且引用片段能在该条款原文里逐字找到。"""
    target = next((c for c in kb.clauses if c.chunk_id == citation.chunk_id), None)
    if target is None:
        return False
    quote = (citation.quote or "").strip()
    if not quote:
        return False
    trimmed = quote.rstrip("…").rstrip("，,、；;")
    return bool(trimmed) and trimmed in target.text


def _actionable_rate(answers: list[ArmAnswer]) -> float:
    if not answers:
        return 0.0
    ok = sum(1 for a in answers if any(is_actionable(x) for x in a.actions))
    return ok / len(answers)


def _traceable_rate(answers: list[ArmAnswer]) -> float:
    """每条处置动作都带条款号引用才算「可溯源」。"""
    if not answers:
        return 0.0
    ok = 0
    for a in answers:
        if a.actions and a.citations and all("〔" in x for x in a.actions):
            ok += 1
    return ok / len(answers)


# ------------------------------------------------------------------ 报告渲染


def render_markdown(report: EvalReport) -> str:
    r = report
    ret = r.retrieval
    a = r.arms
    lines: list[str] = []
    add = lines.append

    add("# 低空之眼 · M6 评估报告（自动生成）\n")
    add(f"生成时间：{r.created_at}  ")
    add(
        f"知识库：{r.kb['chunks']} 条款块 / {r.kb['docs']} 文档；BM25={r.kb['bm25_backend']}、"
        f"向量={r.kb['embedder']}(dim={r.kb['dim']})、重排={r.kb['reranker']}\n"
    )

    best = max(RETRIEVAL_CONFIGS, key=lambda c: ret[c]["mrr"])
    add("## 0 结论摘要\n")
    add(
        f"- 检索：{ret['n_questions']} 道金标题上，本系统（融合+重排）Recall@1 "
        f"{ret['hybrid_rerank']['recall@1']:.0%}、Recall@3 {ret['hybrid_rerank']['recall@3']:.0%}、"
        f"Recall@5 {ret['hybrid_rerank']['recall@5']:.0%}、MRR {ret['hybrid_rerank']['mrr']:.3f}；"
        f"五组配置中 MRR 最高者为「{CONFIG_LABELS[best]}」。"
    )
    para = (ret.get("by_group") or {}).get("paraphrase")
    if para:
        add(
            f"- 改写题（不含文档原词，{para['n_questions']} 题，**开发集**）：本系统 Recall@1 "
            f"{para['hybrid_rerank']['recall@1']:.0%} / MRR {para['hybrid_rerank']['mrr']:.3f}；"
            f"仅 BM25 {para['bm25_only']['recall@1']:.0%} / {para['bm25_only']['mrr']:.3f}；"
            f"仅向量 {para['vector_only']['recall@1']:.0%} / {para['vector_only']['mrr']:.3f}；"
            f"关闭术语表 {para['hybrid_rerank_no_lexicon']['recall@1']:.0%} / "
            f"{para['hybrid_rerank_no_lexicon']['mrr']:.3f}。"
        )
        best_single = max(para["bm25_only"]["mrr"], para["vector_only"]["mrr"])
        if para["hybrid_rerank"]["mrr"] <= best_single + 1e-9:
            add(
                "  - 如实说明：改写题上本系统没有跑赢最强的单路配置。当前向量后端是"
                "哈希 TF-IDF（离线兜底），并不具备真正的语义泛化能力；"
                "换成 bge-m3 + 重排模型后才谈得上语义优势。这一项属于待补强的工程缺口。"
            )
    held = (ret.get("by_group") or {}).get("heldout")
    if held:
        add(
            f"- 留出集（术语表定稿后才写的改写题，{held['n_questions']} 题）：本系统 Recall@1 "
            f"{held['hybrid_rerank']['recall@1']:.0%} / MRR {held['hybrid_rerank']['mrr']:.3f}。"
            "**这一组才是对「没见过的说法」的真实表现**，它明显低于开发集，"
            "说明术语表路线（人工枚举口语别名）在没枚举到的说法上会失效 —— "
            "结论是必须换语义向量 + cross-encoder 重排，而不是继续往词表里堆别名。"
        )
    add(
        f"- 拒答：{r.refusal['n_questions']} 道越界问题拒答率 {r.refusal['refusal_rate']:.0%}"
        f"（覆盖阈值 min_relevance={r.refusal['min_relevance']}）。"
    )
    if r.latency.get("analyze_ms"):
        add(
            f"- 时延：检索 P50 {r.latency['search_ms']['p50']} ms / P95 {r.latency['search_ms']['p95']} ms；"
            f"端到端分析 P50 {r.latency['analyze_ms']['p50']} ms / P95 {r.latency['analyze_ms']['p95']} ms。"
        )
    add(
        f"- 三组对照：可溯源率 A {a.traceable_rate['A']:.0%} / B {a.traceable_rate['B']:.0%} / "
        f"C {a.traceable_rate['C']:.0%}；本系统引用可核验率 {a.citation_verified_rate:.0%}"
        f"（核验 {a.citation_checked} 条）。"
    )
    add("")

    add("## 1 边界声明（不可省略）\n")
    add("- 演示图是程序化合成图，条款是项目自编占位文档（不是任何标准原文），")
    add("  所以本报告**不给出检测精度结论**；真实精度要在实拍标注集上重测。")
    add("- 金标问答集为项目自编，度量的是本系统检索链路的行为，不是行业权威准确率。")
    for note in a.notes:
        add(f"- {note}")
    add("")

    add("## 2 检索消融（同一批金标问题）\n")
    add("| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR | 前 50 命中率 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for cfg_name in RETRIEVAL_CONFIGS:
        s = ret[cfg_name]
        add(
            f"| {CONFIG_LABELS[cfg_name]} | {s['recall@1']:.0%} | {s['recall@3']:.0%} | "
            f"{s['recall@5']:.0%} | {s['mrr']:.3f} | {s['hit_rate@50']:.0%} |"
        )
    add(f"\n域内题「系统认为有依据」比例：{ret['covered_rate']:.0%}\n")
    groups = ret.get("by_group") or {}
    if groups:
        add("### 2.1 分组：原词题是开发/对照，留出集才是泛化能力\n")
        add("| 分组 | n | " + " | ".join(CONFIG_LABELS[c] for c in RETRIEVAL_CONFIGS) + " |")
        add("| --- | --- | " + " | ".join("---" for _ in RETRIEVAL_CONFIGS) + " |")
        for group in ("literal", "paraphrase", "heldout"):
            g = groups.get(group)
            if not g:
                continue
            cells = [f"{g[c]['recall@1']:.0%} / {g[c]['mrr']:.3f}" for c in RETRIEVAL_CONFIGS]
            add(f"| {group} | {g['n_questions']} | " + " | ".join(cells) + " |")
        add(
            "\n每格是「Recall@1 / MRR」。三种题的区别必须说清楚："
            "**literal** 是文档原词提问（对任何检索方式都偏容易）；"
            "**paraphrase** 是开发集，术语表就是照着它调的；"
            "**heldout** 是术语表定稿之后才写的，没参与任何调参 —— "
            "留出集与开发集的差距，就是这套词法路线距离「听懂人话」还差多少。\n"
        )

    add("## 3 拒答能力（越界问题）\n")
    add("| 编号 | 问题 | 最高分 | 判定 | 最近条款 |")
    add("| --- | --- | --- | --- | --- |")
    for row in r.refusal_rows:
        verdict = "已拒答" if not row["covered"] else "**漏判（未拒答）**"
        add(f"| {row['id']} | {row['question']} | {row['top_score']} | {verdict} | {row['top1']} |")
    add("")

    add("## 4 时延\n")
    s = r.latency["search_ms"]
    add(f"- 检索（n={s['n']}）：P50 {s['p50']} ms / P95 {s['p95']} ms / 最大 {s['max']} ms")
    if r.latency.get("analyze_ms"):
        a2 = r.latency["analyze_ms"]
        add(f"- 端到端分析（n={a2['n']}）：P50 {a2['p50']} ms / P95 {a2['p95']} ms / 最大 {a2['max']} ms")
        if r.stage_latency:
            add("- 分阶段 P95：")
            for stage, st in r.stage_latency.items():
                add(f"  - {stage}: {st['p95']} ms")
    else:
        add(f"- 端到端分析未跑通：{r.latency.get('analyze_error')}")
    add("")

    add("## 5 三组对照实验\n")
    add("| 指标 | A 纯检测 | B 直答（无知识库） | C 本系统 |")
    add("| --- | --- | --- | --- |")
    add(f"| 样本数 | {len(a.answers['A'])} | {len(a.answers['B'])} | {len(a.answers['C'])} |")
    add(
        f"| 给出处置动作的比例 | {a.actionable_rate['A']:.0%} | {a.actionable_rate['B']:.0%} | "
        f"{a.actionable_rate['C']:.0%} |"
    )
    add(
        f"| 可溯源率（动作带条款号） | {a.traceable_rate['A']:.0%} | {a.traceable_rate['B']:.0%} | "
        f"{a.traceable_rate['C']:.0%} |"
    )
    add(f"\n本系统引用可核验率：{a.citation_verified_rate:.0%}（引用片段能在条款原文中逐字找到）\n")

    add("### 5.1 同一缺陷的三组输出对比（抽样）\n")
    for ans_c in a.answers["C"][:4]:
        add(f"**{ans_c.label}（{ans_c.cls}）**\n")
        for arm in ("A", "B", "C"):
            match = next((x for x in a.answers[arm] if x.cls == ans_c.cls), None)
            if not match:
                add(f"- {arm}：无对应样本")
                continue
            acts = "；".join(match.actions) if match.actions else "（未给出处置动作）"
            add(f"- {arm}（{match.source}）：{acts}")
        add("")

    add("## 6 检测精度：本报告为什么不给 mAP\n")
    d = r.detection
    add("- 原因：演示图由脚本合成、标注由同一脚本写出，属于自证循环。")
    add(
        f"- 自检数字（mAP={d['mAP']}，评测 {d['classes_evaluated']} 类，"
        f"预测 {d.get('n_pred')} 框 / 真值 {d.get('n_gt')} 框）只能证明评测脚本可运行。"
    )
    add(f"- 结论：{d['conclusion']}")
    add("- 真实精度路径：用实拍数据（电力 CPLID / InsPLAD、农业 PlantDoc 等）训练权重后，")
    add("  在同一批实拍标注上重跑评估，并把检测后端从 sidecar 切到 onnx。")
    add("")

    add("## 附录 A 金标问题逐题命中排名\n")
    add("| 编号 | 问题 | 期望条款 | 仅BM25 | 仅向量 | RRF | 融合+重排 | covered |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in r.retrieval_rows:
        vals = " | ".join("-" if row[c] is None else str(row[c]) for c in RETRIEVAL_CONFIGS)
        add(
            f"| {row['id']} | {row['question']} | {row['expected']} | {vals} | "
            f"{'是' if row['covered'] else '否'} |"
        )
    add("\n（表中数字为命中排名，1 表示排第一，`-` 表示前 50 名内未命中。）")
    return "\n".join(lines)
