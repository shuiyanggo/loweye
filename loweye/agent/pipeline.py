"""编排状态机：识别 → 诊断 → 分级 → 派单 → 归档。

显式状态机而不是 Agent 自由 ReAct 循环，理由（方案 v2 第 4 章 M3）：
巡检业务要求同一张图、同一份知识库，任何时候跑出来的结论和派单必须一致。
自由循环会带来不可复现的分支，也会让「为什么派给这个班组」无法逐条解释。
每一步的产物都进 PipelineResult，前端、报告、评测脚本读的是同一份结构。
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from ..config import Config, load_config, load_routes
from ..detect import build_detector
from ..llm import build_generator
from ..parsing.builder import build_clauses
from ..retrieval import KnowledgeBase
from ..schemas import (
    Citation,
    Degradation,
    Diagnosis,
    PipelineResult,
    Timing,
    WorkOrder,
)
from .grading import GradingTable
from .tools import ToolBox

SCENE_BY_PREFIX = {"crop_": "agriculture", "fire_": "emergency"}


def _scene_for(cls: str) -> str:
    for prefix, scene in SCENE_BY_PREFIX.items():
        if cls.startswith(prefix):
            return scene
    return "power"


def _scene_docs(cfg, scene: str) -> list[str] | None:
    """场景 → 允许回答的文档（config/app.yaml 的 retrieval.scene_docs）。

    为什么要有这道范围：各专业的规程各管一段，输电缺陷的工单引用
    《植保作业指引》里的通用报告条款，读起来荒谬，执行时也无人认账。
    配置留空时不做限制（知识库页的自由检索就该是全库）。
    """
    raw = cfg.get(f"retrieval.scene_docs.{scene}")
    if not raw:
        return None
    if isinstance(raw, str):
        return [raw]
    return [str(d) for d in raw]


def _quote(text: str, limit: int = 120) -> str:
    t = text.strip().replace("\n", " ")
    return t if len(t) <= limit else t[:limit].rstrip("，,、；;") + "…"


class TraceStore:
    """内存里保留最近 N 条分析结果，供前端列表与详情页使用。"""

    def __init__(self, limit: int = 200) -> None:
        self.limit = limit
        self._items: deque[PipelineResult] = deque(maxlen=limit)

    def add(self, result: PipelineResult) -> None:
        self._items.appendleft(result)

    def list(self) -> list[PipelineResult]:
        return list(self._items)

    def get(self, trace_id: str) -> PipelineResult | None:
        for item in self._items:
            if item.trace_id == trace_id:
                return item
        return None

    def __len__(self) -> int:
        return len(self._items)


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        *,
        kb: KnowledgeBase,
        detector,
        generator,
        toolbox: ToolBox,
        kb_degradations: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.kb = kb
        self.detector = detector
        self.generator = generator
        self.toolbox = toolbox
        self.grader = GradingTable(load_routes(cfg.abs_path(cfg.require("severity.rules_file"))))
        self.kb_degradations = list(kb_degradations or [])
        self.traces = TraceStore(int(cfg.get("app.trace_retention", 200)))

    # ------------------------------------------------------------ 构建

    @classmethod
    def from_config(cls, cfg: Config | None = None, *, build_kb: bool = True) -> "Pipeline":
        cfg = cfg or load_config()
        cfg.ensure_dirs()

        kb_degradations: list[str] = []
        index_dir = cfg.path("index_dir")
        if build_kb and (index_dir / "clauses.jsonl").exists():
            kb = KnowledgeBase.load(index_dir, cfg)
            kb_degradations = list(kb.degradations)
        else:
            clauses, report = build_clauses(cfg)
            kb = KnowledgeBase.from_clauses(clauses, cfg)
            kb_degradations = list(kb.degradations)
            if build_kb:
                kb.save(index_dir)
            if not clauses:
                kb_degradations.append("知识库为空：请先运行 `python -m loweye.cli.build_kb` 或往 data/raw 放入文档")

        detector, det_reason = build_detector(cfg)
        if det_reason:
            kb_degradations.append(det_reason)
        generator, gen_reason = build_generator(cfg)
        if gen_reason:
            kb_degradations.append(gen_reason)

        toolbox = ToolBox(cfg, load_routes(cfg.abs_path(cfg.require("severity.rules_file"))))
        return cls(cfg, kb=kb, detector=detector, generator=generator, toolbox=toolbox, kb_degradations=kb_degradations)

    # ------------------------------------------------------------ 主流程

    def run(
        self,
        image_path: str | Path,
        *,
        image_id: str | None = None,
        image_url: str | None = None,
    ) -> PipelineResult:
        image_path = Path(image_path)
        image_id = image_id or image_path.stem
        trace_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        timings: list[Timing] = []
        degradations: list[Degradation] = [
            Degradation(stage="bootstrap", reason=reason, fallback="系统按降级后的后端继续运行")
            for reason in self.kb_degradations
        ]

        # ---- 状态 1：识别
        t0 = time.perf_counter()
        detect = self.detector.detect(image_path, image_id)
        slice_count = self._slice_count(image_path, detect.warnings)
        detect = detect.model_copy(update={"slice_count": slice_count})
        timings.append(Timing(stage="detect", ms=round((time.perf_counter() - t0) * 1000, 2)))
        for w in detect.warnings:
            degradations.append(Degradation(stage="detect", reason=w, fallback="本次按空检测结果继续"))

        # ---- 状态 2：诊断（检索 + 生成）
        t1 = time.perf_counter()
        diagnoses: list[Diagnosis] = []
        retrievals: dict[str, dict] = {}
        class_counts: dict[str, int] = {}
        for det in detect.detections:
            class_counts[det.cls] = class_counts.get(det.cls, 0) + 1
        for idx, det in enumerate(detect.detections):
            grading_now = self.grader.grade(
                det.cls, confidence=det.score, grounded=True,
                human_review_threshold=float(self.cfg.get("detect.human_review_threshold", 0.6)),
            )
            if grading_now.normal:
                # 正常类（例：未见缺陷的绝缘子）不检索条款、不给处置建议、不出工单。
                # 理由是检索必然召回「缺陷处置」类条款，套到正常设备上就是伪依据。
                diagnoses.append(
                    Diagnosis(
                        detection_index=idx,
                        cls=det.cls,
                        label=grading_now.label,
                        finding=f"检出{grading_now.label}，未发现缺陷特征，无需处置（置信度 {det.score:.2f}）。",
                        severity=grading_now.severity,
                        actions=[],
                        citations=[],
                        confidence=det.score,
                        grounded=True,
                        generator="rule",
                    )
                )
                continue

            scene = _scene_for(det.cls)
            query = self.grader.build_query(det.cls)
            resp = self.kb.search(query, doc_ids=_scene_docs(self.cfg, scene))
            retrievals[det.cls] = {
                "query": query,
                "covered": resp.covered,
                "top_score": resp.top_score,
                "elapsed_ms": resp.elapsed_ms,
                "scope_docs": resp.scope_docs,
                "hits": [
                    {
                        "rank": h.rank,
                        "chunk_id": h.clause.chunk_id,
                        "clause_no": h.clause.clause_no,
                        "doc_title": h.clause.doc_title,
                        "hierarchy_path": h.clause.hierarchy_path,
                        "rerank": h.rerank,
                        "rrf": h.rrf,
                        "bm25": h.bm25,
                        "vector": h.vector,
                    }
                    for h in resp.hits
                ],
            }
            ctx = self._context(det, resp, image_id, idx, class_counts[det.cls])
            generated = self.generator.generate(ctx)
            if generated.degraded_reason:
                degradations.append(
                    Degradation(stage="generate", reason=generated.degraded_reason, fallback=f"使用 {generated.generator} 生成")
                )

            lookup = {h.clause.chunk_id: h for h in resp.hits}

            def _citation(cid: str, role: str) -> Citation:
                hit = lookup[cid]
                return Citation(
                    chunk_id=cid,
                    clause_no=hit.clause.clause_no,
                    doc_title=hit.clause.doc_title,
                    quote=_quote(hit.clause.text),
                    score=hit.rerank,
                    role=role,
                )

            citations = [_citation(cid, "basis") for cid in generated.citation_ids if cid in lookup]
            # 参考条款：检索到、过阈值，但没被抽成处置动作（判定句、响应分级等）。
            # 依据为空时它仍留在 citations 里，但 grounded 依旧为假 —— 参考不等于依据。
            citations += [
                _citation(cid, "reference")
                for cid in generated.reference_ids
                if cid in lookup and cid not in generated.citation_ids
            ]
            grounded = any(c.role == "basis" for c in citations) and resp.covered
            if resp.hits and not resp.covered:
                top_clause = resp.hits[0].clause
                if resp.scope_excluded:
                    reason = (
                        f"{det.cls} 的最相似条款是《{top_clause.doc_title}》{top_clause.clause_no}，"
                        "该条为范围排除声明（不适用/不涵盖），判定为超出知识库覆盖范围"
                    )
                else:
                    reason = (
                        f"{det.cls} 检索最高分 {resp.top_score:.3f} 低于拒答阈值 {self.kb.min_relevance}"
                    )
                degradations.append(
                    Degradation(
                        stage="retrieve",
                        reason=reason,
                        fallback="判定知识库未覆盖，不给出处置建议",
                    )
                )

            label = self.grader.label_for(det.cls)
            diagnoses.append(
                Diagnosis(
                    detection_index=idx,
                    cls=det.cls,
                    label=label,
                    finding=generated.finding,
                    severity=self.grader.grade(
                        det.cls, confidence=det.score, grounded=grounded,
                        human_review_threshold=float(self.cfg.get("detect.human_review_threshold", 0.6)),
                    ).severity,
                    actions=generated.actions,
                    citations=citations,
                    confidence=det.score,
                    grounded=grounded,
                    generator=generated.generator,
                )
            )
        timings.append(Timing(stage="diagnose", ms=round((time.perf_counter() - t1) * 1000, 2)))

        # ---- 状态 3+4：分级与派单
        t2 = time.perf_counter()
        orders: list[WorkOrder] = []
        for diag in diagnoses:
            grading = self.grader.grade(
                diag.cls,
                confidence=diag.confidence,
                grounded=diag.grounded,
                human_review_threshold=float(self.cfg.get("detect.human_review_threshold", 0.6)),
            )
            if grading.normal:
                continue  # 正常类不出工单（见 diagnose 阶段同一判断）
            route = "human_review" if grading.needs_human_review else "auto"
            order = WorkOrder(
                order_id=self.toolbox.next_order_id(),
                trace_id=trace_id,
                severity=grading.severity,
                route=route,
                review_reasons=grading.reasons,
                assignee_role=grading.assignee_role,
                sla_hours=grading.sla_hours,
                title=f"{grading.label}（{diag.confidence:.2f}）",
                actions=diag.actions,
                citations=diag.citations,
                created_at=datetime.now().isoformat(timespec="seconds"),
                status="pending_review" if route == "human_review" else "dispatched",
            )
            res = self.toolbox.create_work_order(order)
            self.toolbox.record("workorder", res)
            if res.degraded:
                degradations.append(Degradation(stage="dispatch", reason=res.detail, fallback="工单仅在内存中保留"))
                order = order.model_copy(update={"status": "created"})
            if route == "auto":
                nres = self.toolbox.notify(order)
                self.toolbox.record("notify", nres)
                order = order.model_copy(update={"notify_status": nres.detail})
                if nres.degraded:
                    degradations.append(Degradation(stage="notify", reason=nres.detail, fallback="通知未送达，工单仍已落库"))
            orders.append(order)
        timings.append(Timing(stage="grade_dispatch", ms=round((time.perf_counter() - t2) * 1000, 2)))

        # ---- 状态 5：归档
        t3 = time.perf_counter()
        scene = _scene_for(diagnoses[0].cls) if diagnoses else "power"
        result = PipelineResult(
            trace_id=trace_id,
            image_id=image_id,
            image_url=image_url or f"/assets/{image_path.name}",
            scene=scene,
            detect=detect,
            diagnoses=diagnoses,
            work_orders=orders,
            degradations=degradations,
            retrievals=retrievals,
            timings=timings + [Timing(stage="archive", ms=round((time.perf_counter() - t3) * 1000, 2))],
            total_ms=0.0,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        result = result.model_copy(
            update={"total_ms": round((time.perf_counter() - t0) * 1000, 2)}
        )
        ares = self.toolbox.archive(result)
        self.toolbox.record("archive", ares)
        self.traces.add(result)
        return result

    # ------------------------------------------------------------ 辅助

    def _context(self, det, resp, image_id: str, idx: int, n_same_class: int):
        from ..llm import DiagnosisContext
        from ..schemas import ScoredClause

        clauses = [
            ScoredClause(clause=h.clause, bm25=h.bm25, vector=h.vector, rrf=h.rrf, rerank=h.rerank)
            for h in resp.hits
            # 只有过得了覆盖阈值的条款才有资格支撑处置动作：
            # 0.17 分的弱相关段落不该变成一条要人执行的工单动作。
            if h.rerank >= self.kb.min_relevance
        ]
        return DiagnosisContext(
            cls=det.cls,
            label=self.grader.label_for(det.cls),
            severity=self.grader.grade(
                det.cls, confidence=det.score, grounded=resp.covered,
                human_review_threshold=float(self.cfg.get("detect.human_review_threshold", 0.6)),
            ).severity,
            n_detections=n_same_class,
            confidence=det.score,
            clauses=clauses,
            scene=_scene_for(det.cls),
            image_id=image_id,
            keywords=self.grader.keywords_for(det.cls),
        )

    def _slice_count(self, image_path: Path, warnings: list[str]) -> int:
        """记录按当前切片参数会产生多少块瓦片（切片参数是配置的一部分，必须可见）。"""
        if not self.cfg.get("detect.slice.enabled", True):
            return 0
        try:
            from PIL import Image  # type: ignore

            with Image.open(image_path) as im:
                w, h = im.size
        except Exception as exc:
            warnings.append(
                f"未能读取图片尺寸（{type(exc).__name__}），切片瓦片数未记录；"
                "安装 Pillow 后此项自动生效"
            )
            return 0

        from ..detect.slicing import slice_image

        return len(
            slice_image(w, h, int(self.cfg.get("detect.slice.size", 640)), float(self.cfg.get("detect.slice.overlap", 0.2)))
        )
