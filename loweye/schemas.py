"""结构化契约。

全链路（检测 → 检索 → 诊断 → 派单 → 归档）的每一步输入输出都在这里定义。
（方案 v2 第 4 章 M3：每一步都要有明确的输入输出 schema，失败要能定位到步。）
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["P1", "P2", "P3", "P4"]
Route = Literal["auto", "human_review"]
OrderStatus = Literal["created", "pending_review", "dispatched", "closed"]


class BBox(BaseModel):
    """像素坐标框，左上角为原点。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @field_validator("x2")
    @classmethod
    def _check_x(cls, v: float, info) -> float:
        x1 = info.data.get("x1")
        if x1 is not None and v < x1:
            raise ValueError("bbox 要求 x2 >= x1")
        return v

    @field_validator("y2")
    @classmethod
    def _check_y(cls, v: float, info) -> float:
        y1 = info.data.get("y1")
        if y1 is not None and v < y1:
            raise ValueError("bbox 要求 y2 >= y1")
        return v


class Detection(BaseModel):
    cls: str
    bbox: BBox
    score: float = Field(ge=0.0, le=1.0)
    crop_url: str | None = None
    slice_id: str | None = None  # SAHI 切片来源，便于回溯小目标漏检


class DetectResult(BaseModel):
    image_id: str
    detections: list[Detection] = Field(default_factory=list)
    detector: str
    model_version: str
    elapsed_ms: float = 0.0
    slice_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class Clause(BaseModel):
    """知识库最小单元：一条可独立引用的规程条款。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    clause_no: str
    hierarchy_path: str
    page: int | None = None
    text: str
    source_version: str = ""
    source_date: str = ""

    @property
    def citation_label(self) -> str:
        return f"{self.doc_title} {self.clause_no}"


class ScoredClause(BaseModel):
    """带各路分数与所处阶段的条款，用于把检索过程摊开给人看（可解释）。"""

    clause: Clause
    bm25: float = 0.0
    vector: float = 0.0
    rrf: float = 0.0
    rerank: float = 0.0
    stages: list[str] = Field(default_factory=list)

    @property
    def final_score(self) -> float:
        return self.rerank if self.rerank else self.rrf


class Citation(BaseModel):
    chunk_id: str
    clause_no: str
    doc_title: str
    quote: str
    score: float
    # basis = 动作/结论的直接依据；reference = 检索到但未抽成动作的条款
    # （判定句、响应分级这类）。前端必须把两者分开显示，参考不等于依据。
    role: str = "basis"


class Diagnosis(BaseModel):
    detection_index: int
    cls: str
    label: str
    finding: str
    severity: Severity
    actions: list[str]
    citations: list[Citation]
    confidence: float
    grounded: bool  # 是否有知识库依据；False 时前端必须显式标注「无依据」
    generator: str


class WorkOrder(BaseModel):
    order_id: str
    trace_id: str
    severity: Severity
    route: Route
    review_reasons: list[str] = Field(default_factory=list)
    assignee_role: str
    sla_hours: int
    title: str
    actions: list[str]
    citations: list[Citation]
    created_at: str
    notify_status: str = "skipped"  # sent / failed / skipped / disabled
    status: OrderStatus = "created"


class Degradation(BaseModel):
    """降级链记录：哪一步、为什么、退到了什么实现。答辩时可直接展示这张表。"""

    stage: str
    reason: str
    fallback: str


class Timing(BaseModel):
    stage: str
    ms: float


class RetrievalTrace(BaseModel):
    """一次检索的完整留痕：查了什么、命中什么、够不够判定覆盖。

    前端「可溯源」面板与 M6 评测脚本读的都是这份结构，因此每条命中都保留
    BM25 / 向量 / RRF / 重排四路分数，便于解释「为什么是它排第一」。
    """

    query: str
    covered: bool
    top_score: float = 0.0
    elapsed_ms: float = 0.0
    hits: list[dict] = Field(default_factory=list)
    # 本次检索限定在哪些文档内（场景范围过滤）；空表示全库。
    scope_docs: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    trace_id: str
    image_id: str
    image_url: str | None = None
    scene: str = "power"
    detect: DetectResult
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    work_orders: list[WorkOrder] = Field(default_factory=list)
    degradations: list[Degradation] = Field(default_factory=list)
    timings: list[Timing] = Field(default_factory=list)
    retrievals: dict[str, RetrievalTrace] = Field(default_factory=dict)
    total_ms: float = 0.0
    created_at: str = ""


class SearchHit(BaseModel):
    rank: int
    clause: Clause
    bm25: float = 0.0
    vector: float = 0.0
    rrf: float = 0.0
    rerank: float = 0.0


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    covered: bool  # 是否命中知识库；False 表示「知识库未覆盖」，系统应当拒答
    top_score: float
    elapsed_ms: float
    # 命中「范围排除性条款」（如「本导则不适用于带电作业」）时为 True。
    # 与单纯的分数不够要分开：前者是知识库明确声明不覆盖该范围，
    # 后者只是相关度低。两者的提示文案不同，所以必须区分。
    scope_excluded: bool = False
    # 术语归一化 + 查询扩展之后真正送进检索的查询串，以及命中的规范术语。
    # 回传出来是为了让「系统把这句话听成了什么」可见，而不是黑箱。
    search_query: str = ""
    matched_terms: list[str] = Field(default_factory=list)
    # 检索被限定在哪些文档内（空 = 全库）。场景化问答会限定，
    # 避免把《植保作业指引》的通用条款引到输电缺陷的工单里。
    scope_docs: list[str] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    """人工复核回路（方案 v2 架构里的人工复核 + M1 的 route 字段）。

    order_id 以路径参数为准；这里保留一个可选的同名字段只是为了兼容
    直接把整个表单 POST 过来的客户端，服务端不读它。
    """

    order_id: str = ""
    decision: Literal["approve", "reject", "amend"]
    reviewer: str = "anonymous"
    note: str = ""
    amended_actions: list[str] | None = None


class ReviewResponse(BaseModel):
    order_id: str
    status: OrderStatus
    decision: str
    reviewer: str
    decided_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    kb_chunks: int
    kb_docs: int
    backends: dict[str, str]
    warnings: list[str] = Field(default_factory=list)
