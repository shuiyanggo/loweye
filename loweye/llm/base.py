"""生成器接口与降级策略。

方案 v2 第 4 章 M3 的降级链在这里落地：
    LLM 超时 → 模板生成
    API 失败 → 模板生成
    引用不合法（引了不存在的条款号）→ 判为幻觉，丢弃该条输出并降级
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..config import Config
from ..schemas import ScoredClause


@dataclass
class DiagnosisContext:
    """生成一条诊断所需的全部输入。条款是唯一的事实来源。"""

    cls: str
    label: str
    severity: str
    n_detections: int
    confidence: float
    clauses: list[ScoredClause] = field(default_factory=list)
    scene: str = "power"
    image_id: str = ""
    # 类别词表（来自 config/routes.yaml 的 keywords）：用来判断某条条款的
    # 条件从句说的是不是当前这条缺陷，避免把别的情形的要求当成处置动作。
    keywords: list[str] = field(default_factory=list)


@dataclass
class GeneratedDiagnosis:
    finding: str
    actions: list[str]
    generator: str
    citation_ids: list[str] = field(default_factory=list)  # 动作实际引用的 chunk_id
    # 检索到并过了阈值、但未抽出处置动作的条款（判定句、响应分级等）：
    # 作为「参考条款」呈现，让分级结论也有出处，而不是凭空出现。
    reference_ids: list[str] = field(default_factory=list)
    degraded_reason: str | None = None


@runtime_checkable
class Generator(Protocol):
    name: str

    def generate(self, ctx: DiagnosisContext) -> GeneratedDiagnosis:
        ...


def build_generator(cfg: Config) -> tuple[Generator, str | None]:
    """返回 (生成器, 降级原因)。"""
    provider = str(cfg.get("llm.provider", "template")).lower()
    if provider in ("openai_compat", "openai", "cloud", "vlm"):
        from .openai_compat import OpenAICompatGenerator

        try:
            gen = OpenAICompatGenerator(
                base_url=str(cfg.require("llm.base_url")),
                api_key=cfg.env_for("llm.api_key_env"),
                model=str(cfg.require("llm.model")),
                timeout_s=float(cfg.get("llm.timeout_s", 20)),
                max_retries=int(cfg.get("llm.max_retries", 1)),
            )
            return gen, None
        except Exception as exc:
            from .template import TemplateGenerator

            return (
                TemplateGenerator(),
                f"大模型生成器不可用（{type(exc).__name__}: {exc}），已降级为模板生成",
            )
    from .template import TemplateGenerator

    return TemplateGenerator(), None
