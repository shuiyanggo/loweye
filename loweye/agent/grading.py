"""分级与派单判定：确定性规则表，不交给大模型。

为什么这个决定要在答辩时讲清楚：
分级结果直接决定派给谁、多久之内必须处理（SLA）。这类判断一旦交给生成式模型，
同一条缺陷两次运行可能给出不同等级，既无法审计，也无法向运维班组解释。
所以规则表是 YAML 配置，可被人直接阅读和修改，判定过程逐条可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Severity


@dataclass
class Grading:
    cls: str
    label: str
    severity: Severity
    assignee_role: str
    sla_hours: int
    matched: bool
    keywords: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # 转人工复核的原因
    normal: bool = False  # 规则表标了 normal: true 的类别（例：未见缺陷的绝缘子）→ 不出工单

    @property
    def needs_human_review(self) -> bool:
        return bool(self.reasons)

    def to_dict(self) -> dict:
        return {
            "cls": self.cls,
            "label": self.label,
            "severity": self.severity,
            "assignee_role": self.assignee_role,
            "sla_hours": self.sla_hours,
            "matched": self.matched,
            "keywords": self.keywords,
            "review_reasons": self.reasons,
            "normal": self.normal,
        }


class GradingTable:
    def __init__(self, routes: dict) -> None:
        self.routes = routes or {}
        self.defaults = self.routes.get("defaults", {}) or {}
        self.rules: dict = self.routes.get("rules", {}) or {}
        self.human_review = self.routes.get("human_review", {}) or {}

    @property
    def known_classes(self) -> list[str]:
        return sorted(self.rules.keys())

    def label_for(self, cls: str) -> str:
        rule = self.rules.get(cls)
        if rule and rule.get("label"):
            return str(rule["label"])
        return cls

    def keywords_for(self, cls: str) -> list[str]:
        rule = self.rules.get(cls) or {}
        return [str(k) for k in (rule.get("keywords") or [])]

    def is_normal(self, cls: str) -> bool:
        """规则表里标了 `normal: true` 的类别：出结论，但不出工单。

        为什么需要这个开关：真实权重必然会把「健康的绝缘子」也检出来（一张图里几十串），
        若每个都生成工单，工单队列会被正常设备淹掉，真正的缺陷反而看不见。
        """
        rule = self.rules.get(cls) or {}
        return bool(rule.get("normal", False))

    def build_query(self, cls: str, *, label: str | None = None) -> str:
        """用类别标签 + 规则表关键词拼检索 query。

        关键词来自规则表而不是模型自由生成，保证检索入口也是确定性的。
        """
        parts = [label or self.label_for(cls)]
        parts.extend(self.keywords_for(cls))
        parts.append("处置")
        seen: list[str] = []
        for p in parts:
            if p and p not in seen:
                seen.append(p)
        return " ".join(seen)

    def grade(
        self,
        cls: str,
        *,
        confidence: float,
        grounded: bool,
        human_review_threshold: float = 0.60,
    ) -> Grading:
        rule = self.rules.get(cls)
        matched = rule is not None
        rule = rule or {}

        severity: Severity = str(rule.get("severity", self.defaults.get("severity", "P3")))  # type: ignore[assignment]
        assignee = str(rule.get("assignee_role", self.defaults.get("assignee_role", "运维班组")))
        sla = int(rule.get("sla_hours", self.defaults.get("sla_hours", 72)))

        reasons: list[str] = []
        if self.human_review.get("on_unknown_class", True) and not matched:
            reasons.append(f"检测类别 {cls} 不在分级规则表中")
        if self.human_review.get("on_low_confidence", True) and confidence < human_review_threshold:
            reasons.append(f"检测置信度 {confidence:.2f} 低于阈值 {human_review_threshold:.2f}")
        if self.human_review.get("on_no_citation", True) and not grounded:
            reasons.append("知识库未检索到条款依据")
        sev_list = [str(s) for s in (self.human_review.get("on_severity_in") or [])]
        if severity in sev_list:
            reasons.append(f"等级 {severity} 属强制人工确认范围")

        return Grading(
            cls=cls,
            label=self.label_for(cls),
            severity=severity,
            assignee_role=assignee,
            sla_hours=sla,
            matched=matched,
            keywords=self.keywords_for(cls),
            reasons=reasons,
            normal=self.is_normal(cls),
        )
