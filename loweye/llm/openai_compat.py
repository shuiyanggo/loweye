"""OpenAI 兼容大模型生成器（可选增强）。

约束（方案 v2 M6 的「幻觉率」指标就靠这些约束压下来）：
1. 只允许依据随 prompt 一起给出的条款作答，条款由检索层提供；
2. 输出必须是 JSON，且每条动作必须带 chunk_id；
3. 输出的 chunk_id 若不在给定集合里 → 判为引用不合法 → 整条降级回模板生成；
4. 超时、HTTP 失败、JSON 解析失败 → 同样降级。

也就是说：大模型在这里是「把条款改写成人话」的润色器，不是事实来源。
"""

from __future__ import annotations

import json

import httpx

from .base import DiagnosisContext, GeneratedDiagnosis

SYSTEM_PROMPT = (
    "你是电力与农业巡检的规程解读助手。你只能依据用户提供的【规程条款】作答，"
    "不得引入条款之外的知识，不得编造条款号。"
    "输出必须是 JSON，形如："
    '{"finding": "现象与定级依据", "actions": [{"text": "处置动作", "chunk_id": "引用的条款ID"}]}。'
    "若条款不足以支撑处置动作，actions 返回空数组。"
)


class OpenAICompatGenerator:
    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float = 20.0,
        max_retries: int = 1,
    ) -> None:
        if not base_url or not model:
            raise ValueError("openai_compat 需要同时提供 base_url 与 model")
        if not api_key:
            raise ValueError("未找到 API Key（请设置配置项 llm.api_key_env 指定的环境变量）")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _payload(self, ctx: DiagnosisContext) -> dict:
        clauses = "\n".join(
            f"[{s.clause.chunk_id}] {s.clause.doc_title} {s.clause.clause_no}：{s.clause.text}"
            for s in ctx.clauses
        )
        user = (
            f"场景：{ctx.scene}\n"
            f"缺陷类别：{ctx.label}（代码 {ctx.cls}）\n"
            f"检出数量：{ctx.n_detections}，最高置信度：{ctx.confidence:.2f}\n"
            f"系统初判等级：{ctx.severity}\n\n"
            f"【规程条款】\n{clauses}\n\n"
            "请给出 finding 与 actions。每条 action 必须带 chunk_id。"
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

    def generate(self, ctx: DiagnosisContext) -> GeneratedDiagnosis:
        from .template import TemplateGenerator

        fallback = TemplateGenerator()
        valid_ids = {s.clause.chunk_id for s in ctx.clauses}
        last_error = ""

        for _ in range(max(1, self.max_retries + 1)):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=self._payload(ctx),
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(content)
                actions_raw = data.get("actions", []) or []
                actions: list[str] = []
                citation_ids: list[str] = []
                for item in actions_raw:
                    chunk_id = str(item.get("chunk_id", ""))
                    if chunk_id not in valid_ids:
                        raise ValueError(f"引用不合法：chunk_id={chunk_id} 不在检索结果中")
                    actions.append(str(item.get("text", "")).strip())
                    citation_ids.append(chunk_id)
                if not actions:
                    raise ValueError("模型未给出任何有依据的动作")
                return GeneratedDiagnosis(
                    finding=str(data.get("finding", "")).strip() or fallback.generate(ctx).finding,
                    actions=actions,
                    generator=self.name,
                    citation_ids=citation_ids,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        result = fallback.generate(ctx)
        result.generator = f"{self.name}→{fallback.name}"
        result.degraded_reason = f"大模型生成失败（{last_error}），已降级为模板生成"
        return result
