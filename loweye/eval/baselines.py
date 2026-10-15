"""对照实验的三个 arm（M6 评估的「不比不知道」部分）。

为什么要做这三组对照：如果只报「本系统给出了 4 条带条款号的建议」，
评委没法判断这是不是随便一个多模态大模型也能做到的事。三组跑同一批图、同一批检出：

    A 纯检测：只报「是什么缺陷、几处、置信度、等级」——不检索、不生成；
    B 直答助手：不查知识库，直接凭模型自身知识给处置建议（本项目里默认用
      脚本化代理 scripted，配置了大模型 API 后可切到 api 模式真跑）；
    C 本系统：检索 → 融合 → 重排 → 抽句 → 引用 → 分级派单 → 可溯源交付。

诚实边界（必须写进报告）：
- arm B 的 scripted 模式是**代理基线**，它模拟的是「直答型助手会给出像模像样但
  不可追溯的建议」这一形态，不是某个具体模型的实测输出；要拿真实模型数字，
  需在 config/app.yaml 填好 llm.base_url 与环境变量后跑 `--arm-b api`。
- 「建议可执行率」三组用同一判据（loweye.llm.template.is_actionable），
  所以它只说明**输出形态**，不构成质量优势；真正的差别在可溯源率与引用可核验率，
  而这两项对 A、B 是结构性不可能（它们根本没有引用）。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

from ..config import Config
from ..llm.template import is_actionable


@dataclass
class ArmAnswer:
    """一个缺陷类别在一组 arm 下的产出（同一批检出图汇总）。"""

    arm: str
    cls: str
    label: str
    n_detections: int
    confidence: float
    severity: str
    text: str
    actions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    source: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "cls": self.cls,
            "label": self.label,
            "n_detections": self.n_detections,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "text": self.text,
            "actions": list(self.actions),
            "citations": list(self.citations),
            "source": self.source,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


# 脚本化代理基线的应答模板：像建议、读起来合理、但没有任何条款依据，
# 也不含可核验的时限数字（这正是「直答」形态的典型特征：说得出方向，说不清出处）。
SCRIPTED_ADVICE: dict[str, str] = {
    "insulator_broken": "绝缘子自爆通常与长期机械荷载、温差变化或产品质量有关，建议尽快安排人员登塔核实，并对同批次绝缘子开展一次普查，视情况申请停电更换。",
    "insulator_polluted": "绝缘子污秽会降低外绝缘水平，建议尽快安排清扫或喷涂防污闪涂料，重污区适当缩短清扫周期。",
    "fitting_corrosion": "金具锈蚀发展较慢但会影响导电性能，建议安排除锈并涂覆防腐涂料，必要时测量接触电阻确认是否需要更换。",
    "conductor_strand_break": "导地线断股会降低机械强度，建议尽快安排人员到现场核查断股比例，必要时加装护线条或安排停电更换。",
    "tower_tilt": "杆塔倾斜多与基础沉降或外力破坏有关，建议尽快安排测量并持续观测，倾斜持续发展时安排停电加固纠偏。",
    "foreign_object": "异物悬挂存在放电风险，建议尽快安排清除，清除前加强监视，若已造成放电痕迹应申请停电处理。",
    "vegetation_overreach": "树障净空不足可能导致放电，建议与树木产权方协商后安排砍伐或修剪，速生树种应缩短复核周期。",
    "crop_disease": "叶部病斑建议尽快安排田间核实，确认病害类型后制定防治方案并及时施药，防治后注意复查。",
    "crop_pest": "虫害建议尽快安排防治，优先采用生物防治手段，减少化学农药使用，防治后继续监测虫口密度。",
    "fire_smoke": "发现火情应立即上报并保持监视，必要时申请线路转检修，明火扑灭后仍需继续监视一段时间。",
}


def arm_a_detection_only(items: list[dict], *, grading: dict[str, tuple[str, str]]) -> list[ArmAnswer]:
    """arm A：只输出检测与分级结果。

    items 元素形如 {"cls", "n", "confidence"}；grading 为 cls -> (label, severity)。
    """
    out: list[ArmAnswer] = []
    for it in items:
        cls = it["cls"]
        label, severity = grading.get(cls, (cls, "P3"))
        out.append(
            ArmAnswer(
                arm="A",
                cls=cls,
                label=label,
                n_detections=int(it["n"]),
                confidence=float(it["confidence"]),
                severity=severity,
                text=f"检出{label} {int(it['n'])} 处，最高置信度 {float(it['confidence']):.2f}。",
                actions=[],
                citations=[],
                source="detect-only",
            )
        )
    return out


def arm_b_direct_answer(
    items: list[dict],
    *,
    grading: dict[str, tuple[str, str]],
    mode: str = "scripted",
    cfg: Config | None = None,
) -> tuple[list[ArmAnswer], str]:
    """arm B：不查知识库直接给建议。返回 (答案, 说明)。

    mode="scripted"：脚本化代理基线（默认，离线可跑、可复现）；
    mode="api"：真调大模型（需 config llm.base_url + llm.api_key_env 指向的环境变量）。
    """
    grading = grading or {}
    if mode == "api":
        answers, note = _arm_b_via_api(items, grading=grading, cfg=cfg)
        if answers is not None:
            return answers, note
        # API 不可用则明确退回代理基线，并在说明里写明，绝不静默冒充真实模型结果
        proxy, _ = arm_b_direct_answer(items, grading=grading, mode="scripted")
        return proxy, f"{note}；已退回脚本化代理基线（数字不代表真实模型表现）"

    out: list[ArmAnswer] = []
    for it in items:
        cls = it["cls"]
        label, severity = grading.get(cls, (cls, "P3"))
        advice = SCRIPTED_ADVICE.get(cls, f"检出{label}，建议尽快安排人员现场核实并视情况处置。")
        actions = [advice] if is_actionable(advice) else []
        out.append(
            ArmAnswer(
                arm="B",
                cls=cls,
                label=label,
                n_detections=int(it["n"]),
                confidence=float(it["confidence"]),
                severity=severity,
                text=f"检出{label} {int(it['n'])} 处，最高置信度 {float(it['confidence']):.2f}。{advice}",
                actions=actions,
                citations=[],
                source="direct-answer(scripted proxy)",
            )
        )
    return out, "arm B 使用脚本化代理基线（模拟直答形态，非具体模型实测）"


def _arm_b_via_api(
    items: list[dict],
    *,
    grading: dict[str, tuple[str, str]],
    cfg: Config | None,
) -> tuple[list[ArmAnswer] | None, str]:
    if cfg is None:
        return None, "未提供配置，无法调用大模型"
    base_url = cfg.get("llm.base_url")
    model = cfg.get("llm.model")
    key_env = cfg.get("llm.api_key_env", "LOWEYE_LLM_API_KEY")
    api_key = os.environ.get(str(key_env), "")
    if not base_url or not model:
        return None, "未配置 llm.base_url / llm.model，直答基线无法真跑大模型"
    if not api_key:
        return None, f"环境变量 {key_env} 未设置，直答基线无法真跑大模型"

    out: list[ArmAnswer] = []
    timeout = float(cfg.get("llm.timeout_s", 20.0))
    for it in items:
        cls = it["cls"]
        label, severity = grading.get(cls, (cls, "P3"))
        prompt = (
            f"无人机巡检发现：{label}（类别代码 {cls}），共 {int(it['n'])} 处，"
            f"最高置信度 {float(it['confidence']):.2f}。请给出处置建议，直接输出建议正文。"
        )
        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{str(base_url).rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            text = str(resp.json()["choices"][0]["message"]["content"]).strip()
        except Exception as exc:  # noqa: BLE001 - 基线失败不应中断评估
            return None, f"直答基线调用失败（{type(exc).__name__}: {exc}）"
        elapsed = (time.perf_counter() - started) * 1000
        out.append(
            ArmAnswer(
                arm="B",
                cls=cls,
                label=label,
                n_detections=int(it["n"]),
                confidence=float(it["confidence"]),
                severity=severity,
                text=text,
                actions=[text] if is_actionable(text) else [],
                citations=[],
                source=f"direct-answer({model})",
                elapsed_ms=elapsed,
            )
        )
    return out, f"arm B 真跑大模型完成（model={model}）"
