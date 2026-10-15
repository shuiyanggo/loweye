"""模板生成器：从检索到的条款里抽取可执行句，动作与条款一一绑定。

这是默认后端，也是降级目标。选择它作为默认的理由：
1. 巡检场景的产出是要被执行的工单，宁可句子朴素，也不能出现没有依据的处置动作；
2. 结构上不可能幻觉 —— 每条动作都能指回它来自哪一条条款；
3. 零依赖、毫秒级、断网可用，现场演示不会因为 API 抖动而失败。

它不做的事也写清楚：不生成需要跨条款推理的复杂判断，那类问题由人在回路解决。
"""

from __future__ import annotations

import re

from .base import DiagnosisContext, GeneratedDiagnosis
from ..text import is_scope_statement

# 可执行句的判定：出现「要求类情态词」或「处置类动词」即视为可执行要求。
#
# 为什么不用一份固定的动词白名单：中文规程里同一件事有多种写法
# （「应立即上报」/「应在 48 小时内安排…核实」/「需限期整改」），
# 白名单必然漏。这里改为「情态词 or 处置动词」的双通道判定，覆盖更全，
# 且仍然只从检索到的条款里取句 —— 不会凭空生成动作。
_MODAL_RE = re.compile(
    r"(?:应当"
    # 「应」要排除掉它在别的词里的用法：应用/应对/相应/适应/反应/响应/供应/感应……
    # 前面的字用后顾断言挡（响应、供应、相应），后面的字用前瞻断言挡（应用、应对）。
    r"|(?<![响供相适反效感对])应(?!(?:用|对|相|邀|届|试|征|声|和|承|呼|验|聘|收|战|援|酬|市|有|急|变|力|效|感|付|反|接|答))"
    r"|宜(?!人)|须|必须|严禁|不得|禁止|立即|及时|限期|需要"
    # 「按」只在与执行口径连用时才算要求：「按季度盘点」是常规条款，不是处置动作。
    r"|按(?=[^，。；]{0,12}(?:执行|办理|规定|要求|标准|流程|方案|程序|处理|照)))"
)
_DISPOSAL_RE = re.compile(
    r"(更换|清扫|检修|消缺|处理|加固|整改|上报|复核|复测|监测|巡视|归档|复查|施药"
    r"|除锈|停电|隔离|清除|核实|安排|申请|复检|清理|砍伐|修剪|送检|取样)"
)
# 定义性/适用范围类句子不是处置动作。这类句子也常含「应」「处置」等字，
# 只靠词表会误抽，所以单列一条排除规则（例：本指引适用于……处置建议生成）。
_DEFINITIONAL_RE = re.compile(r"(适用于|用于规范|规定了|本指引|本预案|本导则|本文档|是指|的定义)")
# 判定性句子（……的，判定为严重缺陷）是分级依据，不是操作要求
_CLASSIFY_RE = re.compile(r"(判定为|确定为|定性为|归类为)[^。；]{0,16}(缺陷|等级|级别|病害|虫害|隐患)")
_SENT_SPLIT_RE = re.compile(r"(?<=[。；;！？!?])")
_CLAUSE_MARK_RE = re.compile(r"^[（(]?\d+[）)]?[、.．]?")

# 条件分支识别。
#
# 规程里大量出现「某情形…的，应…」的写法：要求只在条件成立时适用。
# 如果条件说的不是当前这条缺陷，把它当成处置动作就会让工单越权——
# 例如「金具锈蚀」的工单不该照搬 3.1.3「金具出现裂纹、变形或明显磨损的，
# 应按照危急缺陷流程处理，立即申请停电更换」：那是一次不必要的停电。
# 所以：条件片段与当前类别词表无交集时，该句不抽为动作，只在结论里点名。
_CONDITION_SPLIT_RE = re.compile(
    r"^(.{2,60}?)(?:的)?[，,]\s*(?:应|宜|须|必须|立即|及时|限期|需要|按|按照)"
)
_CONDITION_HEAD_RE = re.compile(r"(?:出现|存在|发生|发现|处于|导致|造成|引发|产生)")

MAX_ACTIONS = 4
MIN_ACTION_CHARS = 8
MAX_ACTION_CHARS = 90
MAX_FALLBACK_CITATIONS = 3
MAX_CONDITION_NOTES = 2


def condition_prefix(sentence: str) -> str | None:
    """句子若是「某情形…的，应…」的条件句，返回其中「情形」片段，否则 None。"""
    m = _CONDITION_SPLIT_RE.match(sentence)
    if not m:
        return None
    prefix = m.group(1)
    head = _CONDITION_HEAD_RE.search(prefix)
    if not head:
        return None
    return prefix[head.end():].strip("的 ") or None


def _shingles(text: str) -> set[str]:
    clean = re.sub(r"[\s、，,。；;：:（）()「」【】]", "", text)
    return {clean[i : i + 2] for i in range(max(len(clean) - 1, 0))}


def condition_applies(condition: str, ctx: DiagnosisContext) -> bool:
    """条件片段是否属于当前缺陷类别（用二字片段与类别词表比对）。"""
    class_shards: set[str] = set()
    for word in [ctx.label, *ctx.keywords]:
        class_shards |= _shingles(word)
    return bool(class_shards & _shingles(condition))


def is_actionable(sentence: str) -> bool:
    """判断一句话是否构成「可执行要求」。

    公开这个判据是为了让评估模块（loweye/eval）用同一把尺子给对照组打分：
    如果评估脚本自己复制一份正则，两份规则迟早会悄悄漂移，
    到时候「建议可执行率」这个数字就解释不清了。
    """
    if _DEFINITIONAL_RE.search(sentence):
        return False
    if _CLASSIFY_RE.search(sentence):
        return False
    return bool(_MODAL_RE.search(sentence) or _DISPOSAL_RE.search(sentence))


# 兼容旧名（内部调用点无需改动）
_is_actionable = is_actionable


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for raw in _SENT_SPLIT_RE.split(text):
        sentence = raw.strip()
        if sentence:
            out.append(sentence)
    return out


def _clean_action(sentence: str) -> str:
    s = _CLAUSE_MARK_RE.sub("", sentence.strip())
    s = re.sub(r"\s+", " ", s)
    if len(s) > MAX_ACTION_CHARS:
        s = s[:MAX_ACTION_CHARS].rstrip("，,、；;") + "…"
    return s


class TemplateGenerator:
    name = "template"

    def generate(self, ctx: DiagnosisContext) -> GeneratedDiagnosis:
        if not ctx.clauses:
            # 无依据就明说无依据，不给任何处置动作 —— 这是拒答，不是兜底编造
            return GeneratedDiagnosis(
                finding=f"检出{ctx.label} {ctx.n_detections} 处，但知识库未检索到对应的判定依据，"
                "无法给出处置建议。",
                actions=[],
                generator=self.name,
                citation_ids=[],
                degraded_reason="无检索依据：知识库未覆盖该缺陷类型",
            )

        best = ctx.clauses[0]
        finding = (
            f"检出{ctx.label} {ctx.n_detections} 处，最高置信度 {ctx.confidence:.2f}。"
            f"依据《{best.clause.doc_title}》{best.clause.clause_no}"
            f"（相关度 {best.rerank:.2f}）判定为 {ctx.severity} 级。"
        )

        actions: list[str] = []
        citation_ids: list[str] = []
        skipped: list[str] = []
        seen: set[str] = set()
        for scored in ctx.clauses:
            for sentence in _split_sentences(scored.clause.text):
                if len(sentence) < MIN_ACTION_CHARS:
                    continue
                if not _is_actionable(sentence):
                    continue
                condition = condition_prefix(sentence)
                if condition and not condition_applies(condition, ctx):
                    # 讲的是别的缺陷情形（例：金具锈蚀章节里的裂纹/变形分支），
                    # 不抽为动作，只在结论里点名，人可自行翻原文。
                    note = f"{scored.clause.clause_no}（{condition}）"
                    if note not in skipped:
                        skipped.append(note)
                    continue
                action = _clean_action(sentence)
                key = action[:24]
                if key in seen:
                    continue
                seen.add(key)
                # 动作尾部标注来源条款号，前端据此可点击展开原文
                actions.append(f"{action}〔{scored.clause.clause_no}〕")
                if scored.clause.chunk_id not in citation_ids:
                    citation_ids.append(scored.clause.chunk_id)
                if len(actions) >= MAX_ACTIONS:
                    break
            if len(actions) >= MAX_ACTIONS:
                break

        if not actions:
            if skipped:
                # 有可执行句，但说的都是别的情形：如实说明，不硬塞动作，
                # 也不说成「条款是描述性内容」——那是另一回事。
                actions = [
                    f"该缺陷类型检索到的条款均为其他情形的条件分支"
                    f"（{'、'.join(skipped[:MAX_CONDITION_NOTES])}），未提取到可直接执行的处置要求，"
                    f"建议人工确认后再派单〔{ctx.clauses[0].clause.clause_no}〕"
                ]
            else:
                actions = [
                    f"知识库中该条款为描述性内容，未包含可直接执行的处置要求；"
                    f"建议按《{best.clause.doc_title}》{best.clause.clause_no}人工确认后再派单"
                    f"〔{best.clause.clause_no}〕"
                ]
            # 条款里确实没有可执行要求时如实说明，并把检索到的条款都列出来
            # （而不是只列第一条），让人能自己判断该看哪一条。
            citation_ids = [s.clause.chunk_id for s in ctx.clauses[:MAX_FALLBACK_CITATIONS]]

        # 检索到、过了阈值，但没抽出处置动作的条款：不当作依据，也不让它消失。
        # 例：应急预案的「一级响应／二级响应」是分级判据，判据本身要能被人看到，
        # 否则工单凭 P1 派单却指不出是哪一条说的。
        # 但「适用范围／术语和定义」这类元信息条款不进参考列 —— 它们只是文档头，
        # 列在工单里等于用噪声充数。
        reference_ids = [
            s.clause.chunk_id
            for s in ctx.clauses
            if s.clause.chunk_id not in citation_ids and not is_scope_statement(s.clause.text)
        ]

        if skipped:
            finding += (
                f" 另有 {len(skipped)} 条属其他情形的条件分支，未纳入处置动作，仅作参考："
                f"{'、'.join(skipped[:MAX_CONDITION_NOTES])}。"
            )

        return GeneratedDiagnosis(
            finding=finding,
            actions=actions,
            generator=self.name,
            citation_ids=citation_ids,
            reference_ids=reference_ids,
        )
