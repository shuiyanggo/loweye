"""生成层：把检索到的条款与检测结果合成诊断结论。

两种后端都遵守同一份「不许无依据发挥」的约束：
- template        从条款里抽取可执行句，动作与条款一一绑定（结构上不可能幻觉）
- openai_compat   调用大模型，但只允许在给定条款范围内作答，输出必须带条款号；
                  超时/失败/引用不合法一律降级回 template
"""

from .base import DiagnosisContext, GeneratedDiagnosis, Generator, build_generator
from .template import TemplateGenerator
from .openai_compat import OpenAICompatGenerator

__all__ = [
    "Generator",
    "GeneratedDiagnosis",
    "DiagnosisContext",
    "TemplateGenerator",
    "OpenAICompatGenerator",
    "build_generator",
]
