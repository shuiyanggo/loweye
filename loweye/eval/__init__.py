"""M6 评估包：金标集、指标、对照基线、评估执行器。

对外只暴露四样东西，避免调用方到处 import 内部模块：
    load_goldset / Evaluator / render_markdown / 指标函数（见 metrics.py）
"""

from .baselines import ArmAnswer, arm_a_detection_only, arm_b_direct_answer
from .goldset import GoldItem, GoldSet, RefusalItem, load_goldset
from .metrics import (
    DetBox,
    average_precision,
    detection_report,
    iou,
    mrr,
    percentile,
    rate,
    recall_at_k,
)
from .runner import EvalReport, Evaluator, render_markdown

__all__ = [
    "ArmAnswer",
    "DetBox",
    "EvalReport",
    "Evaluator",
    "GoldItem",
    "GoldSet",
    "RefusalItem",
    "arm_a_detection_only",
    "arm_b_direct_answer",
    "average_precision",
    "detection_report",
    "iou",
    "load_goldset",
    "mrr",
    "percentile",
    "rate",
    "recall_at_k",
    "render_markdown",
]
