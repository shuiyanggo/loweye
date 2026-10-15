"""M6 评估入口：跑金标集、拒答集、时延与三组对照，产出报告。

用法：
    .venv\\Scripts\\python.exe -m loweye.cli.evaluate
    .venv\\Scripts\\python.exe -m loweye.cli.evaluate --report out/eval_report.md
    .venv\\Scripts\\python.exe -m loweye.cli.evaluate --json out/eval_report.json
    .venv\\Scripts\\python.exe -m loweye.cli.evaluate --arm-b api     # 直答基线真调大模型

退出码：0 = 评估完成；1 = 出现「拒答漏判」这类需要人看一眼的失败项。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..eval.runner import Evaluator, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="低空之眼 M6 评估")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--report", default="out/eval_report.md", help="Markdown 报告输出路径")
    parser.add_argument("--json", dest="json_path", default="out/eval_report.json", help="JSON 结果输出路径")
    parser.add_argument(
        "--arm-b",
        dest="arm_b_mode",
        default="scripted",
        choices=["scripted", "api"],
        help="arm B 直答基线：scripted=脚本化代理（默认，离线可复现）；api=真调大模型",
    )
    parser.add_argument("--search-repeat", type=int, default=3, help="检索时延重复轮数")
    parser.add_argument("--analyze-repeat", type=int, default=2, help="端到端时延重复轮数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = Evaluator(cfg)
    report = evaluator.run(
        arm_b_mode=args.arm_b_mode,
        search_repeat=args.search_repeat,
        analyze_repeat=args.analyze_repeat,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(report), encoding="utf-8")

    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    ret = report.retrieval
    print("=" * 68)
    print("低空之眼 · M6 评估完成")
    print("=" * 68)
    print(f"知识库            : {report.kb['chunks']} 块 / {report.kb['docs']} 文档")
    print(
        f"检索 Recall@1/3/5 : {ret['hybrid_rerank']['recall@1']:.0%} / "
        f"{ret['hybrid_rerank']['recall@3']:.0%} / {ret['hybrid_rerank']['recall@5']:.0%}"
        f"（MRR {ret['hybrid_rerank']['mrr']:.3f}）"
    )
    print(
        "五种配置 MRR      : "
        + "、".join(
            f"{k}={ret[k]['mrr']:.3f}"
            for k in [
                "bm25_only",
                "vector_only",
                "hybrid_rrf",
                "hybrid_rerank",
                "hybrid_rerank_no_lexicon",
            ]
            if k in ret
        )
    )
    groups = ret.get("by_group") or {}
    if groups:
        print(
            "分组 Recall@1     : "
            + "、".join(
                f"{g}={v['hybrid_rerank']['recall@1']:.0%}(n={v['n_questions']})"
                for g, v in sorted(groups.items())
            )
        )
    print(f"越界拒答率        : {report.refusal['refusal_rate']:.0%}（漏判 {report.refusal['false_accept'] or '无'}）")
    if report.latency.get("analyze_ms"):
        print(
            f"时延 P50/P95      : 检索 {report.latency['search_ms']['p50']}/"
            f"{report.latency['search_ms']['p95']} ms；端到端 {report.latency['analyze_ms']['p50']}/"
            f"{report.latency['analyze_ms']['p95']} ms"
        )
    a = report.arms
    print(
        "可溯源率 A/B/C    : "
        f"{a.traceable_rate['A']:.0%} / {a.traceable_rate['B']:.0%} / {a.traceable_rate['C']:.0%}"
        f"（引用可核验率 {a.citation_verified_rate:.0%}，核验 {a.citation_checked} 条）"
    )
    print(f"报告              : {report_path}")
    print(f"JSON              : {json_path}")

    return 1 if report.refusal["false_accept"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
